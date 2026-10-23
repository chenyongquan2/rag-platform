"""Prove from the corpus that no single page answers a hard question (spec EVAL-2, AC3).

For every fact of a hard question the checker searches the whole corpus:

- the evidence text is on the fact's page (whitespace-insensitive);
- the key pattern matches that page and no page outside the declared copies (`also_on`);
- the loose patterns (inflection-aware anchor and fact pairs of `rag.eval.absence`, another way
  of writing the statement) match that page;
- no page matches the loose patterns of all facts together, i.e. no single page can state
  everything the question needs, in these spellings.

A word search cannot prove that a page never says something in other words; it is the same
limit the no-answer checks of P3-03 have.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import ServiceSettings, repo_path
from rag.eval.absence import WINDOW, concept_pages, lowered
from rag.eval.hard.spec import AmbiguousMeta, Fact, HardMeta
from rag.eval.questions import LabeledQuestion


@dataclass
class QuestionFinding:
    qid: str
    problems: list[str] = field(default_factory=list)
    searched_pages: int = 0
    key_pages: dict[str, int] = field(default_factory=dict)
    """Per gold page: how many corpus pages the key pattern matches (itself and its copies)."""
    loose_pages: dict[str, int] = field(default_factory=dict)
    co_mentions: list[str] = field(default_factory=list)
    """Pages (gold pages included) that contain every term, or every subject word, of all facts."""

    @property
    def ok(self) -> bool:
        return not self.problems


def corpus_file(data_dir: str | Path | None = None) -> Path | None:
    root = repo_path(data_dir or ServiceSettings().hf_data_dir)
    return next(iter(root.glob("m-ric/huggingface_doc/*/default/train/0000.parquet")), None)


def load_corpus(data_dir: str | Path | None = None) -> dict[str, str] | None:
    """doc_id -> page text of the downloaded Hugging Face corpus; None when it is not there."""
    file = corpus_file(data_dir)
    if file is None:
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(file, columns=["text", "source"])
    return {
        f"hf_docs:{s}": t
        for s, t in zip(table["source"].to_pylist(), table["text"].to_pylist(), strict=True)
    }


def _words(text: str) -> str:
    return " ".join(text.split())


def corpus_report(
    questions: list[LabeledQuestion], corpus: Mapping[str, str]
) -> list[QuestionFinding]:
    """One finding per question; `problems` is empty when every check passes."""
    docs = dict(corpus)
    squeezed: dict[str, str] = {}
    low = lowered(docs)
    cache = PageCache(low)
    out: list[QuestionFinding] = []
    for q in questions:
        finding = QuestionFinding(qid=q.qid, searched_pages=len(docs))
        if q.meta.get("type") == "hard":
            _check_hard(q, docs, low, squeezed, cache, finding)
        else:
            _check_ambiguous(q, docs, squeezed, finding)
        out.append(finding)
    return out


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "over",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "has",
        "have",
        "its",
        "into",
        "than",
        "then",
        "which",
        "what",
        "how",
        "many",
        "much",
        "more",
        "most",
        "tokens",
        "token",
        "model",
        "models",
        "page",
        "pages",
        "when",
        "does",
        "did",
        "your",
        "you",
    ]
)


def _words_of(phrases: list[str]) -> set[str]:
    out: set[str] = set()
    for phrase in phrases:
        for w in _TOKEN.findall(phrase.lower()):
            if w not in _STOP and (len(w) >= 3 or (w.isdigit() and len(w) >= 2)):
                out.add(w)
    return out


def fact_terms(fact: Fact) -> set[str]:
    """The words that name a fact: those of its loose anchor and fact phrases and its `subjects`
    (lower case, at least three letters or a number of two digits or more; common words left
    out)."""
    return _words_of([str(x) for pair in fact.loose for x in pair[:2]] + list(fact.subjects))


def subject_phrases(fact: Fact) -> list[str]:
    """What the fact is about, as phrases (an entity such as Mask2Former or a distinctive value
    such as 25,000), normalised like page text. A fact without `subjects` is left out of that
    search."""
    return [_normalised(x) for x in fact.subjects]


def _normalised(text: str) -> str:
    return " ".join(re.sub(r"[-_]+", " ", text.lower()).split())


class PageCache:
    """Word sets and normalised texts of the pages, each made once for a whole report."""

    def __init__(self, low: Mapping[str, str]) -> None:
        self._low = low
        self._words: dict[str, set[str]] = {}
        self._norm: dict[str, str] = {}

    def words(self, doc: str) -> set[str]:
        if doc not in self._words:
            self._words[doc] = set(_TOKEN.findall(self._low[doc]))
        return self._words[doc]

    def norm(self, doc: str) -> str:
        if doc not in self._norm:
            self._norm[doc] = _normalised(self._low[doc])
        return self._norm[doc]


def co_mentions(meta: HardMeta, cache: PageCache) -> list[str]:
    """Pages that may state several facts at once, found without looking at the wording.

    Two searches, over every page including the gold pages (a gold page that holds the other
    facts as well is a single-page answer):

    - all terms: the page contains every term of every fact, anywhere on the page (a table row
      "Falcon | 40B | ... | 1,000B" is caught although no phrase search finds "Falcon-40B ...
      1 trillion tokens" in it);
    - all subjects: the page contains every `subjects` phrase of every fact (a blog post that
      names Mask2Former and MaskFormer and explains both, in words no listed phrase uses).

    The author reads each page found and either rewrites the question or lists the page in
    `read_and_cleared` with the reason it does not answer."""
    terms = set().union(*(fact_terms(f) for f in meta.facts))
    phrases = [p for f in meta.facts for p in subject_phrases(f)]
    if not terms:
        return []
    out = []
    for doc in cache._low:
        if terms <= cache.words(doc):
            out.append(doc)
        elif phrases:
            norm = cache.norm(doc)
            if all(p in norm for p in phrases):
                out.append(doc)
    return sorted(out)


def _page_words(doc: str, docs: Mapping[str, str], squeezed: dict[str, str]) -> str | None:
    if doc not in docs:
        return None
    if doc not in squeezed:
        squeezed[doc] = _words(docs[doc])
    return squeezed[doc]


def _check_ambiguous(
    q: LabeledQuestion, docs: Mapping[str, str], squeezed: dict[str, str], finding: QuestionFinding
) -> None:
    meta = AmbiguousMeta.model_validate(q.meta)
    for reading in meta.readings:
        page = _page_words(reading.page, docs, squeezed)
        if page is None:
            finding.problems.append(f"reading page {reading.page} is not in the corpus")
        elif _words(reading.evidence) not in page:
            finding.problems.append(f"evidence of a reading is not in {reading.page}")


def _check_hard(
    q: LabeledQuestion,
    docs: Mapping[str, str],
    low: Mapping[str, str],
    squeezed: dict[str, str],
    cache: PageCache,
    finding: QuestionFinding,
) -> None:
    meta = HardMeta.model_validate(q.meta)
    loose_sets: list[set[str]] = []
    for fact in meta.facts:
        loose_sets.append(_check_fact(fact, docs, low, squeezed, finding))
    joint = set.intersection(*loose_sets) if loose_sets else set()
    if joint:
        finding.problems.append(
            f"a single page matches the loose patterns of every fact: {sorted(joint)}"
        )
    found = co_mentions(meta, cache)
    finding.co_mentions = found
    unread = sorted(set(found) - set(meta.read_and_cleared))
    if unread:
        finding.problems.append(
            "pages may state several facts at once (all terms or all subjects of every fact) "
            f"and are not in read_and_cleared: {unread}"
        )
    stale = sorted(set(meta.read_and_cleared) - set(found))
    if stale:
        finding.problems.append(f"read_and_cleared lists pages that are not co-mentions: {stale}")


def _check_fact(
    fact: Fact,
    docs: Mapping[str, str],
    low: Mapping[str, str],
    squeezed: dict[str, str],
    finding: QuestionFinding,
) -> set[str]:
    page = _page_words(fact.page, docs, squeezed)
    if page is None:
        finding.problems.append(f"gold page {fact.page} is not in the corpus")
        return set()
    for text in [fact.evidence, *fact.evidence_also]:
        if _words(text) not in page:
            finding.problems.append(f"evidence is not in {fact.page}: {text[:60]!r}")
    rx = re.compile(fact.key, re.IGNORECASE)
    matched = [doc for doc, text in docs.items() if rx.search(text)]
    finding.key_pages[fact.page] = len(matched)
    if fact.page not in matched:
        finding.problems.append(f"key pattern does not match its own page {fact.page}")
    allowed = {fact.page, *fact.also_on}
    extra = sorted(set(matched) - allowed)
    if extra:
        finding.problems.append(f"key pattern of {fact.page} also matches {extra}")
    for copy in fact.also_on:
        if copy not in matched:
            finding.problems.append(f"declared copy {copy} does not match the key pattern")
    loose: set[str] = set()
    for pair in fact.loose:
        window = int(pair[2]) if len(pair) == 3 else WINDOW
        loose |= set(concept_pages(str(pair[0]), str(pair[1]), low, window=window, is_lowered=True))
    finding.loose_pages[fact.page] = len(loose)
    if fact.page not in loose:
        finding.problems.append(f"loose patterns do not match the gold page {fact.page}")
    return loose
