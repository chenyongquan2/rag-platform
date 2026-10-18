"""Which generated questions are kept (spec P1-05, design s14-2).

A question is dropped when any critique score is below its threshold or missing (HF
rag_evaluation keeps only questions with all three scores >= 4), when the answer is longer than
`max_answer_chars` (HF asserts answers under 300 characters), and when the question copies the
span's wording: its longest run of consecutive words that also appears in the span is longer than
`max_copy_run`. Every reason that applies is kept, so the rejection breakdown counts each.

The copy limit 6 comes from the 65 public HF questions: against their source pages the longest
shared run is 4 words at the median and 7 or more for only 8 of 64 (docs/decisions.md).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from rag.eval.synthetic.prompts import CRITERIA

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORD_TEXT = (
    "a an the of to in on for and or is are was were be been by with what which "
    "how does do did can could should would will from as at that this these those it its why when "
    "where who whom i my me you your we our there their if into than then not no so"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split(" "))


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_groundedness: float = 4
    min_relevance: float = 4
    min_standalone: float = 4
    max_copy_run: int = 6
    max_answer_chars: int = 300


class Overlap(BaseModel):
    longest_run: int
    """Longest run of consecutive question words (lower case, stop words included) in the text."""
    content_recall: float
    """Share of the question's content words (no stop words) that occur in the text."""
    question_words: int


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lexical_overlap(question: str, text: str) -> Overlap:
    q = tokens(question)
    t = tokens(text)
    joined = f" {' '.join(t)} "
    best = 0
    for i in range(len(q)):
        j = i + best + 1
        while j <= len(q) and f" {' '.join(q[i:j])} " in joined:
            best = j - i
            j += 1
    content = [w for w in q if w not in STOPWORDS]
    vocabulary = set(t)
    recall = sum(w in vocabulary for w in content) / len(content) if content else 0.0
    return Overlap(longest_run=best, content_recall=round(recall, 4), question_words=len(q))


def rejection_reasons(
    *,
    question: str | None,
    answer: str | None,
    span: str,
    scores: Mapping[str, float | None],
    config: FilterConfig,
) -> list[str]:
    """Names of every failed check, in a fixed order; empty means the question is kept."""
    if not question or not answer:
        return ["generation_failed"]
    reasons: list[str] = []
    for criterion in CRITERIA:
        score = scores.get(criterion)
        if score is None:
            reasons.append(f"{criterion}_missing")
        elif score < getattr(config, f"min_{criterion}"):
            reasons.append(f"{criterion}_low")
    if lexical_overlap(question, span).longest_run > config.max_copy_run:
        reasons.append("copied_wording")
    if len(answer) > config.max_answer_chars:
        reasons.append("answer_too_long")
    return reasons
