"""Checks that a no-answer question's answer is really absent from the corpus (spec P3-03,
review round 1).

A keyword search for the exact spelling misses facts written another way: "stored for" misses
"storing logs for 30 days", "cpu-upgrade" misses "CPU Upgrade". So every no-answer question
records, next to its own regular expressions (`meta.checks`), `meta.concepts`: pairs of an
anchor phrase and a fact phrase ("logs", "days"). `concept_regex` turns a pair into a pattern
that tolerates inflections (store, stores, stored, storing), hyphen, space and underscore
variants, and either order within `WINDOW` characters; a page that matches is a page that may
state the answer and must be read.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

WINDOW = 150
"""How many characters apart an anchor and a fact may be and still count as one statement."""

_ENDINGS = re.compile(r"(ings|ing|ed|es|e|s)$")


def inflected(word: str) -> str:
    """A pattern for `word` and its endings: the stem of a word longer than four letters, then up
    to four more word characters (store matches stored, storing, storage, stores)."""
    word = word.lower()
    if re.fullmatch(r"\W+|\d+", word):
        return re.escape(word)
    stem = _ENDINGS.sub("", word) if len(word) > 4 else word
    return re.escape(stem) + r"\w{0,4}"


def phrase_regex(phrase: str) -> str:
    """The words of `phrase` inflected and joined by an optional hyphen, space or underscore."""
    words = re.split(r"[\s_-]+", phrase.strip())
    return r"[-_\s]*".join(inflected(w) for w in words if w)


def concept_regex(anchor: str, fact: str, *, window: int = WINDOW) -> re.Pattern[str]:
    """A pattern for `anchor` and `fact` within `window` characters, in either order."""
    a, f = phrase_regex(anchor), phrase_regex(fact)
    gap = rf"[\s\S]{{0,{window}}}?"
    return re.compile(rf"(?:{a}{gap}{f})|(?:{f}{gap}{a})", re.IGNORECASE)


def lowered(corpus: dict[str, str]) -> dict[str, str]:
    """The corpus in lower case, made once: with it `pages_matching` and `concept_pages` run
    without the case-insensitive flag, which is several times faster over a large corpus."""
    return {doc: text.lower() for doc, text in corpus.items()}


def pages_matching(
    pattern: str | re.Pattern[str],
    corpus: dict[str, str] | Sequence[tuple[str, str]],
    *,
    is_lowered: bool = False,
) -> list[str]:
    """Doc ids of the pages whose text matches `pattern`. A string is case-insensitive; with
    `is_lowered` (a corpus from `lowered`) the pattern is lower-cased and searched as it is
    (so a pattern must not use upper-case regex escapes such as backslash-D)."""
    if isinstance(pattern, str):
        rx = re.compile(pattern.lower()) if is_lowered else re.compile(pattern, re.IGNORECASE)
    else:
        rx = pattern
    items = corpus.items() if isinstance(corpus, dict) else corpus
    return [doc for doc, text in items if rx.search(text)]


def concept_pages(
    anchor: str,
    fact: str,
    corpus: dict[str, str] | Sequence[tuple[str, str]],
    *,
    window: int = WINDOW,
    is_lowered: bool = False,
) -> list[str]:
    """Doc ids of the pages where `anchor` and `fact` occur within `window` characters (the
    same search as `concept_regex`, made fast: for each place the anchor occurs only the text
    around it is searched for the fact; give a corpus from `lowered` and `is_lowered=True` to
    skip the case-insensitive flag)."""
    flags = 0 if is_lowered else re.IGNORECASE
    a = re.compile(phrase_regex(anchor), flags)
    f = re.compile(phrase_regex(fact), flags)
    items = corpus.items() if isinstance(corpus, dict) else corpus
    found = []
    for doc, text in items:
        if not f.search(text):
            continue
        for m in a.finditer(text):
            if f.search(text, max(0, m.start() - window), m.end() + window):
                found.append(doc)
                break
    return found
