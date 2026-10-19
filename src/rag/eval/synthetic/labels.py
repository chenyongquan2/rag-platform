"""Page labels of a synthetic question (spec P1-05, design s14-2).

The source page of the span is always a gold page. Many HF pages share text verbatim (timm
model cards repeat the same inference snippet, course pages repeat their subtitles), so another
page can hold the very sentence the question was written from; with only the source page as gold,
retrieval that returns that page is scored wrong.

A page is added as gold only when it contains, verbatim (lower case, whitespace collapsed), an
answer-bearing unit of the span: a line or sentence of the span that contains the answer and is
at least `min_unit_chars` long. A short or common answer alone never adds a page ("accuracy",
"two" and "huggingface_hub" occur on hundreds of pages that do not answer the question). The
answer must occur in the span; a paraphrased answer adds nothing. Pages outside the sampled
languages and the public questions' pages are never added, as in the sampler.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rag.eval.synthetic.sampler import Page

_UNITS = re.compile(r"\n|(?<=[.!?])\s+")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_answer(answer: str) -> str:
    return normalize(answer).rstrip(".").strip("`'\" ")


def answer_units(answer: str, span: str, *, min_unit_chars: int) -> list[str]:
    """Normalized lines and sentences of `span` that contain the answer and are long enough."""
    wanted = normalize_answer(answer)
    if not wanted:
        return []
    units = {normalize(u) for u in _UNITS.split(span)}
    return sorted(u for u in units if wanted in u and len(u) >= min_unit_chars)


class PageIndex:
    """The cleaned pages that may be labelled, normalized once for exact-text searches."""

    def __init__(
        self,
        pages: Iterable[Page],
        *,
        min_unit_chars: int = 40,
        languages: Sequence[str] = ("en",),
        exclude_pages: Iterable[str] = (),
    ) -> None:
        self.min_unit_chars = min_unit_chars
        excluded = set(exclude_pages)
        self._text = {
            p.doc_id: normalize(p.text)
            for p in pages
            if p.language in languages and p.doc_id not in excluded
        }

    def __len__(self) -> int:
        return len(self._text)

    def answering_pages(self, *, answer: str, span: str, source: str) -> list[str]:
        """Other pages holding an answer-bearing line or sentence of the span (sorted)."""
        units = answer_units(answer, span, min_unit_chars=self.min_unit_chars)
        if not units:
            return []
        return sorted(
            doc
            for doc, text in self._text.items()
            if doc != source and any(u in text for u in units)
        )
