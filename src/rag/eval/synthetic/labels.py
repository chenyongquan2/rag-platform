"""Page labels of a synthetic question (spec P1-05, design s14-2).

The source page of the span is always a gold page. Many HF pages share text (timm model cards
repeat the same fine-tuning section, course pages repeat their subtitles), so other pages often
answer the question too; with only the source page as gold, retrieval that returns one of them
is scored wrong. A page is added as gold when, by exact text search over the cleaned corpus:

- it contains the answer (lower case, whitespace collapsed, trailing period and quotes removed),
  and the answer also occurs in the source span (else the answer is a paraphrase and cannot be
  searched for), and is at least `min_answer_chars` long;
- it contains every content word of the question that also occurs in the source span.

Searching for the span itself finds almost nothing (spans are cut at random offsets: 2 of 200
match a whole other page), and searching for a piece of the span labels pages that share the
template but not the answer (a timm model card's model name); see docs/decisions.md.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag.eval.synthetic.filters import STOPWORDS, tokens
from rag.eval.synthetic.sampler import Page


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_answer(answer: str) -> str:
    return normalize(answer).rstrip(".").strip("`'\" ")


class PageIndex:
    """The cleaned pages, normalized once for repeated exact-text searches."""

    def __init__(self, pages: Iterable[Page], *, min_answer_chars: int = 3) -> None:
        self.min_answer_chars = min_answer_chars
        self._text: dict[str, str] = {}
        self._words: dict[str, set[str]] = {}
        for page in pages:
            self._text[page.doc_id] = normalize(page.text)
            self._words[page.doc_id] = set(tokens(page.text))

    def __len__(self) -> int:
        return len(self._text)

    def answering_pages(self, *, question: str, answer: str, span: str, source: str) -> list[str]:
        """Other pages that contain the answer and the question's words (sorted doc_ids)."""
        wanted = normalize_answer(answer)
        span_text = normalize(span)
        if len(wanted) < self.min_answer_chars or wanted not in span_text:
            return []
        span_words = set(tokens(span))
        words = [w for w in tokens(question) if w not in STOPWORDS and w in span_words]
        return sorted(
            doc
            for doc, text in self._text.items()
            if doc != source and wanted in text and all(w in self._words[doc] for w in words)
        )
