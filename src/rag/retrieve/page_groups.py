"""One page = one group: keep one language version of a page in the result (spec P4-01, s8-4).

The MT5 manual exists twice, in Chinese and in English, with the same page names. Both versions
of a page match the same question, and the second one would only take a place in the result list
and a slot in the reranker. Every chunk of such a page carries a `page_key` (the page path
without language and extension) and a `language`. `PageKeyDedupe` runs after the text dedupe and
before the reranker, and for every page key that appears in two or more languages keeps the
chunks of one language:

- the language of the question, when that version is among them (a question with Chinese words is
  a Chinese question, whatever identifiers it holds; otherwise `detect_language`);
- else the language of the page's best-ranked chunk.

A page that appears in one language only is left alone, whatever the question's language: a
Chinese question that found only the English page (cross-language retrieval) keeps it. All chunks
of the kept version stay (they are different parts of the page). Nodes without a `page_key`
(every other knowledge base) are not touched, order and scores never change. Each removed chunk
is recorded in the trace, and the kept chunks of such a page say which other languages of the
page also matched this search (`also_matched_languages`). It comes from the candidates only: a
page whose other version was not retrieved has no such entry, and it says nothing about which
languages the page exists in (storing that per chunk would make a page's hash depend on the other
volumes, design s5-4).
"""

from __future__ import annotations

from typing import ClassVar

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.ingest.language import UNDETERMINED, detect_language
from rag.retrieve.query import HybridQuery

ALSO_MATCHED_LANGUAGES = "also_matched_languages"
"""Node metadata key of the languages a kept page also exists in."""


def _chinese_characters(text: str) -> int:
    return sum("㐀" <= c <= "鿿" for c in text)


def question_language(question: str) -> str:
    """The language of a question: `zh` when it has two or more Chinese characters (English
    identifiers inside a Chinese question are normal), else `detect_language`'s answer, which is
    `und` for a question with too little text to tell."""
    if _chinese_characters(question) >= 2 and not any("぀" <= c <= "ヿ" for c in question):
        return "zh"
    return detect_language(question)


class PageKeyDedupe(BaseNodePostprocessor):
    """Keep one language version per `page_key` (see the module doc)."""

    timing_name: ClassVar[str] = "page_groups"
    """Key of this step in `RetrievalResult.timings_ms`."""

    @classmethod
    def class_name(cls) -> str:
        return "PageKeyDedupe"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        asked = (
            question_language(query_bundle.query_str) if query_bundle is not None else UNDETERMINED
        )
        languages: dict[str, list[str]] = {}  # page key -> languages in rank order
        for item in nodes:
            key = item.node.metadata.get("page_key")
            if key:
                seen = languages.setdefault(str(key), [])
                language = str(item.node.metadata.get("language", ""))
                if language not in seen:
                    seen.append(language)
        keep: dict[str, str] = {}
        for key, found in languages.items():
            if len(found) > 1:
                keep[key] = asked if asked in found else found[0]

        kept: list[NodeWithScore] = []
        dropped: dict[str, str] = {}
        for item in nodes:
            key = str(item.node.metadata.get("page_key") or "")
            wanted = keep.get(key)
            if wanted is not None and item.node.metadata.get("language") != wanted:
                dropped[item.node.node_id] = key
                continue
            if wanted is not None:
                others = sorted(lang for lang in languages[key] if lang != wanted)
                item.node.metadata[ALSO_MATCHED_LANGUAGES] = others
            kept.append(item)
        if isinstance(query_bundle, HybridQuery):
            query_bundle.trace.page_key_duplicates.update(dropped)
        return kept
