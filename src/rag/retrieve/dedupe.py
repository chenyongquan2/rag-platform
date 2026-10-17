"""Drop chunks whose body text is the same (design s9-3).

The same paragraph often appears on several pages (shared install notes, licence blocks, API
snippets): the chunk ids differ, the bodies do not. Keeping both wastes reranker time and
final-list slots on one piece of information. The node's body (`MetadataMode.NONE`: no
metadata in the text) is normalized - Unicode NFKC, case folded, whitespace runs collapsed -
after dropping whole lines equal to the chunk's own page path (`source_rel`, `source_url`),
and the first node (best fused rank) of each key is kept. There is no backfill query: if
deduping leaves fewer candidates than wanted, the chain goes on with what is left (s9-3).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import ClassVar

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import Field

from rag.retrieve.query import HybridQuery

_WS = re.compile(r"\s+")


def dedupe_key(text: str, drop_lines: Iterable[str] = ()) -> str:
    """Normalized body: lines equal (after stripping) to one of `drop_lines` removed, then NFKC,
    case folded, every whitespace run one space."""
    drop = {line.strip() for line in drop_lines if line and line.strip()}
    lines = [line for line in text.splitlines() if line.strip() not in drop]
    normalized = unicodedata.normalize("NFKC", "\n".join(lines)).casefold()
    return _WS.sub(" ", normalized).strip()


class TextDedupe(BaseNodePostprocessor):
    """Keep the first node of each normalized body; record removed ids in the HybridQuery trace."""

    timing_name: ClassVar[str] = "dedupe"
    """Key of this step in `RetrievalResult.timings_ms`."""

    path_keys: tuple[str, ...] = Field(
        default=("source_rel", "source_url"),
        description="Metadata keys whose values, as whole lines of the text, are page-path lines.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TextDedupe"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        kept: list[NodeWithScore] = []
        first: dict[str, str] = {}
        duplicates: dict[str, str] = {}
        for item in nodes:
            node = item.node
            paths = [str(node.metadata[k]) for k in self.path_keys if node.metadata.get(k)]
            key = dedupe_key(node.get_content(metadata_mode=MetadataMode.NONE), paths)
            if key in first:
                duplicates[node.node_id] = first[key]
                continue
            first[key] = node.node_id
            kept.append(item)
        if isinstance(query_bundle, HybridQuery):
            query_bundle.trace.duplicates.update(duplicates)
            query_bundle.trace.candidates_after_dedupe = len(kept)
        return kept
