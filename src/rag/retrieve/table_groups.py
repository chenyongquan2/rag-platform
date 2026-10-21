"""A hit on a table summary becomes the table itself (spec P6-01; design s7-2).

At ingest a table gets a short summary chunk next to its own chunk(s); both carry the same
`table_id` (`rag.ingest.tables`). A question that does not use the table's words matches the
summary, but a summary is a model-written sentence, not what the document says. After the text
dedupe and before the reranker, `TableSummaryReplace` therefore

1. asks the index for the table chunks of every `table_id` that a summary hit names (one query,
   with the caller's knowledge base and ACL groups like every other read, `EsTableFetch`);
2. puts those chunks in the place of the summary hit, with its score, and notes the summary's id
   in the chunk's `matched_table_summary` metadata;
3. keeps the first place of every chunk: a table found by itself and through its summary, or
   through two summaries, appears once, at the better rank.

The reranker then reads the table text, and the answer prompt never holds a summary. A summary
whose table cannot be found (deleted, other ACL group) is dropped. When the index cannot be read,
the summary hit stays and `RetrievalTrace.table_fetch_error` names the error: a degraded result
is better than none. Chunks without a `chunk_kind` (every other knowledge base) pass untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from elasticsearch import Elasticsearch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import PrivateAttr

from rag.retrieve.filters import build_es_filter
from rag.retrieve.hybrid import SOURCE_FIELDS
from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import RERANK_SCORE

TABLE_SUMMARY = "table_summary"
TABLE = "table"
MATCHED_SUMMARY = "matched_table_summary"
"""Node metadata key: the id of the summary chunk that led to this table chunk."""
DEFAULT_MAX_PARTS = 4
MAX_TABLE_HITS = 200

TableFetch = Callable[..., Mapping[str, Sequence[TextNode]]]
"""`fetch(table_ids, acl_groups, timeout_s=None) -> {table_id: [table chunks in page order]}`."""


class EsTableFetch:
    """Reads the table chunks of some tables from one knowledge base's index."""

    def __init__(self, es: Elasticsearch, *, kb: str, index: str) -> None:
        self._es = es
        self.kb = kb
        self.index = index

    def __call__(
        self, table_ids: Sequence[str], acl_groups: Sequence[str], timeout_s: float | None = None
    ) -> dict[str, list[TextNode]]:
        es_filter = build_es_filter(kb=self.kb, acl_groups=acl_groups)
        query = {
            "bool": {
                "filter": [
                    *es_filter,
                    {"terms": {"table_id": list(table_ids)}},
                    {"term": {"chunk_kind": TABLE}},
                ]
            }
        }
        client: Any = self._es if timeout_s is None else self._es.options(request_timeout=timeout_s)
        response = client.search(
            index=self.index,
            query=query,
            size=MAX_TABLE_HITS,
            sort=[{"chunk_index": "asc"}],
            source={"includes": list(SOURCE_FIELDS)},
            track_total_hits=False,
        )
        found: dict[str, list[TextNode]] = {}
        hits = sorted(
            response["hits"]["hits"],
            key=lambda h: (str(h["_source"].get("doc_id", "")), h["_source"].get("chunk_index", 0)),
        )
        for hit in hits:
            source = dict(hit["_source"])
            text = str(source.pop("content", "") or "")
            metadata = {k: v for k, v in source.items() if v is not None}
            node = TextNode(id_=str(hit["_id"]), text=text, metadata=metadata)
            found.setdefault(str(metadata["table_id"]), []).append(node)
        return found


class TableSummaryReplace(BaseNodePostprocessor):
    """Replace summary hits by their table chunks (see the module doc)."""

    timing_name: ClassVar[str] = "table_groups"
    """Key of this step in `RetrievalResult.timings_ms`."""
    max_parts: int = DEFAULT_MAX_PARTS
    """Most chunks taken for one table (a table cut into more parts is cut off there)."""
    _fetch: TableFetch = PrivateAttr()

    def __init__(self, fetch: TableFetch, *, max_parts: int = DEFAULT_MAX_PARTS) -> None:
        super().__init__(max_parts=max_parts)
        self._fetch = fetch

    @classmethod
    def class_name(cls) -> str:
        return "TableSummaryReplace"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        summaries = [n for n in nodes if n.node.metadata.get("chunk_kind") == TABLE_SUMMARY]
        if not summaries:
            return nodes
        trace = query_bundle.trace if isinstance(query_bundle, HybridQuery) else None
        acl = query_bundle.acl_groups if isinstance(query_bundle, HybridQuery) else ()
        table_ids = list(dict.fromkeys(str(n.node.metadata["table_id"]) for n in summaries))
        timeout_s = None
        if isinstance(query_bundle, HybridQuery) and query_bundle.deadline is not None:
            timeout_s = max(0.05, query_bundle.deadline.remaining_s())
        try:
            tables = self._fetch(table_ids, acl, timeout_s)
        except Exception as exc:  # a degraded result beats none: the summary hits stay
            if trace is not None:
                trace.table_fetch_error = type(exc).__name__
            return nodes

        out: list[NodeWithScore] = []
        seen: set[str] = set()

        def add(item: NodeWithScore) -> None:
            if item.node.node_id not in seen:
                seen.add(item.node.node_id)
                out.append(item)

        for item in nodes:
            meta = item.node.metadata
            if meta.get("chunk_kind") != TABLE_SUMMARY:
                add(item)
                continue
            parts = list(tables.get(str(meta["table_id"]), []))[: self.max_parts]
            if trace is not None:
                trace.table_replacements[item.node.node_id] = [p.node_id for p in parts]
            for part in parts:
                replaced = TextNode(
                    id_=part.node_id,
                    text=part.get_content(),
                    metadata={
                        **part.metadata,
                        MATCHED_SUMMARY: item.node.node_id,
                        **{
                            k: meta[k]
                            for k in ("route_ranks", "route_scores", RERANK_SCORE)
                            if k in meta
                        },
                    },
                )
                add(NodeWithScore(node=replaced, score=item.score))
        # A table chunk that was found by itself and listed before its summary keeps its place and
        # no `matched_table_summary` note; one found only through the summary has the note.
        return out
