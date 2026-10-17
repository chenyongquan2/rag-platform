"""Context around retrieved chunks, and `get_context` for one chunk id (design s10-3, s7-2, s7-6).

Chunks are small so that retrieval is precise (s7-2); a caller often needs the text around a
hit. Every chunk stores `doc_id`, `chunk_index` (its position on the page) and `parent_id` (its
heading section), so the text around it is one Elasticsearch query away:

- `neighbors`: chunks of the same page whose `chunk_index` is within `window` of the hit's;
- `parent`: the other chunks of the hit's heading section (same `parent_id`).

`ContextExpander` (a LlamaIndex postprocessor, after the top_k cut) adds these as
`Chunk.context` pieces. A piece is never a chunk that is itself in the result, and never given
twice. The token budget (`max_tokens`, counted with the reranker's tokenizer) covers the
returned chunks plus all pieces: pieces are added nearest first, one distance step at a time
for every chunk in rank order, and a side of a chunk stops at the first piece that does not
fit, so each chunk's context stays contiguous. Returned chunks are never cut by the budget. If
Elasticsearch fails or no time is left, the chunks are returned without context
(`trace.expand_error`): context is a convenience, not worth failing the search for.

`ContextReader.get_context(chunk_id, mode)` returns a chunk and its context in page order. Chunk
ids are `doc_id#<n>` (P0-04); when the id no longer exists - the page was re-chunked or
deleted after the caller got it - it returns the page's current chunks by that doc_id with
`updated` set (s7-6), or `found=False` when the page is gone too. Every read goes through the
same filter as the search (knowledge base and the caller's ACL groups, s16-4), so a chunk the
caller may not see counts as missing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from elastic_transport import TransportError
from elasticsearch import ApiError, Elasticsearch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from pydantic import Field, PrivateAttr

from rag.retrieve.deadline import Deadline
from rag.retrieve.filters import build_es_filter
from rag.retrieve.hybrid import SOURCE_FIELDS, RetrievalError
from rag.retrieve.query import HybridQuery
from rag.retrieve.search import CONTEXT_KEY, node_to_chunk
from rag.types import Chunk, ContextResult, ErrorCategory

logger = logging.getLogger(__name__)

ExpandMode = Literal["none", "neighbors", "parent"]
ContextMode = Literal["chunk", "neighbors", "parent", "page"]
CONTEXT_MODES: tuple[str, ...] = ("chunk", "neighbors", "parent", "page")
MAX_HITS = 500
"""Most chunks one context read returns (a page has about 10 on average, a few hundreds at most)."""

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class _Row:
    chunk_id: str
    source: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return str(self.source.get("doc_id", ""))

    @property
    def index(self) -> int:
        return int(self.source.get("chunk_index", 0))

    @property
    def text(self) -> str:
        return str(self.source.get("content", "") or "")

    def to_chunk(self, kb: str) -> Chunk:
        meta = {k: v for k, v in self.source.items() if k != "content" and v is not None}
        node = TextNode(id_=self.chunk_id, text=self.text, metadata=meta)
        return node_to_chunk(NodeWithScore(node=node, score=None), kb)


def _contiguous(rows: Sequence[_Row], center: int, budget: int, count: TokenCounter) -> list[_Row]:
    """Rows nearest to `center` first, alternating sides, while they fit `budget`; a side stops
    at its first row that does not fit, so the result has no gaps."""
    sides = [
        sorted((r for r in rows if r.index < center), key=lambda r: -r.index),
        sorted((r for r in rows if r.index > center), key=lambda r: r.index),
    ]
    chosen: list[_Row] = []
    while any(sides):
        for k, side in enumerate(sides):
            if not side:
                continue
            row = side.pop(0)
            cost = count(row.text)
            if cost > budget:
                sides[k] = []
                continue
            chosen.append(row)
            budget -= cost
    return chosen


class ContextReader:
    """Reads chunks around others from one knowledge base's alias (current content)."""

    def __init__(self, es: Elasticsearch, *, kb: str, index: str, count_tokens: TokenCounter):
        self._es = es
        self.kb = kb
        self.index = index
        self.count_tokens = count_tokens

    # ---- Elasticsearch ----------------------------------------------------------------------

    def _search(
        self,
        *,
        acl_groups: Sequence[str],
        filters: Mapping[str, Any] | None = None,
        must: Iterable[Mapping[str, Any]] = (),
        should: Sequence[Mapping[str, Any]] = (),
        size: int = MAX_HITS,
        deadline: Deadline | None = None,
    ) -> list[_Row]:
        """Chunks matching `must` (and one of `should`) under the caller's filter, sorted by
        doc_id and chunk_index. Raises `RetrievalError` (timeout) when no time is left."""
        es_filter = build_es_filter(kb=self.kb, acl_groups=acl_groups, filters=filters)
        query: dict[str, Any] = {"bool": {"filter": [*es_filter, *must]}}
        if should:
            query["bool"]["should"] = list(should)
            query["bool"]["minimum_should_match"] = 1
        client: Any = self._es
        if deadline is not None:
            remaining = deadline.remaining_s()
            if remaining <= 0:
                raise RetrievalError(
                    "deadline passed before the context was read",
                    category=ErrorCategory.RETRIEVAL_TIMEOUT,
                )
            client = self._es.options(request_timeout=remaining)
        resp = client.search(
            index=self.index,
            query=query,
            size=size,
            sort=[{"doc_id": "asc"}, {"chunk_index": "asc"}],
            source={"includes": list(SOURCE_FIELDS)},
            track_total_hits=False,
        )
        rows = [_Row(str(h["_id"]), dict(h["_source"])) for h in resp["hits"]["hits"]]
        rows.sort(key=lambda r: (r.doc_id, r.index))
        return rows

    def around(
        self,
        nodes: Sequence[NodeWithScore],
        *,
        mode: ExpandMode,
        window: int,
        acl_groups: Sequence[str],
        filters: Mapping[str, Any] | None = None,
        deadline: Deadline | None = None,
    ) -> list[_Row]:
        """Neighbours or section chunks of `nodes` in one query (the nodes themselves included)."""
        should: list[dict[str, Any]] = []
        size = 0
        for item in nodes:
            meta = item.node.metadata
            if mode == "neighbors" and meta.get("doc_id") and meta.get("chunk_index") is not None:
                i = int(meta["chunk_index"])
                indices = [j for j in range(i - window, i + window + 1) if j >= 0 and j != i]
                should.append(
                    {
                        "bool": {
                            "filter": [
                                {"term": {"doc_id": meta["doc_id"]}},
                                {"terms": {"chunk_index": indices}},
                            ]
                        }
                    }
                )
                size += len(indices)
            elif mode == "parent" and meta.get("parent_id"):
                clause = {"term": {"parent_id": meta["parent_id"]}}
                if clause not in should:
                    should.append(clause)
                size = MAX_HITS
        if not should:
            return []
        return self._search(
            acl_groups=acl_groups,
            filters=filters,
            should=should,
            size=min(MAX_HITS, size),
            deadline=deadline,
        )

    # ---- get_context ------------------------------------------------------------------------

    def get_context(
        self,
        chunk_id: str,
        mode: ContextMode,
        *,
        acl_groups: Sequence[str],
        window: int = 1,
        max_tokens: int = 4000,
        deadline: Deadline | None = None,
    ) -> ContextResult:
        """The chunk and its context per `mode` (`chunk`, `neighbors`, `parent`, `page`) in page
        order, within `max_tokens` (the chunk itself always); see the module doc for a chunk id
        that no longer exists. ES errors are raised."""
        if mode not in CONTEXT_MODES:
            raise ValueError(f"mode must be one of {CONTEXT_MODES}, got {mode!r}")
        doc_id, sep, _ = chunk_id.rpartition("#")
        if not sep or not doc_id:
            raise ValueError(f"chunk id must look like doc_id#<n>, got {chunk_id!r}")
        if window < 1 or max_tokens < 1:
            raise ValueError("window and max_tokens must be >= 1")
        found = self._search(
            acl_groups=acl_groups,
            must=[{"term": {"chunk_id": chunk_id}}],
            size=1,
            deadline=deadline,
        )
        if not found:
            return self._current_page(chunk_id, doc_id, mode, acl_groups, max_tokens, deadline)
        target = found[0]
        others: list[_Row] = []
        if mode != "chunk":
            must: list[dict[str, Any]] = [{"term": {"doc_id": target.doc_id}}]
            if mode == "neighbors":
                indices = list(range(max(0, target.index - window), target.index + window + 1))
                must.append({"terms": {"chunk_index": indices}})
            elif mode == "parent":
                parent = target.source.get("parent_id") or target.doc_id
                must.append({"term": {"parent_id": parent}})
            rows = self._search(acl_groups=acl_groups, must=must, deadline=deadline)
            others = [r for r in rows if r.chunk_id != target.chunk_id]
        budget = max_tokens - self.count_tokens(target.text)
        chosen = _contiguous(others, target.index, budget, self.count_tokens)
        page = sorted([target, *chosen], key=lambda r: r.index)
        return ContextResult(
            chunk_id=chunk_id,
            doc_id=target.doc_id,
            mode=mode,
            truncated=len(chosen) < len(others),
            chunks=[r.to_chunk(self.kb) for r in page],
        )

    def _current_page(
        self,
        chunk_id: str,
        doc_id: str,
        mode: str,
        acl_groups: Sequence[str],
        max_tokens: int,
        deadline: Deadline | None,
    ) -> ContextResult:
        """The page's current chunks from its start, within the budget (s7-6)."""
        rows = self._search(
            acl_groups=acl_groups, must=[{"term": {"doc_id": doc_id}}], deadline=deadline
        )
        if not rows:
            return ContextResult(chunk_id=chunk_id, doc_id=doc_id, mode=mode, found=False)
        chosen: list[_Row] = []
        budget = max_tokens
        for row in rows:
            cost = self.count_tokens(row.text)
            if cost > budget:
                break
            chosen.append(row)
            budget -= cost
        return ContextResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            mode=mode,
            updated=True,
            truncated=len(chosen) < len(rows),
            chunks=[r.to_chunk(self.kb) for r in chosen],
        )


class ContextExpander(BaseNodePostprocessor):
    """Add neighbour or section chunks to each returned chunk (see the module doc).

    Needs a `HybridQuery`: the reads use the caller's ACL groups and filters and the deadline.
    """

    timing_name: ClassVar[str] = "expand"

    mode: ExpandMode = Field(default="none", description="none, neighbors or parent.")
    window: int = Field(default=1, ge=1, description="Neighbours on each side.")
    max_tokens: int = Field(default=4000, gt=0, description="Chunks plus context, in tokens.")
    _reader: ContextReader = PrivateAttr()

    def __init__(
        self,
        *,
        reader: ContextReader,
        mode: ExpandMode = "none",
        window: int = 1,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> None:
        super().__init__(mode=mode, window=window, max_tokens=max_tokens, **kwargs)
        self._reader = reader

    @property
    def reader(self) -> ContextReader:
        return self._reader

    @classmethod
    def class_name(cls) -> str:
        return "ContextExpander"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if self.mode == "none" or not nodes:
            return nodes
        if not isinstance(query_bundle, HybridQuery):
            raise TypeError(
                "ContextExpander reads Elasticsearch and needs a HybridQuery carrying the "
                f"caller's ACL groups, got {type(query_bundle).__name__}"
            )
        trace = query_bundle.trace
        deadline = query_bundle.deadline
        if deadline is not None and deadline.expired:
            trace.expand_error = "deadline"
            logger.warning("context expansion skipped: no time left")
            return nodes
        try:
            rows = self._reader.around(
                nodes,
                mode=self.mode,
                window=self.window,
                acl_groups=query_bundle.acl_groups,
                filters=query_bundle.filters,
                deadline=deadline,
            )
        except RetrievalError:
            trace.expand_error = "deadline"
            logger.warning("context expansion skipped: no time left")
            return nodes
        except (TransportError, ApiError) as exc:
            trace.expand_error = type(exc).__name__
            logger.warning("context expansion skipped: %s", type(exc).__name__)
            return nodes

        count = self._reader.count_tokens
        used = {item.node.node_id for item in nodes}
        budget = self.max_tokens - sum(
            count(item.node.get_content(metadata_mode=MetadataMode.NONE)) for item in nodes
        )
        pieces: dict[str, list[_Row]] = {item.node.node_id: [] for item in nodes}
        sides: list[tuple[str, list[_Row]]] = []
        for item in nodes:
            meta = item.node.metadata
            if meta.get("chunk_index") is None:
                continue
            center = int(meta["chunk_index"])
            if self.mode == "neighbors":
                mine = [r for r in rows if r.doc_id == meta.get("doc_id")]
                mine = [r for r in mine if abs(r.index - center) <= self.window]
            else:
                mine = [r for r in rows if r.source.get("parent_id") == meta.get("parent_id")]
                mine = [r for r in mine if r.doc_id == meta.get("doc_id")]
            before = sorted((r for r in mine if r.index < center), key=lambda r: -r.index)
            after = sorted((r for r in mine if r.index > center), key=lambda r: r.index)
            sides.extend([(item.node.node_id, before), (item.node.node_id, after)])

        # One distance step at a time over every chunk (rank order), so the best chunk does not
        # take the whole budget; a side stops at its first piece that does not fit.
        open_sides = [(owner, list(side)) for owner, side in sides if side]
        while open_sides and budget > 0:
            still_open: list[tuple[str, list[_Row]]] = []
            for owner, side in open_sides:
                while side and side[0].chunk_id in used:
                    side.pop(0)  # shown already: the side goes on past it
                if not side:
                    continue
                row = side.pop(0)
                cost = count(row.text)
                if cost > budget:
                    continue  # this side is closed
                pieces[owner].append(row)
                used.add(row.chunk_id)
                budget -= cost
                if side:
                    still_open.append((owner, side))
            open_sides = still_open

        added = 0
        for item in nodes:
            mine = sorted(pieces[item.node.node_id], key=lambda r: r.index)
            if mine:
                item.node.metadata[CONTEXT_KEY] = [
                    {"chunk_id": r.chunk_id, "chunk_index": r.index, "text": r.text} for r in mine
                ]
                added += len(mine)
        trace.expanded = added
        return nodes
