"""`EsVectorStore`: LlamaIndex vector store over one knowledge-base index; writes, then deletes.

Design s7-4. LlamaIndex's `IngestionPipeline` (UPSERTS_AND_DELETE) calls `delete(ref_doc_id)` for
a changed page before it chunks and embeds the page again, and for a page that disappeared. If
that deleted at once, a refresh between the delete and the `add()` would show a query no chunk
of the page, and a crash in between would lose the page until the next run. So:

- `delete(ref_doc_id)` only records the page in a pending list.
- `add(nodes)` bulk-`index`es the new chunks under fixed ids `doc_id#n` (an old chunk with the
  same number is overwritten); each carries this run's `ingest_run_id`.
- After the run and a refresh, `finalize()` deletes, for every pending page, the chunks whose
  `ingest_run_id` is not this run: the leftovers of a page that now has fewer chunks, or every
  chunk of a page that produced none (deleted or renamed).

Every chunk also gets the write-time fields of s5-4 (`ingest_run_id`, `content_hash`,
`source_version`) from the run (`begin_run`), never from Document metadata, because metadata is
part of the page hash. `content` holds the body text only; `title_path` is its own field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elasticsearch import Elasticsearch
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import Field, PrivateAttr

from rag.index.mapping import (
    BOOLEAN_FIELDS,
    INTEGER_FIELDS,
    KEYWORD_FIELDS,
    STORED_OBJECT_FIELDS,
    TEXT_FIELDS,
    VECTOR_FIELD,
)

RUN_FIELDS = ("ingest_run_id", "content_hash", "source_version")
"""Written per chunk from the run, never taken from metadata (s5-4)."""
_SET_BY_STORE = {"content", "chunk_id", VECTOR_FIELD, *RUN_FIELDS}
METADATA_FIELDS = frozenset(
    {*TEXT_FIELDS, *KEYWORD_FIELDS, *INTEGER_FIELDS, *BOOLEAN_FIELDS, *STORED_OBJECT_FIELDS}
    - _SET_BY_STORE
)
"""Chunk metadata keys that have a field in the index mapping (s7-3)."""


class StoreFieldError(ValueError):
    """A chunk cannot be written as it is (no run fields, no vector, key outside the mapping)."""


@dataclass(frozen=True)
class BulkItemFailure:
    id: str
    status: int
    error_type: str
    reason: str


class BulkWriteError(RuntimeError):
    """Elasticsearch rejected some chunks of a bulk request (the request itself got HTTP 200)."""

    def __init__(self, failed: Sequence[BulkItemFailure], written: int) -> None:
        self.failed = list(failed)
        self.written = written
        shown = "; ".join(f"{f.id}: {f.status} {f.error_type}: {f.reason}" for f in failed[:5])
        more = f" (and {len(failed) - 5} more)" if len(failed) > 5 else ""
        super().__init__(
            f"bulk write failed for {len(failed)} chunks, {written} written: {shown}{more}"
        )


class FinalizeError(RuntimeError):
    """`delete_by_query` reported failures; old chunks may remain until the next run."""


@dataclass(frozen=True)
class WriteFields:
    """Write-time fields of one page (s5-4): upstream version and hash of the cleaned text."""

    source_version: str
    content_hash: str


@dataclass(frozen=True)
class FinalizeResult:
    updated_docs: list[str]
    """Pending pages with new chunks in this run: only their leftovers were deleted."""
    removed_docs: list[str]
    """Pending pages without new chunks: all their chunks were deleted."""
    deleted_chunks: int


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EsVectorStore(BasePydanticVectorStore):
    """Writes chunks of one knowledge base into one concrete index (`kb_<kb>_v<N>`, not the
    alias: a rebuild writes a new version while queries still use the old one, s7-5)."""

    stores_text: bool = True
    flat_metadata: bool = False
    index_name: str
    bulk_batch_size: int = Field(default=500, gt=0)
    delete_batch_size: int = Field(default=1000, gt=0)

    _es: Elasticsearch = PrivateAttr()
    _run_id: str | None = PrivateAttr(default=None)
    _fields: dict[str, WriteFields] = PrivateAttr(default_factory=dict)
    _pending: dict[str, None] = PrivateAttr(default_factory=dict)
    _written: dict[str, int] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        es: Elasticsearch,
        index_name: str,
        *,
        bulk_batch_size: int = 500,
        delete_batch_size: int = 1000,
    ) -> None:
        super().__init__(
            index_name=index_name,
            bulk_batch_size=bulk_batch_size,
            delete_batch_size=delete_batch_size,
        )
        self._es = es

    @classmethod
    def class_name(cls) -> str:
        return "EsVectorStore"

    @property
    def client(self) -> Elasticsearch:
        return self._es

    # ---- run state --------------------------------------------------------------------------

    def begin_run(self, run_id: str, fields: Mapping[str, WriteFields]) -> None:
        """Start a run: its id and the write-time fields of every page it may write."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        self._run_id = run_id
        self._fields = dict(fields)
        self._pending = {}
        self._written = {}

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def pending_deletes(self) -> list[str]:
        """Pages LlamaIndex asked to delete in this run, in the order it asked."""
        return list(self._pending)

    @property
    def written_docs(self) -> dict[str, int]:
        """Chunks written in this run per page."""
        return dict(self._written)

    # ---- writes -----------------------------------------------------------------------------

    def _check(self, node: BaseNode) -> str:
        """The chunk's doc id; raises `StoreFieldError` if the chunk cannot be written."""
        doc_id = node.ref_doc_id or node.metadata.get("doc_id")
        if not doc_id or doc_id not in self._fields:
            raise StoreFieldError(f"{node.node_id}: no write fields for document {doc_id!r}")
        if node.embedding is None:
            raise StoreFieldError(f"{node.node_id}: no embedding; the pipeline must embed it")
        unknown = sorted(k for k in node.metadata if k not in METADATA_FIELDS)
        if unknown:
            raise StoreFieldError(
                f"{node.node_id}: metadata keys {unknown} have no field in the index mapping "
                f"(or are write-time fields set by the store: {list(RUN_FIELDS)}); add them to "
                "rag.index.mapping first"
            )
        return str(doc_id)

    def _source(self, node: BaseNode, run_id: str) -> dict[str, Any]:
        """The `_source` of a checked chunk (copies its vector)."""
        doc_id = self._check(node)
        source: dict[str, Any] = {k: v for k, v in node.metadata.items() if v is not None}
        written = self._fields[doc_id]
        source.update(
            chunk_id=node.node_id,
            doc_id=doc_id,
            content=node.get_content(metadata_mode=MetadataMode.NONE),
            embedding=list(node.embedding or []),
            ingest_run_id=run_id,
            content_hash=written.content_hash,
            source_version=written.source_version,
        )
        return source

    def add(self, nodes: Sequence[BaseNode], **_: Any) -> list[str]:
        """Bulk-index the chunks; raise `BulkWriteError` if any item failed.

        Every chunk is checked before the first request, so a bad chunk sends nothing. Request
        bodies are built one batch at a time, so only one batch of vectors is copied at once.
        """
        run_id = self._run_id
        if run_id is None:
            raise StoreFieldError("call begin_run(run_id, fields) before add()")
        doc_ids = [self._check(node) for node in nodes]
        ids: list[str] = []
        for start in range(0, len(nodes), self.bulk_batch_size):
            batch = nodes[start : start + self.bulk_batch_size]
            batch_docs = doc_ids[start : start + self.bulk_batch_size]
            operations: list[dict[str, Any]] = []
            for node in batch:
                operations.append({"index": {"_index": self.index_name, "_id": node.node_id}})
                operations.append(self._source(node, run_id))
            resp = self._es.bulk(operations=operations, refresh=False)
            del operations
            failed: list[BulkItemFailure] = []
            for item, node, doc_id in zip(resp["items"], batch, batch_docs, strict=True):
                result = next(iter(item.values()))
                status = int(result.get("status", 0))
                if status >= 300 or "error" in result:
                    error = result.get("error") or {}
                    failed.append(
                        BulkItemFailure(
                            id=str(result.get("_id", node.node_id)),
                            status=status,
                            error_type=str(error.get("type", "unknown")),
                            reason=str(error.get("reason", ""))[:300],
                        )
                    )
                else:
                    self._written[doc_id] = self._written.get(doc_id, 0) + 1
                    ids.append(node.node_id)
            if failed:
                raise BulkWriteError(failed, written=len(ids))
        return ids

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Record the page; its old chunks are deleted by `finalize()` (s7-4)."""
        self._pending[ref_doc_id] = None

    def refresh(self) -> None:
        self._es.indices.refresh(index=self.index_name)

    def _delete_by_query(self, query: dict[str, Any]) -> int:
        """Delete matching chunks, refresh, return how many; raise `FinalizeError` when
        Elasticsearch reports failures or a time-out (then only part may be deleted)."""
        resp = self._es.delete_by_query(
            index=self.index_name,
            query=query,
            refresh=True,
            conflicts="abort",
            wait_for_completion=True,
        )
        failures = resp.get("failures") or []
        if failures:
            raise FinalizeError(
                f"delete_by_query on {self.index_name} reported {len(failures)} failures, "
                f"first: {failures[0]}"
            )
        if resp.get("timed_out"):
            raise FinalizeError(
                f"delete_by_query on {self.index_name} timed out after deleting "
                f"{resp.get('deleted', 0)} chunks"
            )
        return int(resp.get("deleted", 0))

    def finalize(self, also_clean: Iterable[str] = ()) -> FinalizeResult:
        """Delete chunks of pending pages that this run did not write, then refresh.

        `also_clean` adds pages LlamaIndex did not mark (new to the docstore) whose chunks from
        another run must go too: a run that failed after writing them never saved its docstore.
        Call after the pipeline run and `refresh()`: with the new chunks visible, the query
        cannot meet a chunk that this run is overwriting.
        """
        if self._run_id is None:
            raise StoreFieldError("call begin_run(run_id, fields) before finalize()")
        pending = list(self._pending)
        targets = list(dict.fromkeys([*pending, *also_clean]))
        deleted = 0
        for batch in _batches(targets, self.delete_batch_size):
            deleted += self._delete_by_query(
                {
                    "bool": {
                        "filter": [{"terms": {"doc_id": list(batch)}}],
                        "must_not": [{"term": {"ingest_run_id": self._run_id}}],
                    }
                }
            )
        result = FinalizeResult(
            updated_docs=[d for d in pending if d in self._written],
            removed_docs=[d for d in pending if d not in self._written],
            deleted_chunks=deleted,
        )
        self._pending = {}
        return result

    def sweep(self, doc_ids: Iterable[str]) -> int:
        """Delete every chunk of these pages (pages in the index the docstore does not know)."""
        targets = list(doc_ids)
        deleted = 0
        for batch in _batches(targets, self.delete_batch_size):
            deleted += self._delete_by_query({"terms": {"doc_id": list(batch)}})
        return deleted

    def indexed_doc_ids(self, *, page_size: int = 1000) -> set[str]:
        """Every doc id that has a chunk in the index (composite aggregation, paged)."""
        found: set[str] = set()
        after: dict[str, Any] | None = None
        while True:
            composite: dict[str, Any] = {
                "size": page_size,
                "sources": [{"doc_id": {"terms": {"field": "doc_id"}}}],
            }
            if after is not None:
                composite["after"] = after
            resp = self._es.search(
                index=self.index_name, size=0, aggs={"doc_ids": {"composite": composite}}
            )
            agg = resp["aggregations"]["doc_ids"]
            found.update(str(b["key"]["doc_id"]) for b in agg["buckets"])
            after = agg.get("after_key")
            if not agg["buckets"] or after is None:
                return found

    def chunk_count(self) -> int:
        return int(self._es.count(index=self.index_name)["count"])

    # ---- reads ------------------------------------------------------------------------------

    def query(self, query: VectorStoreQuery, **_: Any) -> VectorStoreQueryResult:
        """Dense kNN only, optionally limited to `doc_ids`. Hybrid search is EsHybridRetriever
        (P0-07); metadata filters are not supported here."""
        if query.query_embedding is None:
            raise ValueError("EsVectorStore.query needs query_embedding (dense search only)")
        if query.filters is not None:
            raise NotImplementedError("metadata filters: use the hybrid retriever (P0-07)")
        k = query.similarity_top_k
        knn: dict[str, Any] = {
            "field": VECTOR_FIELD,
            "query_vector": list(query.query_embedding),
            "k": k,
            "num_candidates": max(50, 2 * k),
        }
        if query.doc_ids:
            knn["filter"] = {"terms": {"doc_id": list(query.doc_ids)}}
        resp = self._es.search(index=self.index_name, knn=knn, size=k)
        nodes: list[TextNode] = []
        scores: list[float] = []
        ids: list[str] = []
        for hit in resp["hits"]["hits"]:
            source = dict(hit["_source"])
            text = source.pop("content", "")
            source.pop(VECTOR_FIELD, None)
            nodes.append(TextNode(id_=hit["_id"], text=text, metadata=source))
            scores.append(float(hit["_score"]))
            ids.append(hit["_id"])
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)
