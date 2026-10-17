"""EsVectorStore against an in-memory fake Elasticsearch (spec P0-06, design s7-4, s5-4)."""

from __future__ import annotations

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from rag.index.mapping import build_index_body
from rag.index.store import (
    BulkWriteError,
    EsVectorStore,
    FinalizeError,
    StoreFieldError,
    WriteFields,
)
from rag.testing.fakes import FakeEmbedding
from rag.testing.services import FakeElasticsearch

INDEX = "kb_t_v1"
DIMS = 4
A = "t:docs/a.md"
B = "t:docs/b.md"
C = "t:docs/c.md"
EMBED = FakeEmbedding(dim=DIMS)


@pytest.fixture
def es() -> FakeElasticsearch:
    fake = FakeElasticsearch()
    body = build_index_body(analyzer="standard", dims=DIMS)
    fake.indices.create(index=INDEX, settings=body["settings"], mappings=body["mappings"])
    return fake


def chunk(doc_id: str, n: int, text: str | None = None, **extra: object) -> TextNode:
    text = text if text is not None else f"body {n} of {doc_id}"
    node = TextNode(
        id_=f"{doc_id}#{n}",
        text=text,
        metadata={
            "kb": "t",
            "doc_id": doc_id,
            "title_path": "Guide > Install",
            "source_rel": doc_id.split(":", 1)[1],
            "unit": None,
            "source_url": None,
            "doc_type": "markdown",
            "language": "en",
            "acl_groups": ["public"],
            "images": [],
            "pipeline_version": "v2-test",
            "suspicious_instruction": False,
            "parent_id": f"{doc_id}#s0",
            "chunk_index": n,
            **extra,
        },
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)},
    )
    node.embedding = EMBED.get_text_embedding(text)
    return node


def fields(*doc_ids: str, version: str = "src@1") -> dict[str, WriteFields]:
    return {d: WriteFields(source_version=version, content_hash=f"hash-{d}") for d in doc_ids}


def test_add_writes_chunk_fields(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    store.begin_run("run-1", fields(A))

    front_matter = {"title": "Install", "tags": ["setup", "pip"]}
    ids = store.add([chunk(A, 0, "Install with pip.", front_matter=front_matter), chunk(A, 1)])

    assert ids == [f"{A}#0", f"{A}#1"]
    source = es.docs[INDEX][f"{A}#0"]
    assert source["chunk_id"] == f"{A}#0"
    assert source["content"] == "Install with pip."  # body only: no title path, no metadata
    assert source["title_path"] == "Guide > Install"
    assert source["doc_id"] == A
    assert source["chunk_index"] == 0
    assert source["parent_id"] == f"{A}#s0"
    assert source["acl_groups"] == ["public"]
    assert source["images"] == []
    assert source["front_matter"] == front_matter  # stored, not indexed
    assert source["ingest_run_id"] == "run-1"
    assert source["content_hash"] == f"hash-{A}"
    assert source["source_version"] == "src@1"
    assert len(source["embedding"]) == DIMS
    assert "unit" not in source  # None values are left out
    assert store.written_docs == {A: 2}
    assert store.stores_text is True


def test_add_rejects_bad_chunks_before_sending(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    with pytest.raises(StoreFieldError, match="begin_run"):
        store.add([chunk(A, 0)])

    store.begin_run("run-1", fields(A))
    with pytest.raises(StoreFieldError, match=B):  # no write fields for this document
        store.add([chunk(A, 0), chunk(B, 0)])
    with pytest.raises(StoreFieldError, match="surprise"):  # not in the strict mapping
        store.add([chunk(A, 0, surprise="x")])
    with pytest.raises(StoreFieldError, match="ingest_run_id"):  # write-time keys come from the run
        store.add([chunk(A, 0, ingest_run_id="other")])
    no_vector = chunk(A, 1)
    no_vector.embedding = None
    with pytest.raises(StoreFieldError, match="embedding"):
        store.add([chunk(A, 0), no_vector])

    assert es.bulk_calls == []
    assert es.docs[INDEX] == {}


def test_add_raises_on_partial_bulk_failure(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    store.begin_run("run-1", fields(A))
    es.fail_ids = {f"{A}#1"}

    with pytest.raises(BulkWriteError, match="injected failure") as info:
        store.add([chunk(A, 0), chunk(A, 1), chunk(A, 2)])

    assert [item.id for item in info.value.failed] == [f"{A}#1"]
    assert info.value.failed[0].status == 400
    # Elasticsearch answered HTTP 200 and kept the other items; the run must still fail.
    assert set(es.docs[INDEX]) == {f"{A}#0", f"{A}#2"}


def test_add_sends_batches(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX, bulk_batch_size=2)
    store.begin_run("run-1", fields(A))

    store.add([chunk(A, n) for n in range(5)])

    assert es.bulk_calls == [2, 2, 1]
    assert len(es.docs[INDEX]) == 5


def test_delete_only_records_pending(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    store.begin_run("run-1", fields(A))
    store.add([chunk(A, 0), chunk(A, 1)])

    store.delete(A)
    store.delete(A)

    assert store.pending_deletes == [A]
    assert len(es.docs[INDEX]) == 2  # still searchable until finalize()
    assert es.delete_queries == []


def test_finalize_deletes_leftovers_and_removed_docs(es: FakeElasticsearch) -> None:
    first = EsVectorStore(es, INDEX)
    first.begin_run("run-1", fields(A, B, C))
    first.add([chunk(A, 0), chunk(A, 1), chunk(A, 2), chunk(B, 0), chunk(B, 1), chunk(C, 0)])
    assert first.finalize().deleted_chunks == 0

    store = EsVectorStore(es, INDEX)
    store.begin_run("run-2", fields(A, B, C, version="src@2"))
    store.delete(A)  # A changed: now two chunks
    store.add([chunk(A, 0, "new 0"), chunk(A, 1, "new 1")])
    store.delete(B)  # B is gone from the source
    store.refresh()

    result = store.finalize()

    assert result.updated_docs == [A]
    assert result.removed_docs == [B]
    assert result.deleted_chunks == 3  # A#2 (leftover) and B#0, B#1
    assert set(es.docs[INDEX]) == {f"{A}#0", f"{A}#1", f"{C}#0"}
    assert es.docs[INDEX][f"{A}#0"]["ingest_run_id"] == "run-2"
    assert es.docs[INDEX][f"{C}#0"]["ingest_run_id"] == "run-1"  # untouched
    assert es.delete_queries[-1]["refresh"] is True
    assert store.pending_deletes == []


def test_finalize_raises_on_failures(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    store.begin_run("run-1", fields(A))
    store.delete(A)
    es.delete_failures = [{"id": f"{A}#0", "cause": {"type": "version_conflict_engine_exception"}}]

    with pytest.raises(FinalizeError, match="version_conflict"):
        store.finalize()


def test_query_returns_nodes_by_vector(es: FakeElasticsearch) -> None:
    store = EsVectorStore(es, INDEX)
    store.begin_run("run-1", fields(A, B))
    store.add([chunk(A, 0, "alpha"), chunk(B, 0, "beta")])

    result = store.query(
        VectorStoreQuery(query_embedding=EMBED.get_query_embedding("beta"), similarity_top_k=1)
    )
    assert result.ids == [f"{B}#0"]
    assert result.nodes[0].get_content() == "beta"
    assert result.nodes[0].metadata["doc_id"] == B
    assert result.similarities and result.similarities[0] > 0.99

    only_a = store.query(
        VectorStoreQuery(
            query_embedding=EMBED.get_query_embedding("beta"), similarity_top_k=2, doc_ids=[A]
        )
    )
    assert only_a.ids == [f"{A}#0"]
