"""Spec P0-09 AC1-AC2: the public response schema (design s16-1, s16-4, s7-6)."""

from __future__ import annotations

import json

from rag.retrieve.filters import CALLER_FILTER_KEYS
from rag.serve.schema import (
    ContextResponse,
    ErrorResponse,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    SearchResponse,
)
from rag.types import (
    Chunk,
    ContextPiece,
    ContextResult,
    RetrievalResult,
    RouteHit,
    ThresholdDecision,
)


def _chunk(n: int, *, suspicious: bool = False) -> Chunk:
    return Chunk(
        chunk_id=f"hf_docs:p{n}.md#0",
        doc_id=f"hf_docs:p{n}.md",
        kb="hf_docs",
        text=f"body {n}",
        suspicious_instruction=suspicious,
        title_path=["Page", str(n)],
        source_rel=f"p{n}.md",
        source_url=f"https://example.org/p{n}.md",
        source_version="ds@abc",
        score=0.03,
        rerank_score=0.9,
        context=[ContextPiece(chunk_id=f"hf_docs:p{n}.md#1", chunk_index=1, text="next")],
        metadata={"doc_type": "md", "language": "en", "chunk_index": 0},
    )


def _result() -> RetrievalResult:
    return RetrievalResult(
        request_id="req-1",
        kb="hf_docs",
        index_version="kb_hf_docs_v1",
        data_generation=3,
        degraded=True,
        rerank_timeout=False,
        timings_ms={"search": 12.0, "rerank": 800.0, "total": 900.0},
        chunks=[_chunk(1), _chunk(2, suspicious=True)],
        routes={"bm25": [RouteHit(chunk_id="hf_docs:p1.md#0", doc_id="hf_docs:p1.md", rank=1)]},
        rerank=[RouteHit(chunk_id="hf_docs:p1.md#0", doc_id="hf_docs:p1.md", rank=1, score=0.9)],
        threshold=ThresholdDecision(threshold=0.3, top_score=0.9, passed=True, kept=2),
        config_version="cfg1",
    )


def test_search_response_fields_and_routes_excluded() -> None:
    response = SearchResponse.from_result(_result(), timings_ms={"request": 950.0})
    data = json.loads(response.model_dump_json())

    assert data["request_id"] == "req-1"
    assert data["kb"] == "hf_docs"
    assert data["index_version"] == "kb_hf_docs_v1"
    assert data["data_generation"] == 3
    assert data["config_version"] == "cfg1"
    assert data["degraded"] is True and data["rerank_timeout"] is False
    assert data["not_found"] is False
    assert data["error_categories"] == ["degraded"]
    assert data["timings_ms"] == {"search": 12.0, "rerank": 800.0, "total": 900.0, "request": 950.0}
    assert data["below_threshold"] == []
    first, second = data["chunks"]
    assert first["text"] == "body 1" and first["untrusted"] is True
    assert first["suspicious_instruction"] is False and second["suspicious_instruction"] is True
    assert first["source_url"] == "https://example.org/p1.md"
    assert first["source_version"] == "ds@abc"
    assert first["chunk_id"] == "hf_docs:p1.md#0" and first["doc_id"] == "hf_docs:p1.md"
    assert first["score"] == 0.03 and first["rerank_score"] == 0.9
    assert first["context"] == [{"chunk_id": "hf_docs:p1.md#1", "chunk_index": 1, "text": "next"}]
    # Per-route lists, the reranker's list and the threshold decision stay inside (P0-10, P0-08).
    for internal in ("routes", "rerank", "threshold"):
        assert internal not in data
    assert "routes" not in SearchResponse.model_json_schema()["properties"]

    # A chunk can never be returned as trusted.
    trusted = _chunk(3).model_copy(update={"untrusted": False})
    result = _result().model_copy(update={"chunks": [trusted], "not_found": False})
    assert SearchResponse.from_result(result).chunks[0].untrusted is True

    not_found = _result().model_copy(
        update={"chunks": [], "not_found": True, "below_threshold": [_chunk(4)]}
    )
    data = json.loads(SearchResponse.from_result(not_found).model_dump_json())
    assert data["not_found"] is True and data["chunks"] == []
    assert data["below_threshold"][0]["untrusted"] is True


def test_context_kb_list_and_error_responses() -> None:
    result = ContextResult(
        chunk_id="hf_docs:p1.md#5",
        doc_id="hf_docs:p1.md",
        mode="neighbors",
        updated=True,
        chunks=[_chunk(1)],
    )
    response = ContextResponse.from_result(
        result, request_id="req-2", kb="hf_docs", timings_ms={"request": 5.0}
    )
    data = json.loads(response.model_dump_json())
    assert data == {
        "request_id": "req-2",
        "kb": "hf_docs",
        "chunk_id": "hf_docs:p1.md#5",
        "doc_id": "hf_docs:p1.md",
        "mode": "neighbors",
        "found": True,
        "updated": True,
        "truncated": False,
        "chunks": [json.loads(_chunk(1).model_dump_json())],
        "timings_ms": {"request": 5.0},
    }

    listing = KnowledgeBaseList(
        request_id="req-3",
        knowledge_bases=[
            KnowledgeBaseInfo(
                kb="hf_docs",
                description="Hugging Face docs",
                language="en",
                available=True,
                default_top_k=8,
                max_top_k=30,
            )
        ],
    )
    data = json.loads(listing.model_dump_json())
    assert data["knowledge_bases"][0]["filters"] == list(CALLER_FILTER_KEYS)
    assert data["knowledge_bases"][0]["available"] is True

    error = ErrorResponse.of("invalid_filter", "req-4")
    assert json.loads(error.model_dump_json()) == {
        "error": "invalid_filter",
        "request_id": "req-4",
        "message": error.message,
    }
    assert "source_rel_prefix" in error.message
