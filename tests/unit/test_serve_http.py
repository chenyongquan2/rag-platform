"""Spec P0-09 AC11: the HTTP API (Starlette) over the search service (design s16-1, s16-4)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.serve.http_app import build_http_app
from rag.serve.schema import ContextResponse, KnowledgeBaseList, SearchResponse
from rag.serve.service import SearchService
from rag.testing import memory_observability, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

ERROR_KEYS = {"error", "request_id", "message"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _service(builder: FakePipelineBuilder | None = None) -> SearchService:
    configs = {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs", description="HF docs", access=AccessConfig(callers=["*"])
        )
    }
    builder = builder or FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")})
    return SearchService(observe_settings(), configs=configs, builder=builder)


@pytest.fixture
def client() -> Iterator[TestClient]:
    service = _service()
    with TestClient(build_http_app(service)) as client:  # the lifespan starts the load
        assert service.wait_ready(5)
        yield client


def test_http_operations(client: TestClient) -> None:
    r = client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "how?", "top_k": 2})
    assert r.status_code == 200, r.text
    search = SearchResponse.model_validate(r.json())
    assert search.kb == "hf_docs" and len(search.chunks) == 2
    assert r.headers["x-request-id"] == search.request_id
    assert "routes" not in r.json() and r.json()["chunks"][0]["untrusted"] is True

    r = client.post("/v1/get_context", json={"chunk_id": "hf_docs:p1.md#0", "mode": "chunk"})
    assert r.status_code == 200, r.text
    context = ContextResponse.model_validate(r.json())
    assert context.chunk_id == "hf_docs:p1.md#0" and context.found

    r = client.get("/v1/knowledge_bases")
    assert r.status_code == 200
    listing = KnowledgeBaseList.model_validate(r.json())
    assert [k.kb for k in listing.knowledge_bases] == ["hf_docs"]
    assert listing.knowledge_bases[0].description == "HF docs"


def _error(r: Any, status: int, category: str) -> dict[str, Any]:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == ERROR_KEYS and body["error"] == category
    assert r.headers["x-request-id"] == body["request_id"]
    return body


def test_http_errors_and_request_id() -> None:
    pipeline = FakeSearchPipeline("hf_docs")
    service = _service(FakePipelineBuilder({"hf_docs": pipeline}))
    with TestClient(build_http_app(service)) as client:
        assert service.wait_ready(5)
        r = client.post(
            "/v1/search_docs",
            json={"kb": "hf_docs", "query": "q"},
            headers={"X-Request-Id": "abc-1"},
        )
        assert r.json()["request_id"] == "abc-1" and r.headers["x-request-id"] == "abc-1"
        # A malformed id is replaced, not trusted into logs and records.
        r = client.post(
            "/v1/search_docs", json={"kb": "hf_docs", "query": "q"}, headers={"X-Request-Id": "a b"}
        )
        assert r.status_code == 200 and len(r.json()["request_id"]) == 32

        body = _error(
            client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "q", "k": 3}),
            400,
            "invalid_request",
        )
        assert "k" not in body["message"].split()  # fixed text, no echo of the input
        _error(client.post("/v1/search_docs", content=b"{not json"), 400, "invalid_request")
        _error(client.post("/v1/search_docs", json=["x"]), 400, "invalid_request")
        _error(
            client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "q", "top_k": "x"}),
            400,
            "invalid_request",
        )
        _error(
            client.post(
                "/v1/search_docs", json={"kb": "hf_docs", "query": "q", "filters": {"x": 1}}
            ),
            400,
            "invalid_filter",
        )
        _error(
            client.post("/v1/search_docs", json={"kb": "nope", "query": "q"}), 404, "kb_not_found"
        )
        _error(client.post("/v1/get_context", json={"chunk_id": "bad"}), 400, "invalid_request")

        pipeline.raises = RuntimeError("secret detail C:/x")
        r = client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "q"})
        _error(r, 500, "internal_error")
        assert "secret" not in r.text


def test_health_and_readiness() -> None:
    gate = threading.Event()
    builder = FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")}, prepare_gate=gate)
    service = _service(builder)
    with TestClient(build_http_app(service)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        r = client.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        r = client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "q", "deadline_ms": 50})
        _error(r, 503, "not_ready")
        assert r.headers["retry-after"] == "10"
        gate.set()
        assert service.wait_ready(5)
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        # Counts only: per-KB names would show KBs the caller may not see (s16-4).
        assert r.json()["ready_kbs"] == 1 and r.json()["unavailable_kbs"] == 0
        assert "knowledge_bases" not in r.json()
        assert client.get("/healthz").status_code == 200


def test_rejected_bodies_write_records(tmp_path: Path) -> None:
    # A body rejected before the service runs still gets a request span and a record line.
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)) as obs:
        service = _service()
        with TestClient(build_http_app(service)) as client:
            assert service.wait_ready(5)
            r = client.post(
                "/v1/search_docs", json={"kb": "hf_docs", "x": 1}, headers={"X-Request-Id": "b-1"}
            )
            _error(r, 400, "invalid_request")
            _error(client.post("/v1/get_context", content=b"{"), 400, "invalid_request")
        spans = [s.name for s in obs.span_exporter.get_finished_spans()]
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert [(r["tool"], r["flags"]) for r in records] == [
        ("search_docs", ["invalid_request"]),
        ("get_context", ["invalid_request"]),
    ]
    assert records[0]["request_id"] == "b-1" and records[0]["trace_id"]
    assert {"search_docs", "get_context"} <= set(spans)
