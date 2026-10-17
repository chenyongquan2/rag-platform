"""Spec P0-09 AC14: the HTTP app in-process over the real reranker and the fake-vector index.

Needs the compose Elasticsearch with `fake_kb_hf_docs` (50 documents, fake vectors, so no query
embedding costs money) and `BAAI/bge-reranker-base` in the local Hugging Face cache. The app's
lifespan loads the reranker in the background (about a minute), as the service does at start.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from starlette.testclient import TestClient

from rag.config import ServiceSettings, repo_path
from rag.eval.questions import load_question_set
from rag.observe import attributes as attr
from rag.observe import init_observability, shutdown_observability
from rag.retrieve import rerank as rerank_module
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService

pytestmark = pytest.mark.integration

QUESTION_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"
KNOWN_QID = "huggingface_doc_qa_eval:025"
KNOWN_PAGE = "hf_docs:huggingface/hf-endpoints-documentation/blob/main/docs/source/guides/logs.mdx"
PREFIX = "huggingface/hf-endpoints-documentation/"
LOAD_TIMEOUT_S = 300
SEARCH_FIELDS = {
    "request_id",
    "kb",
    "index_version",
    "data_generation",
    "config_version",
    "degraded",
    "rerank_timeout",
    "not_found",
    "error_categories",
    "timings_ms",
    "chunks",
    "below_threshold",
}
CHUNK_FIELDS = {"chunk_id", "text", "untrusted", "suspicious_instruction", "source_url"}


def known_question() -> str:
    questions = load_question_set(repo_path(QUESTION_SET), kb="hf_docs").questions
    return next(q.question for q in questions if q.qid == KNOWN_QID)


def require_fake_index(es: Elasticsearch) -> None:
    if not es.indices.exists_alias(name="fake_kb_hf_docs"):
        pytest.skip("alias fake_kb_hf_docs is missing (run the ingest job with EMBED_BACKEND=fake)")


def require_reranker(settings: ServiceSettings) -> None:
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(settings.reranker_model, "config.json"), str):
        pytest.skip(f"{settings.reranker_model} is not in the local Hugging Face cache")


@pytest.fixture
def records_path(tmp_path: Path) -> Path:
    return tmp_path / "business.jsonl"


@pytest.fixture
def settings(records_path: Path) -> ServiceSettings:
    return ServiceSettings(
        embed_backend="fake",
        search_index_prefix="fake_",
        otel_mode="memory",
        otel_auto_instrument=False,
        business_record_path=str(records_path),
        log_dir="",
        log_stderr_level="OFF",
    )


@pytest.fixture
def client(es: Elasticsearch, settings: ServiceSettings) -> Iterator[tuple[TestClient, Any]]:
    require_fake_index(es)
    require_reranker(settings)
    shutdown_observability()
    rerank_module.reset_rerank_runners()
    obs = init_observability(settings, "rag-serve-it", configure_logging=False)
    service = SearchService(settings)
    try:
        with TestClient(build_http_app(service)) as http:  # the lifespan starts the load
            yield http, obs
    finally:
        service.close()
        shutdown_observability()
        rerank_module.reset_rerank_runners()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_http_end_to_end(client: tuple[TestClient, Any], records_path: Path) -> None:
    http, obs = client
    assert http.get("/healthz").status_code == 200
    assert http.get("/readyz").status_code == 503  # the reranker is still loading
    began = time.monotonic()
    while http.get("/readyz").status_code != 200:
        assert time.monotonic() - began < LOAD_TIMEOUT_S, "the service did not get ready"
        time.sleep(1.0)
    ready = http.get("/readyz").json()
    assert ready["knowledge_bases"]["hf_docs"] == "ready" and ready["load_s"] > 0

    t0 = time.perf_counter()
    r = http.post(
        "/v1/search_docs",
        json={"kb": "hf_docs", "query": known_question(), "top_k": 5, "deadline_ms": 60_000},
        headers={"X-Request-Id": "it-search-025"},
    )
    search_s = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == SEARCH_FIELDS  # no routes, rerank or threshold
    assert body["request_id"] == "it-search-025" == r.headers["x-request-id"]
    assert body["kb"] == "hf_docs" and body["index_version"].startswith("fake_kb_hf_docs")
    assert body["config_version"] and body["data_generation"] >= 0
    assert body["chunks"] and body["chunks"][0]["doc_id"] == KNOWN_PAGE
    assert all(set(c) >= CHUNK_FIELDS and c["untrusted"] is True for c in body["chunks"])
    assert body["chunks"][0]["source_version"]
    assert body["timings_ms"]["request"] > 0
    print(f"\nsearch_docs over HTTP: {search_s * 1000:.0f} ms, timings {body['timings_ms']}")

    narrowed = http.post(
        "/v1/search_docs",
        json={
            "kb": "hf_docs",
            "query": known_question(),
            "filters": {"source_rel_prefix": PREFIX},
            "deadline_ms": 60_000,
        },
    ).json()
    assert narrowed["chunks"]
    assert all(c["source_rel"].startswith(PREFIX) for c in narrowed["chunks"])

    bad = http.post(
        "/v1/search_docs", json={"kb": "hf_docs", "query": "q", "filters": {"author": "x"}}
    )
    assert bad.status_code == 400 and bad.json()["error"] == "invalid_filter"
    assert set(bad.json()) == {"error", "request_id", "message"}

    top = body["chunks"][0]["chunk_id"]
    context = http.post("/v1/get_context", json={"chunk_id": top, "mode": "neighbors"}).json()
    assert context["found"] is True and top in {c["chunk_id"] for c in context["chunks"]}
    listing = http.get("/v1/knowledge_bases").json()
    assert "hf_docs" in {k["kb"] for k in listing["knowledge_bases"]}

    spans = {
        s.attributes.get(attr.REQUEST_ID): s
        for s in obs.span_exporter.get_finished_spans()
        if s.name == "search_docs"
    }
    record = next(r for r in _records(records_path) if r["request_id"] == "it-search-025")
    assert record["tool"] == "search_docs" and record["kb"] == "hf_docs"
    assert record["trace_id"] == format(spans["it-search-025"].context.trace_id, "032x")
    assert record["rerank"] and record["threshold"] is not None
    assert record["extra"]["query"] == known_question()
    tools = [r["tool"] for r in _records(records_path)]
    assert tools.count("search_docs") == 3 and "get_context" in tools
    assert "list_knowledge_bases" in tools
