"""Spec P0-09 AC3-AC10, AC13: the search service behind the MCP tools and the HTTP API.

Fake pipelines (`rag.testing.serve`) stand in for `SearchPipeline`; observability runs in memory.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.config import AccessConfig, KnowledgeBaseConfig, RetrievalConfig
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.retrieve.hybrid import RetrievalError
from rag.security.access import Credentials
from rag.serve.errors import ServiceError
from rag.serve.service import SearchService
from rag.testing import memory_observability, metric_points, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline, retrieval_result
from rag.types import ErrorCategory, ThresholdDecision

CREDS = Credentials(transport="http")


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _cfg(kb: str, callers: list[str] | None = None, **retrieval: Any) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        kb=kb,
        description=f"{kb} pages",
        language="en",
        access=AccessConfig(callers=callers if callers is not None else ["*"]),
        retrieval=RetrievalConfig(**retrieval),
    )


def _service(
    *,
    configs: dict[str, KnowledgeBaseConfig] | None = None,
    builder: FakePipelineBuilder | None = None,
    load: bool = True,
    **settings: Any,
) -> SearchService:
    configs = configs if configs is not None else {"hf_docs": _cfg("hf_docs")}
    if builder is None:
        builder = FakePipelineBuilder({kb: FakeSearchPipeline(kb) for kb in configs})
    service = SearchService(observe_settings(**settings), configs=configs, builder=builder)
    if load:
        service.load()
    return service


def _error(fn: Any, *args: Any, **kwargs: Any) -> ServiceError:
    with pytest.raises(ServiceError) as info:
        fn(*args, **kwargs)
    return info.value


def test_serve_settings_defaults() -> None:
    s = observe_settings()
    assert (s.service_caller, s.service_acl_groups) == ("local", ["public"])
    assert (s.service_host, s.service_port) == ("127.0.0.1", 28080)
    assert s.search_index_prefix == ""
    assert s.search_max_deadline_ms == 60_000
    assert s.search_max_query_chars == 2000
    with pytest.raises(ValueError):
        observe_settings(service_acl_groups=[])
    with pytest.raises(ValueError):
        observe_settings(search_max_deadline_ms=5000, search_deadline_ms=10_000)


def test_list_shows_only_permitted_kbs() -> None:
    configs = {
        "open": _cfg("open"),
        "mine": _cfg("mine", ["local"]),
        "theirs": _cfg("theirs", ["bob"]),
    }
    service = _service(configs=configs)
    listing = service.list_knowledge_bases(CREDS, request_id="req-l")
    assert listing.request_id == "req-l"
    assert [(k.kb, k.description, k.available) for k in listing.knowledge_bases] == [
        ("mine", "mine pages", True),
        ("open", "open pages", True),
    ]
    info = listing.knowledge_bases[0]
    assert (info.language, info.default_top_k, info.max_top_k) == ("en", 8, 30)


def test_hidden_and_unknown_kb_give_same_error() -> None:
    configs = {"open": _cfg("open"), "theirs": _cfg("theirs", ["bob"])}
    builder = FakePipelineBuilder({kb: FakeSearchPipeline(kb) for kb in configs})
    service = _service(configs=configs, builder=builder)
    hidden = _error(service.search_docs, CREDS, kb="theirs", query="q")
    unknown = _error(service.search_docs, CREDS, kb="nope", query="q")
    assert hidden.category == unknown.category == ErrorCategory.KB_NOT_FOUND
    context_hidden = _error(service.get_context, CREDS, chunk_id="theirs:p.md#0")
    context_unknown = _error(service.get_context, CREDS, chunk_id="nope:p.md#0")
    assert context_hidden.category == context_unknown.category == ErrorCategory.KB_NOT_FOUND
    assert (
        builder.pipelines["theirs"].calls == [] and builder.pipelines["theirs"].context_calls == []
    )


def test_every_tool_uses_caller_acl_groups() -> None:
    service = _service(service_caller="alice", service_acl_groups=["public", "team-a"])
    pipeline = service.pipeline("hf_docs")
    assert isinstance(pipeline, FakeSearchPipeline)
    service.search_docs(CREDS, kb="hf_docs", query="how to log?")
    service.get_context(CREDS, chunk_id="hf_docs:p1.md#0", mode="neighbors")
    assert pipeline.calls[0]["acl_groups"] == ("public", "team-a")
    assert pipeline.context_calls[0]["acl_groups"] == ("public", "team-a")
    assert pipeline.context_calls[0]["mode"] == "neighbors"


def test_unknown_filter_key_rejected() -> None:
    service = _service()
    pipeline = service.pipeline("hf_docs")
    for filters in ({"acl_groups": ["admin"]}, {"author": "x"}, {"language": {"$ne": 1}}, ["x"]):
        err = _error(service.search_docs, CREDS, kb="hf_docs", query="q", filters=filters)
        assert err.category == ErrorCategory.INVALID_FILTER, filters
    assert pipeline.calls == []  # rejected before retrieval runs
    ok = service.search_docs(
        CREDS, kb="hf_docs", query="q", filters={"language": "en", "source_rel_prefix": "docs/"}
    )
    assert ok.kb == "hf_docs"
    assert pipeline.calls[0]["filters"] == {"language": "en", "source_rel_prefix": "docs/"}


def test_invalid_arguments_rejected() -> None:
    service = _service(search_max_query_chars=50)
    pipeline = service.pipeline("hf_docs")
    bad_search = [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 51},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": 31},  # rerank_n is 30
        {"query": "q", "top_k": True},
        {"query": "q", "deadline_ms": 0},
        {"query": "q", "deadline_ms": 60_001},
    ]
    for kwargs in bad_search:
        err = _error(service.search_docs, CREDS, kb="hf_docs", **kwargs)
        assert err.category == ErrorCategory.INVALID_REQUEST, kwargs
    for kwargs in (
        {"chunk_id": ""},
        {"chunk_id": "hf_docs:p1.md"},  # no #<n>
        {"chunk_id": "no-kb#0"},
        {"chunk_id": "hf_docs:p1.md#0", "mode": "everything"},
    ):
        err = _error(service.get_context, CREDS, **kwargs)
        assert err.category == ErrorCategory.INVALID_REQUEST, kwargs
    assert pipeline.calls == [] and pipeline.context_calls == []


def test_defaults_from_config() -> None:
    configs = {"hf_docs": _cfg("hf_docs", final_k=5, rerank_n=12)}
    service = _service(configs=configs, search_deadline_ms=4000)
    pipeline = service.pipeline("hf_docs")
    service.search_docs(CREDS, kb="hf_docs", query="q")
    call = pipeline.calls[0]
    assert call["top_k"] == 5
    assert 3500 < call["deadline_ms"] <= 4000
    assert call["filters"] == {}
    service.search_docs(CREDS, kb="hf_docs", query="q", top_k=12, deadline_ms=900)
    assert pipeline.calls[1]["top_k"] == 12 and pipeline.calls[1]["deadline_ms"] <= 900
    service.get_context(CREDS, chunk_id="hf_docs:p1.md#0")
    assert pipeline.context_calls[0]["mode"] is None  # the KB's own default mode


def test_error_details_only_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    pipeline = FakeSearchPipeline("hf_docs", raises=RuntimeError("secret detail C:/x/y"))
    service = _service(builder=FakePipelineBuilder({"hf_docs": pipeline}))
    with caplog.at_level(logging.WARNING, logger="rag.serve"):
        err = _error(service.search_docs, CREDS, kb="hf_docs", query="q", request_id="req-e")
    assert (err.category, err.request_id) == (ErrorCategory.INTERNAL_ERROR, "req-e")
    assert "secret" not in str(err) and "secret" not in repr(err)
    logged = "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert "req-e" in logged and "secret detail" in logged

    timeout = FakeSearchPipeline(
        "hf_docs", raises=RetrievalError("x", category=ErrorCategory.RETRIEVAL_TIMEOUT)
    )
    service = _service(builder=FakePipelineBuilder({"hf_docs": timeout}))
    err = _error(service.search_docs, CREDS, kb="hf_docs", query="q")
    assert err.category == ErrorCategory.RETRIEVAL_TIMEOUT and len(err.request_id) == 32
    timeout.context_raises = RetrievalError("x", category=ErrorCategory.RETRIEVAL_TIMEOUT)
    err = _error(service.get_context, CREDS, chunk_id="hf_docs:p1.md#0")
    assert err.category == ErrorCategory.RETRIEVAL_TIMEOUT


def test_readiness_and_wait_for_load(caplog: pytest.LogCaptureFixture) -> None:
    gate = threading.Event()
    builder = FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")}, prepare_gate=gate)
    service = _service(builder=builder, load=False)
    assert service.status()["status"] == "starting" and not service.ready
    with caplog.at_level(logging.INFO, logger="rag.serve"):
        service.start()
        service.start()  # idempotent: one load per process
        assert service.status()["status"] == "loading"
        # Config-only answers work while loading; searches wait for the load within their
        # deadline, then give up with not_ready.
        assert [k.kb for k in service.list_knowledge_bases(CREDS).knowledge_bases] == ["hf_docs"]
        began = time.monotonic()
        err = _error(service.search_docs, CREDS, kb="hf_docs", query="q", deadline_ms=200)
        assert err.category == ErrorCategory.NOT_READY
        assert 0.15 <= time.monotonic() - began < 2.0

        # A search that arrives during the load runs once the load ends within its deadline.
        results: list[Any] = []
        waiter = threading.Thread(
            target=lambda: results.append(
                service.search_docs(CREDS, kb="hf_docs", query="q", deadline_ms=5000)
            )
        )
        waiter.start()
        time.sleep(0.1)
        gate.set()
        waiter.join(5)
        assert service.wait_ready(5)
    assert len(results) == 1 and results[0].kb == "hf_docs"
    assert builder.prepared == 1
    status = service.status()
    assert status["status"] == "ready" and status["load_s"] >= 0.0
    assert status["knowledge_bases"] == {"hf_docs": "ready"}
    assert any("ready in" in r.getMessage() for r in caplog.records)

    failing = FakePipelineBuilder({}, prepare_error=OSError("model missing"))
    broken = _service(builder=failing, load=False)
    broken.start()
    assert not broken.wait_ready(5)
    assert broken.status()["status"] == "failed"
    began = time.monotonic()
    err = _error(broken.search_docs, CREDS, kb="hf_docs", query="q", deadline_ms=5000)
    assert err.category == ErrorCategory.NOT_READY and time.monotonic() - began < 1.0
    assert _error(broken.get_context, CREDS, chunk_id="hf_docs:p#0").category == "not_ready"


def test_failed_kb_is_unavailable() -> None:
    configs = {"good": _cfg("good"), "broken": _cfg("broken")}
    builder = FakePipelineBuilder(
        {"good": FakeSearchPipeline("good")},
        failures={"broken": RuntimeError("alias kb_broken points to no index")},
    )
    service = _service(configs=configs, builder=builder)
    assert service.ready
    assert service.status()["knowledge_bases"] == {"broken": "unavailable", "good": "ready"}
    listing = service.list_knowledge_bases(CREDS).knowledge_bases
    assert [(k.kb, k.available) for k in listing] == [("broken", False), ("good", True)]
    err = _error(service.search_docs, CREDS, kb="broken", query="q")
    assert err.category == ErrorCategory.RETRIEVAL_FAILED
    assert service.search_docs(CREDS, kb="good", query="q").kb == "good"


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_search_span_metrics_and_business_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    result = retrieval_result(
        "hf_docs",
        n=3,
        timings_ms={"embed": 5.0, "search": 12.0, "rerank": 300.0, "total": 320.0},
        threshold=ThresholdDecision(threshold=0.3, top_score=0.95, passed=True, kept=3),
    )
    pipeline = FakeSearchPipeline("hf_docs", result=result, config_version="cfg-7")
    with memory_observability(business_record_path=str(path)) as obs:
        service = _service(builder=FakePipelineBuilder({"hf_docs": pipeline}))
        response = service.search_docs(
            CREDS, kb="hf_docs", query="how to log?", top_k=2, request_id="req-s"
        )
        spans = {s.name: s for s in obs.span_exporter.get_finished_spans()}
        steps = {
            a[attr.STEP]: p.count
            for a, p in metric_points(obs, attr.METRIC_STEP_DURATION)
            if a.get(attr.KB) == "hf_docs"
        }
        top_scores = metric_points(obs, attr.METRIC_TOP_RERANK_SCORE)

    assert response.request_id == "req-s" and len(response.chunks) == 2
    assert "request" in response.timings_ms
    root, layer = spans["search_docs"], spans["retrieve"]
    assert layer.parent is not None and layer.parent.span_id == root.context.span_id
    assert root.attributes[attr.REQUEST_ID] == "req-s"
    assert root.attributes[attr.CALLER] == "local"
    assert root.attributes[attr.KB] == "hf_docs"
    assert root.attributes[attr.CONFIG_VERSION] == "cfg-7"
    assert root.attributes[attr.HITS] == 2
    assert layer.attributes[attr.TOP_K] == 2
    assert layer.attributes[attr.RERANK_N] == 30 and layer.attributes[attr.THRESHOLD] == 0.3
    assert {"search_docs", "retrieve", "embed", "search", "rerank"} <= set(steps)
    assert "total" not in steps
    assert [p.count for _, p in top_scores] == [1]

    (record,) = _records(path)
    assert record["request_id"] == "req-s"
    assert record["trace_id"] == format(root.context.trace_id, "032x")
    assert (record["tool"], record["caller"], record["kb"]) == ("search_docs", "local", "hf_docs")
    assert record["config_version"] == "cfg-7"
    assert record["index_version"] == result.index_version
    assert set(record["routes"]) == {"bm25", "vector"}
    assert [r["chunk_id"] for r in record["rerank"]] == [h.chunk_id for h in result.rerank]
    assert record["threshold"]["top_score"] == 0.95
    assert [f["chunk_id"] for f in record["final"]] == [c.chunk_id for c in response.chunks]
    assert record["extra"]["query"] == "how to log?"
    assert record["extra"]["top_k"] == 2 and record["extra"]["not_found"] is False
    assert "request" in record["timings_ms"]


def test_error_and_other_tools_write_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)) as obs:
        service = _service()
        _error(service.search_docs, CREDS, kb="hf_docs", query="q", filters={"x": 1})
        _error(service.search_docs, CREDS, kb="Nope!" * 40, query="q")
        service.get_context(CREDS, chunk_id="hf_docs:p1.md#0", request_id="req-c")
        service.list_knowledge_bases(CREDS, request_id="req-l")
        errors = {
            (a.get(attr.KB), a[attr.ERROR_CATEGORY]): p.value
            for a, p in metric_points(obs, attr.METRIC_ERRORS)
        }
        spans = [s.name for s in obs.span_exporter.get_finished_spans()]

    bad_filter, unknown_kb, context, listing = _records(path)
    assert bad_filter["tool"] == "search_docs" and bad_filter["flags"] == ["invalid_filter"]
    assert bad_filter["kb"] == "hf_docs" and bad_filter["extra"]["query"] == "q"
    assert bad_filter["trace_id"]
    # Unknown knowledge bases are not put into kb fields, span attributes or metric labels
    # (caller-chosen strings would make unbounded metric series); the record keeps a prefix.
    assert unknown_kb["kb"] == "" and unknown_kb["flags"] == ["kb_not_found"]
    assert unknown_kb["extra"]["requested_kb"] == ("Nope!" * 40)[:64]
    assert (None, "kb_not_found") in errors and ("hf_docs", "invalid_filter") in errors
    assert context["tool"] == "get_context" and context["request_id"] == "req-c"
    assert context["final"] and context["extra"]["chunk_id"] == "hf_docs:p1.md#0"
    assert listing["tool"] == "list_knowledge_bases" and listing["request_id"] == "req-l"
    assert listing["extra"]["knowledge_bases"] == ["hf_docs"]
    assert spans.count("search_docs") == 2
    assert {"get_context", "list_knowledge_bases"} <= set(spans)
