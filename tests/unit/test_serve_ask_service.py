"""Spec P3-01 AC15: `SearchService.ask_docs` (caller, ACL, arguments, errors, spans, record)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from elastic_transport import ConnectionTimeout

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.retrieve.hybrid import RetrievalError
from rag.security.access import Credentials
from rag.serve.errors import ServiceError
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel, memory_observability, metric_points
from rag.testing.agent import ScriptedSearchBackend, agent_settings, pages_result
from rag.testing.serve import FakePipelineBuilder
from rag.types import ErrorCategory

CREDS = Credentials(transport="http")
REPLY = "Use push_to_hub [1]. Or upload_folder [2]."


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _cfg(kb: str, callers: list[str] | None = None) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        kb=kb,
        description=f"{kb} pages",
        language="en",
        access=AccessConfig(callers=callers if callers is not None else ["*"]),
    )


def _service(
    *,
    backends: dict[str, ScriptedSearchBackend] | None = None,
    model: Any = None,
    agent: Any = None,
    load: bool = True,
    **settings: Any,
) -> SearchService:
    configs = {"hf_docs": _cfg("hf_docs"), "theirs": _cfg("theirs", ["bob"])}
    backends = backends or {kb: ScriptedSearchBackend(kb) for kb in configs}
    service = SearchService(
        agent_settings(**settings),
        configs=configs,
        builder=FakePipelineBuilder(backends),
        answer_model=model or StreamingFakeChatModel(default_response=REPLY),
        agent=agent,
    )
    if load:
        service.load()
    return service


def _ask(service: SearchService, **kwargs: Any) -> Any:
    kwargs.setdefault("kb", "hf_docs")
    kwargs.setdefault("question", "How do I push a model?")
    return asyncio.run(service.ask_docs(CREDS, **kwargs))


def _error(service: SearchService, **kwargs: Any) -> ServiceError:
    with pytest.raises(ServiceError) as info:
        _ask(service, **kwargs)
    return info.value


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_ask_docs_permissions_and_arguments() -> None:
    backends = {kb: ScriptedSearchBackend(kb) for kb in ("hf_docs", "theirs")}
    service = _service(
        backends=backends, service_caller="alice", service_acl_groups=["public", "t"]
    )

    response = _ask(service, session_id="s-1", request_id="req-1")
    assert (response.request_id, response.kb, response.status) == ("req-1", "hf_docs", "answered")
    assert response.answer == "Use push_to_hub. Or upload_folder."
    assert [c.untrusted for c in response.sources] == [True] * 3
    assert backends["hf_docs"].calls[0]["acl_groups"] == ("public", "t")  # the caller's groups

    # A KB the caller may not use looks like one that does not exist; nothing is retrieved.
    hidden = _error(service, kb="theirs")
    unknown = _error(service, kb="nope")
    assert hidden.category == unknown.category == ErrorCategory.KB_NOT_FOUND
    assert backends["theirs"].calls == []

    bad: list[dict[str, Any]] = [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": 5},
        {"session_id": "a:b"},
        {"session_id": ""},
        {"session_id": "s" * 65},
        {"deadline_ms": 0},
        {"deadline_ms": 60_001},
        {"deadline_ms": "100"},
    ]
    for kwargs in bad:
        assert _error(service, **kwargs).category == ErrorCategory.INVALID_REQUEST, kwargs
    assert len(backends["hf_docs"].calls) == 1

    # Not loaded yet: the call waits for its deadline, then not_ready.
    cold = _service(load=False)
    assert _error(cold, deadline_ms=100).category == ErrorCategory.NOT_READY


def test_ask_docs_errors_map_to_categories(caplog: pytest.LogCaptureFixture) -> None:
    cases: list[tuple[BaseException, str]] = [
        (
            RetrievalError("both routes failed", category=ErrorCategory.RETRIEVAL_FAILED),
            "retrieval_failed",
        ),
        (ConnectionTimeout("slow"), "retrieval_timeout"),
        (RuntimeError("secret detail C:/x"), "internal_error"),
    ]
    for raised, category in cases:
        backend = ScriptedSearchBackend("hf_docs", raises=raised)
        service = _service(backends={"hf_docs": backend, "theirs": ScriptedSearchBackend("t")})
        with caplog.at_level(logging.WARNING):
            err = _error(service, request_id="req-e")
        assert (err.category, err.request_id) == (category, "req-e")
        assert "secret" not in str(err) and "secret" not in repr(err)
    assert "secret detail" in caplog.text  # the details are in the log, with the request id

    # Waiting for the session's lock longer than the deadline: agent_deadline.
    service = _service()

    async def main() -> float:
        async with service.agent.locks.hold(service.agent.session_key("local", "hf_docs", "s1")):
            began = time.monotonic()
            with pytest.raises(ServiceError) as info:
                await service.ask_docs(
                    CREDS, kb="hf_docs", question="q", session_id="s1", deadline_ms=200
                )
            assert info.value.category == ErrorCategory.AGENT_DEADLINE
            return time.monotonic() - began

    assert asyncio.run(main()) < 5


def test_ask_docs_spans_and_business_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    degraded = pages_result("hf_docs", [1, 2], degraded=True)
    backend = ScriptedSearchBackend("hf_docs", default=degraded)
    with memory_observability(business_record_path=str(path)) as obs:
        service = _service(
            backends={"hf_docs": backend, "theirs": ScriptedSearchBackend("theirs")},
            business_record_path=str(path),
        )
        first = _ask(service, session_id="s-1", request_id="req-a")
        second = _ask(service, session_id="s-1", request_id="req-b")
        _error(service, kb="nope", request_id="req-c")
        spans = obs.span_exporter.get_finished_spans()
        requests = {
            (a.get(attr.TOOL), a[attr.OUTCOME]): p.value
            for a, p in metric_points(obs, attr.METRIC_REQUESTS)
        }
        steps = {a[attr.STEP] for a, _ in metric_points(obs, attr.METRIC_STEP_DURATION)}

    assert first.degraded and second.used_history and not second.cache_allowed
    mine = [s for s in spans if s.attributes.get(attr.REQUEST_ID) == "req-a"]
    by_name = {s.name: s for s in mine}
    root = by_name["ask_docs"]
    assert {"route", "retrieve", "answer", "generate", "verify"} <= set(by_name)
    by_id = {s.context.span_id: s for s in mine}

    def under_root(span: Any) -> bool:
        while span.parent is not None:
            if span.parent.span_id == root.context.span_id:
                return True
            span = by_id.get(span.parent.span_id)
            if span is None:
                return False
        return False

    for name in ("route", "retrieve", "answer", "generate", "verify"):
        assert under_root(by_name[name]), name
    assert root.attributes[attr.CALLER] == "local" and root.attributes[attr.KB] == "hf_docs"
    assert root.attributes[attr.DEGRADED] is True
    assert requests[("ask_docs", "ok")] == 2 and requests[("ask_docs", "error")] == 1
    assert {"route", "retrieve", "answer", "verify"} <= steps

    ok_a, ok_b, failed = _records(path)
    assert (ok_a["tool"], ok_a["request_id"], ok_a["caller"], ok_a["kb"]) == (
        "ask_docs",
        "req-a",
        "local",
        "hf_docs",
    )
    assert ok_a["trace_id"] == format(root.context.trace_id, "032x")
    assert [f["chunk_id"] for f in ok_a["final"]] == [c.chunk_id for c in first.sources][:2]
    assert ok_a["flags"] == ["degraded"] and ok_a["index_version"] == "kb_hf_docs_v1"
    extra = ok_a["extra"]
    assert extra["session_id"] == "s-1" and extra["question"] == "How do I push a model?"
    assert (extra["status"], extra["rounds"], extra["used_history"]) == ("answered", 1, False)
    assert extra["tokens_used"] > 0 and extra["coverage_unconfirmed"] is False
    assert extra["thread_id"] == first.thread_id and first.thread_id.startswith("s-1:")
    assert {"route", "retrieve", "answer", "verify", "request"} <= set(ok_a["timings_ms"])
    assert ok_b["extra"]["used_history"] is True
    assert (failed["tool"], failed["flags"], failed["request_id"]) == (
        "ask_docs",
        ["kb_not_found"],
        "req-c",
    )


def test_history_is_kept_per_caller_and_knowledge_base() -> None:
    # Two services share one runtime (as two callers of one process would); the callers use
    # the same session id.
    from rag.agent import AgentRuntime

    runtime = AgentRuntime.from_settings(agent_settings())
    alice = _service(agent=runtime, service_caller="alice")
    bob = _service(agent=runtime, service_caller="bob")
    assert _ask(alice, session_id="s1").used_history is False
    assert _ask(alice, session_id="s1").used_history is True
    assert _ask(bob, session_id="s1").used_history is False
