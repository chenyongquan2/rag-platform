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
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability, metric_points
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


# ---- P3-02: rewrite and clarify through the service -----------------------------------------

VAGUE = "How do I fix this error?"


def _asking_service(
    helper: Any = None, *, path: Path | None = None, agent: Any = None, **settings: Any
) -> SearchService:
    """A service whose KB rewrites, and asks back about a question that names nothing."""
    from rag.config import AgentConfig

    cfg = KnowledgeBaseConfig(
        kb="hf_docs",
        description="hf_docs pages",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(rewrite=True, clarify=True, route_model=True),
    )
    service = SearchService(
        agent_settings(**({"business_record_path": str(path)} if path else {}), **settings),
        configs={"hf_docs": cfg},
        builder=FakePipelineBuilder({"hf_docs": ScriptedSearchBackend("hf_docs")}),
        answer_model=StreamingFakeChatModel(default_response=REPLY),
        helper_model=helper or FakeChatModel(responses=[]),
        agent=agent,
    )
    service.load()
    return service


def test_resume_arguments_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        service = _asking_service(path=path)
        first = _ask(service, question=VAGUE, session_id="s1", request_id="req-1")
        assert first.status == "need_clarification" and first.resume_token
        assert "Could you be more specific?" in first.answer and first.sources == []
        token = first.resume_token

        def refused(category: str, **kwargs: Any) -> ServiceError:
            err = _error(service, **kwargs)
            assert err.category == category, kwargs
            return err

        # Arguments that cannot be a resume: the request itself is wrong.
        for kwargs in (
            {"resume_token": token},  # no answer
            {"resume_token": token, "answer": ""},
            {"resume_token": token, "answer": "   "},
            {"resume_token": token, "answer": "x" * 2001},
            {"resume_token": token, "answer": 5},
            {"resume_token": 5, "answer": "a reply"},
            {"answer": "a reply"},  # an answer with no token
        ):
            refused(ErrorCategory.INVALID_REQUEST, session_id="s1", **kwargs)
        # A token that cannot be ours (shape) and one that is not known look the same.
        for bad in ("short", "has space in it but is long enough", "ä" * 30, token + "x"):
            err = refused(
                ErrorCategory.INVALID_RESUME_TOKEN,
                session_id="s1",
                resume_token=bad,
                answer="a reply",
            )
            assert bad not in str(err) and bad not in repr(err)
        # Another session and another caller (a service of the same process with its own
        # configured caller): refused, and the token is still good for its owner.
        refused(ErrorCategory.INVALID_RESUME_TOKEN, session_id="s2", resume_token=token, answer="r")
        bob = _asking_service(agent=service.agent, service_caller="bob")
        with pytest.raises(ServiceError) as info:
            asyncio.run(
                bob.ask_docs(
                    CREDS,
                    kb="hf_docs",
                    question=VAGUE,
                    session_id="s1",
                    resume_token=token,
                    answer="a reply",
                )
            )
        assert info.value.category == ErrorCategory.INVALID_RESUME_TOKEN

        done = _ask(
            service,
            question="(the question need not be repeated right)",
            session_id="s1",
            resume_token=token,
            answer="transformers raises KeyError 'llama'",
            request_id="req-2",
        )
        assert (done.status, done.clarified, done.resume_token) == ("answered", True, None)
        assert done.request_id == "req-2" and done.thread_id == first.thread_id
        refused(
            ErrorCategory.INVALID_RESUME_TOKEN, session_id="s1", resume_token=token, answer="r"
        )  # used once

    records = _records(path)
    by_id = {r["request_id"]: r for r in records}
    assert by_id["req-1"]["extra"]["status"] == "need_clarification"
    assert by_id["req-2"]["extra"]["resumed"] is True and by_id["req-2"]["extra"]["clarified"]
    # The record of the resumed call names the question that was asked, not what the caller sent
    # along with its reply; a refused resume has no question to name.
    assert by_id["req-2"]["extra"]["question"] == VAGUE
    assert by_id["req-1"]["extra"]["question"] == VAGUE
    # No record or log line carries the token itself.
    assert token not in path.read_text("utf-8")
    refused_records = [r for r in records if r["flags"] == ["invalid_resume_token"]]
    assert refused_records and all(token not in json.dumps(r) for r in refused_records)
    assert all(r["extra"]["question"] is None for r in refused_records)


def test_rewrite_in_the_business_record(tmp_path: Path) -> None:
    from rag.testing.agent import structured_reply

    question = "How do I push a model to the hub?"
    statement = "Push a model to the hub with push_to_hub"
    helper = FakeChatModel(
        responses=[
            structured_reply(
                "RewritePlan", standalone_question=question, search_queries=[statement]
            ),
        ]
    )
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        service = _asking_service(helper, path=path)
        response = _ask(service, question=question, request_id="req-r")
        # The rewrite fails (script says garbage next): a fallback, flagged degraded.
        helper.responses.append("not a tool call")
        failed = _ask(service, question="How do I load a model?", request_id="req-f")

    assert response.search_queries == [statement] and response.standalone_question is None
    assert response.fallbacks == [] and response.degraded is False
    assert failed.fallbacks == ["rewrite:parse_failed"] and failed.degraded is True
    assert "degraded" in failed.error_categories
    ok, bad = _records(path)
    assert ok["extra"]["search_queries"] == [statement]
    assert ok["extra"]["standalone_question"] is None and ok["extra"]["fallbacks"] == []
    assert ok["extra"]["tokens_used"] > 0
    assert bad["extra"]["fallbacks"] == ["rewrite:parse_failed"] and bad["flags"] == ["degraded"]


def test_the_judge_result_reaches_the_response_and_the_business_record(tmp_path: Path) -> None:
    """Spec P3-03 AC23: sufficiency, missing and the nearest chunks are part of the response,
    and the record names the verdict."""
    from rag.config import AgentConfig
    from rag.testing.agent import judge_reply, scored_result

    cfg = KnowledgeBaseConfig(
        kb="hf_docs",
        description="hf_docs pages",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(judge=True),
    )
    question = "How do I push a model?"
    backend = ScriptedSearchBackend(
        "hf_docs",
        by_question={question: scored_result("hf_docs", [0.1, 0.05])},
        default=scored_result("hf_docs", [0.9]),
    )
    helper = FakeChatModel(responses=[judge_reply(True)])
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        service = SearchService(
            agent_settings(business_record_path=str(path)),
            configs={"hf_docs": cfg},
            builder=FakePipelineBuilder({"hf_docs": backend}),
            answer_model=StreamingFakeChatModel(default_response=REPLY),
            helper_model=helper,
        )
        service.load()
        nothing = _ask(service, question=question, request_id="req-n")
        found = _ask(service, question="Where is the cache?", request_id="req-f")

    # Nothing passed the threshold: not enough, the closest passages are returned apart from
    # the sources. (The KB allows no rewrite, so no new phrasing: the run says why it stopped.)
    assert nothing.status == "not_found" and nothing.sufficiency == "not_enough"
    assert nothing.missing == [question] and nothing.sources == []
    assert [c.chunk_id for c in nothing.nearest] == ["hf_docs:p1.md#0", "hf_docs:p2.md#0"]
    assert all(c.untrusted for c in nothing.nearest)
    assert nothing.stop_reason == "no_new_query" and nothing.coverage_unconfirmed
    assert found.status == "answered" and found.sufficiency == "enough" and found.nearest == []
    assert len(helper.calls) == 1  # only the question with chunks above the threshold was judged
    bad, good = _records(path)
    assert bad["extra"]["sufficiency"] == "not_enough" and bad["extra"]["missing"] == [question]
    assert good["extra"]["sufficiency"] == "enough" and good["extra"]["missing"] == []
