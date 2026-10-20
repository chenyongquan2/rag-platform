"""Spec P3-04 AC6-AC16, AC26: `POST /v1/ask_docs/stream` over the Starlette test client (the whole
body is read before it is parsed, so what is checked here is the content and the order of the
events; that a disconnect really stops the run is in `test_serve_stream_cancel.py`)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.retrieve.hybrid import RetrievalError
from rag.serve.http_app import build_http_app
from rag.serve.schema import AskResponse
from rag.serve.service import SearchService
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.testing.agent import (
    ScriptedSearchBackend,
    SlowStreamingChatModel,
    agent_settings,
)
from rag.testing.serve import FakePipelineBuilder
from rag.types import ErrorCategory

REPLY = "Push it with push_to_hub [1]. Or upload_folder [2]."
VAGUE = "How do I fix this error?"
BODY = {"kb": "hf_docs", "question": "How do I push a model?", "session_id": "s1"}

Parsed = tuple[str, Any]


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def parse_sse(text: str) -> list[Parsed]:
    """Events (`("progress", {...})`) and comment lines (`("comment", "keep-alive")`) of a
    response body, as the page parses them: blocks end at a blank line, `data:` lines of one
    event are joined with a newline."""
    items: list[Parsed] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith(":"):
                items.append(("comment", line[1:].strip()))
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
        if data:
            items.append((name, json.loads("\n".join(data))))
    return items


def events(text: str) -> list[Parsed]:
    return [i for i in parse_sse(text) if i[0] != "comment"]


def _configs(**agent: Any) -> dict[str, KnowledgeBaseConfig]:
    return {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs",
            description="Hugging Face library documentation.",
            language="en",
            access=AccessConfig(callers=["*"]),
            agent=AgentConfig(**agent),
        ),
        "secret_kb": KnowledgeBaseConfig(
            kb="secret_kb", description="Someone else's pages", access=AccessConfig(callers=["bob"])
        ),
    }


def _service(
    *,
    backend: ScriptedSearchBackend | None = None,
    model: Any = None,
    helper: Any = None,
    agent: dict[str, Any] | None = None,
    builder: FakePipelineBuilder | None = None,
    load: bool = True,
    **settings: Any,
) -> tuple[SearchService, ScriptedSearchBackend]:
    backend = backend or ScriptedSearchBackend("hf_docs")
    service = SearchService(
        agent_settings(**settings),
        configs=_configs(**(agent or {})),
        builder=builder
        or FakePipelineBuilder({"hf_docs": backend, "secret_kb": ScriptedSearchBackend("s")}),
        answer_model=model or StreamingFakeChatModel(default_response=REPLY, chunk_chars=6),
        helper_model=helper or FakeChatModel(responses=[]),
    )
    if load:
        service.load()
    return service, backend


def _stream(service: SearchService, body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)
        return client.post("/v1/ask_docs/stream", json=BODY if body is None else body, **kwargs)


# ---- AC6, AC7, AC9, AC13 --------------------------------------------------------------------


def test_event_order_progress_token_answer_done() -> None:
    service, _ = _service(agent={"judge": False})
    r = _stream(service)
    assert r.status_code == 200, r.text
    got = events(r.text)
    names = [n for n, _ in got]
    assert names[0] == "progress" and names[-1] == "done" and names.count("done") == 1
    assert names.count("answer") == 1 and "error" not in names
    # Tokens only between the answer node's start and end; the answer event after every token
    # and after the verify node; done last.
    marks = [(n, d.get("node"), d.get("phase")) for n, d in got]
    first_token = names.index("token")
    last_token = len(names) - 1 - names[::-1].index("token")
    assert marks[first_token - 1] == ("progress", "answer", "start")
    assert marks[last_token + 1] == ("progress", "answer", "end")
    assert names.index("answer") > last_token
    assert marks.index(("progress", "verify", "end")) < names.index("answer")
    # Every node ends after it started, and no node starts before the one before it ended.
    open_nodes: list[str] = []
    for n, d in got:
        if n != "progress" or d["node"] == "queued":
            continue
        if d["phase"] == "start":
            assert not open_nodes, f"{d['node']} started inside {open_nodes}"
            open_nodes.append(d["node"])
        elif d["phase"] == "end":
            assert open_nodes == [d["node"]]
            open_nodes.clear()
    assert open_nodes == []
    assert [d["node"] for n, d in got if n == "progress" and d["phase"] == "start"] == [
        "route",
        "rewrite",
        "retrieve",
        "judge",
        "answer",
        "verify",
    ]


def test_answer_event_carries_the_checked_answer_and_the_removed_markers() -> None:
    model = StreamingFakeChatModel(default_response="Push it [1]. A bad one [9].", chunk_chars=5)
    service, _ = _service(model=model)
    got = events(_stream(service).text)
    answer = next(d for n, d in got if n == "answer")
    response = AskResponse.model_validate(answer)  # the same shape as POST /v1/ask_docs
    assert response.status == "answered" and response.found
    assert response.citation_check.removed_markers == [9]
    assert [s.text for s in response.sentences] == ["Push it.", "A bad one."]
    assert [c.marker for c in response.sentences[0].citations] == [1]
    assert len(response.sources) == 3 and all(c.untrusted for c in response.sources)
    # The text on the wire was shown before the check: it still holds the marker [9].
    streamed = "".join(d["text"] for n, d in got if n == "token")
    assert streamed == "Push it [1]. A bad one [9]."


def test_done_event_has_timings_rounds_flags_and_token_usage() -> None:
    service, _ = _service(model=StreamingFakeChatModel(default_response=REPLY, chunk_chars=6))
    r = _stream(service, headers={"X-Request-Id": "req-done"})
    done = next(d for n, d in events(r.text) if n == "done")
    assert set(done) == {
        "request_id",
        "status",
        "rounds",
        "degraded",
        "rerank_timeout",
        "coverage_unconfirmed",
        "stop_reason",
        "tokens_used",
        "timings_ms",
    }
    assert done["request_id"] == "req-done" and done["status"] == "answered"
    assert done["rounds"] == 1 and done["degraded"] is False and done["stop_reason"] is None
    assert done["tokens_used"] > 0
    t = done["timings_ms"]
    assert {"route", "retrieve", "answer", "request", "first_progress", "first_token"} <= set(t)
    assert 0 <= t["first_progress"] <= t["first_token"] <= t["request"]


def test_stream_response_headers() -> None:
    service, _ = _service()
    r = _stream(service, headers={"X-Request-Id": "req-h"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"
    assert r.headers["x-request-id"] == "req-h"
    assert re.match(r"^event: progress\ndata: \{", r.text)  # one event per block, one line of JSON


# ---- AC8 ------------------------------------------------------------------------------------


def test_need_clarification_then_resume_over_the_stream() -> None:
    service, _ = _service(
        agent={"rewrite": True, "clarify": True, "route_model": True},
        builder=None,
    )
    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)
        first = client.post(
            "/v1/ask_docs/stream",
            json={"kb": "hf_docs", "question": VAGUE, "session_id": "s1"},
            headers={"X-Request-Id": "req-1"},
        )
        got = events(first.text)
        names = [n for n, _ in got]
        assert names[-2:] == ["need_clarification", "done"] and "answer" not in names
        ask = got[-2][1]
        assert set(ask) == {"request_id", "question", "resume_token", "session_id"}
        assert "Could you be more specific?" in ask["question"] and ask["request_id"] == "req-1"
        assert ask["session_id"] == "s1" and len(ask["resume_token"]) >= 16
        assert got[-1][1]["status"] == "need_clarification"
        # No model text was streamed while the run was stopped at the question.
        assert "token" not in names

        second = client.post(
            "/v1/ask_docs/stream",
            json={
                "kb": "hf_docs",
                "question": "(not needed again)",
                "session_id": "s1",
                "resume_token": ask["resume_token"],
                "answer": "transformers raises KeyError 'llama'",
            },
        )
        got2 = events(second.text)
        answer = AskResponse.model_validate(next(d for n, d in got2 if n == "answer"))
        assert answer.status == "answered" and answer.clarified and answer.resume_token is None
        assert got2[-1][0] == "done" and "token" in [n for n, _ in got2]
        # Used once: the same token is now refused before any byte, as JSON.
        again = client.post(
            "/v1/ask_docs/stream",
            json={
                "kb": "hf_docs",
                "question": "x",
                "session_id": "s1",
                "resume_token": ask["resume_token"],
                "answer": "again",
            },
        )
        assert again.status_code == 400 and again.json()["error"] == "invalid_resume_token"


# ---- AC10 -----------------------------------------------------------------------------------

BAD_REQUESTS: list[tuple[str, Any, int, str]] = [
    ("not json", b"{not json", 400, "invalid_request"),
    ("unknown field", {**BODY, "colour": "red"}, 400, "invalid_request"),
    ("blank question", {**BODY, "question": "  "}, 400, "invalid_request"),
    ("bad session id", {**BODY, "session_id": "no spaces allowed"}, 400, "invalid_request"),
    ("deadline out of range", {**BODY, "deadline_ms": 0}, 400, "invalid_request"),
    ("answer without a token", {**BODY, "answer": "x"}, 400, "invalid_request"),
    ("unknown kb", {**BODY, "kb": "nope"}, 404, "kb_not_found"),
    ("hidden kb", {**BODY, "kb": "secret_kb"}, 404, "kb_not_found"),
    (
        "malformed token",
        {**BODY, "resume_token": "short", "answer": "a"},
        400,
        "invalid_resume_token",
    ),
    (
        "unknown token",
        {**BODY, "resume_token": "A" * 32, "answer": "a reply"},
        400,
        "invalid_resume_token",
    ),
]


@pytest.mark.parametrize(("label", "body", "status", "category"), BAD_REQUESTS)
def test_errors_before_streaming_are_json_with_the_status_of_ask_docs(
    label: str, body: Any, status: int, category: str
) -> None:
    service, _ = _service()
    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)
        kwargs: dict[str, Any] = {"content": body} if isinstance(body, bytes) else {"json": body}
        kwargs["headers"] = {"X-Request-Id": "req-bad", "Content-Type": "application/json"}
        plain = client.post("/v1/ask_docs", **kwargs)
        streamed = client.post("/v1/ask_docs/stream", **kwargs)
    assert streamed.status_code == plain.status_code == status, label
    assert streamed.headers["content-type"].startswith("application/json"), label
    assert streamed.json() == plain.json()
    assert set(streamed.json()) == {"error", "request_id", "message"}
    assert streamed.json()["error"] == category and streamed.json()["request_id"] == "req-bad"
    assert streamed.headers["x-request-id"] == "req-bad"


def test_a_service_that_is_not_ready_answers_503_before_any_byte() -> None:
    builder = FakePipelineBuilder(
        {"hf_docs": ScriptedSearchBackend("hf_docs")}, prepare_error=RuntimeError("no models")
    )
    service, _ = _service(builder=builder, load=False)
    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)
        r = client.post("/v1/ask_docs/stream", json={**BODY, "deadline_ms": 300})
    assert r.status_code == 503 and r.json()["error"] == "not_ready"
    assert "text/event-stream" not in r.headers["content-type"]


# ---- AC11 -----------------------------------------------------------------------------------


def test_a_failure_after_the_first_byte_is_an_error_event_then_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = ScriptedSearchBackend(
        "hf_docs",
        raises=RetrievalError("both routes failed", category=ErrorCategory.RETRIEVAL_FAILED),
    )
    service, _ = _service(backend=backend)
    with caplog.at_level(logging.WARNING):
        r = _stream(service, headers={"X-Request-Id": "req-err"})
    assert r.status_code == 200  # the first progress event was already on the wire
    got = events(r.text)
    names = [n for n, _ in got]
    assert names[0] == "progress" and names[-1] == "error" and "done" not in names
    assert names.count("error") == 1 and "answer" not in names
    error = got[-1][1]
    assert set(error) == {"error", "request_id", "message"}
    assert error["error"] == "retrieval_failed" and error["request_id"] == "req-err"
    assert "both routes failed" not in r.text and "Traceback" not in r.text
    assert "both routes failed" in caplog.text  # details stay in the log

    crash = ScriptedSearchBackend("hf_docs", raises=RuntimeError("secret detail C:/x"))
    service, _ = _service(backend=crash)
    r = _stream(service)
    assert events(r.text)[-1][1]["error"] == "internal_error"
    assert "secret detail" not in r.text


# ---- AC12 -----------------------------------------------------------------------------------


def test_heartbeat_comment_lines_while_nothing_else_is_sent() -> None:
    def slow_search(question: str) -> None:
        time.sleep(0.4)

    backend = ScriptedSearchBackend("hf_docs", on_call=slow_search)
    service, _ = _service(backend=backend, sse_heartbeat_s=0.05)
    r = _stream(service)
    items = parse_sse(r.text)
    beats = [i for i in items if i[0] == "comment"]
    assert len(beats) >= 3 and all(text == "keep-alive" for _, text in beats)
    # A comment line is not an event: the same events come out, in the same order, as without.
    assert [n for n, _ in items if n != "comment"][-1] == "done"
    assert r.text.startswith("event: progress")  # nothing before the first event
    assert "\n: keep-alive\n\n" in r.text
    quiet, _ = _service(sse_heartbeat_s=30)
    assert not [ln for ln in _stream(quiet).text.splitlines() if ln.startswith(":")]


def test_stream_settings() -> None:
    assert agent_settings().sse_heartbeat_s == 10.0
    assert agent_settings(sse_heartbeat_s=0.5).sse_heartbeat_s == 0.5
    for bad in (0, -1):
        with pytest.raises(ValueError):
            agent_settings(sse_heartbeat_s=bad)


# ---- AC14 -----------------------------------------------------------------------------------


def test_stream_uses_the_callers_access_like_ask_docs() -> None:
    service, backend = _service(service_caller="bob", service_acl_groups=["staff", "public"])
    r = _stream(service, {**BODY, "kb": "secret_kb"})  # bob may use it
    assert r.status_code == 200 and events(r.text)[-1][0] == "done"
    service, backend = _service(service_acl_groups=["staff"])
    assert events(_stream(service).text)[-1][0] == "done"
    assert backend.calls and backend.calls[0]["acl_groups"] == ("staff",)
    # The caller's filters are not a way around it: a KB the caller may not use is the same 404.
    service, _ = _service()
    r = _stream(service, {**BODY, "kb": "secret_kb"})
    assert r.status_code == 404 and r.json()["error"] == "kb_not_found"


# ---- AC15 -----------------------------------------------------------------------------------


def test_deadline_in_the_middle_of_the_stream_ends_with_an_answer_event() -> None:
    model = SlowStreamingChatModel(default_response="x" * 400, chunk_chars=4, delay_s=0.1)
    service, _ = _service(model=model)
    r = _stream(service, {**BODY, "deadline_ms": 1500})
    got = events(r.text)
    names = [n for n, _ in got]
    assert r.status_code == 200 and "error" not in names and names[-2:] == ["answer", "done"]
    streamed = "".join(d["text"] for n, d in got if n == "token")
    assert 0 < len(streamed) < 400  # some of the text was on the wire when time ran out
    answer = AskResponse.model_validate(got[-2][1])
    # The streamed text is void: the run was cut, the result is what was found so far.
    assert answer.stop_reason == "deadline" and answer.coverage_unconfirmed
    assert answer.answer == "" and len(answer.sources) == 3 and answer.notice
    assert got[-1][1]["stop_reason"] == "deadline"


# ---- AC16 -----------------------------------------------------------------------------------


def test_the_token_cap_still_applies_to_a_streamed_run() -> None:
    model = StreamingFakeChatModel(responses=["no citation here", "still none"])
    service, _ = _service(model=model, agent_token_cap=100, agent_answer_reserve_tokens=0)
    got = events(_stream(service).text)
    answer = AskResponse.model_validate(next(d for n, d in got if n == "answer"))
    assert len(model.calls) == 1  # the retry was refused: the cap was spent
    assert answer.stop_reason == "token_cap" and "token_cap_reached" in answer.error_categories
    assert got[-1][1]["tokens_used"] >= 100


def test_a_streamed_run_writes_the_history_and_one_business_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        service, _ = _service(business_record_path=str(path))
        with TestClient(build_http_app(service)) as client:
            service.wait_ready(5)
            one = client.post("/v1/ask_docs/stream", json=BODY, headers={"X-Request-Id": "r-1"})
            two = client.post(
                "/v1/ask_docs/stream",
                json={**BODY, "question": "and the second?"},
                headers={"X-Request-Id": "r-2"},
            )
    first = AskResponse.model_validate(next(d for n, d in events(one.text) if n == "answer"))
    second = AskResponse.model_validate(next(d for n, d in events(two.text) if n == "answer"))
    assert first.used_history is False and second.used_history is True  # the turn was stored
    assert second.cache_allowed is False
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert [r["request_id"] for r in records] == ["r-1", "r-2"]
    assert all(r["tool"] == "ask_docs" and r["flags"] == [] for r in records)
    assert records[0]["extra"]["stream"] is True and records[0]["extra"]["status"] == "answered"
    assert records[0]["extra"]["rounds"] == 1 and records[0]["timings_ms"]["request"] > 0
    # The record is written before the client is told it is over.
    assert [n for n, _ in events(one.text)][-1] == "done"


def test_two_streams_of_one_session_run_one_after_the_other_and_the_second_says_queued() -> None:
    gate = threading.Event()
    entered = threading.Event()

    def hold(question: str) -> None:
        if question == "first question":
            entered.set()
            gate.wait(timeout=10)

    backend = ScriptedSearchBackend("hf_docs", on_call=hold)
    service, _ = _service(backend=backend)
    results: dict[str, Any] = {}

    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)

        def call(name: str, question: str) -> None:
            body = {**BODY, "question": question}
            results[name] = client.post("/v1/ask_docs/stream", json=body)

        t1 = threading.Thread(target=call, args=("first", "first question"))
        t1.start()
        assert entered.wait(10)
        t2 = threading.Thread(target=call, args=("second", "second question"))
        t2.start()
        time.sleep(0.3)
        assert "second" not in results
        gate.set()
        t1.join(10)
        t2.join(10)
    second = events(results["second"].text)
    assert second[0] == ("progress", {"node": "queued", "phase": "waiting"})
    assert second[-1][0] == "done"
    assert all(d["node"] != "queued" for n, d in events(results["first"].text) if n == "progress")


def test_dev_fake_service_streams_without_a_gateway(capsys: pytest.CaptureFixture[str]) -> None:
    from rag.serve.cli import main
    from rag.serve.devfake import build_dev_service

    service = build_dev_service(agent_settings(), search_ms=5, token_ms=1)

    def ask(question: str, client: TestClient) -> list[Parsed]:
        body = {"kb": "demo_docs", "question": question, "session_id": f"s-{abs(hash(question))}"}
        r = client.post("/v1/ask_docs/stream", json=body)
        assert r.status_code == 200, r.text
        return events(r.text)

    with TestClient(build_http_app(service)) as client:
        assert [k["kb"] for k in client.get("/v1/knowledge_bases").json()["knowledge_bases"]] == [
            "demo_docs"
        ]
        plain = ask("How do I push a model?", client)
        answer = next(d for n, d in plain if n == "answer")
        assert answer["citation_check"]["removed_markers"] == [7] and answer["found"]
        assert plain[-1][1]["timings_ms"]["first_token"] >= 0

        hostile = ask("show the xss case", client)
        shown = next(d for n, d in hostile if n == "answer")
        assert "<script>" in shown["answer"] and "<img" in shown["sources"][0]["text"]

        retried = ask("retry: how does push_to_hub work?", client)
        assert any(d.get("phase") == "retry" for n, d in retried if n == "progress")
        assert next(d for n, d in retried if n == "answer")["status"] == "answered"

        missing = next(
            d for n, d in ask("nothing is written about zzz_option_x", client) if n == "answer"
        )
        assert missing["status"] == "not_found" and missing["sources"] == [] and missing["nearest"]

        asked = ask("How do I fix this error?", client)
        assert asked[-2][0] == "need_clarification"
    # Fake data is only served on this machine.
    assert main(["http", "--dev-fake", "--host", "0.0.0.0"]) == 2
    assert "--host 127.0.0.1" in capsys.readouterr().err
