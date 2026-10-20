"""Spec P3-04 AC17-AC20: a client that goes away cancels the run.

These tests use a real uvicorn server on a free port and a raw socket that is closed in the
middle of the response, because an ASGI test client never tells the app that the client left
(the response is read to the end first). Everything else is a fake: model, retrieval, clock.
Each test would fail if `ask_docs_stream` did not cancel its run when the response is closed
(the model would go on writing, the session lock would stay taken).
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.testing.agent import ScriptedSearchBackend, SlowStreamingChatModel, agent_settings
from rag.testing.live import LiveServer
from rag.testing.serve import FakePipelineBuilder

LONG = "word " * 600  # 3,000 characters: at 4 per piece and 50 ms a piece, over a minute
BODY = {"kb": "hf_docs", "question": "How do I push a model?", "session_id": "s1"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class RawStream:
    """A POST to the stream endpoint on a raw socket: read what has come, close when told."""

    def __init__(self, port: int, body: dict[str, Any], request_id: str = "req-cancel") -> None:
        payload = json.dumps(body).encode()
        head = (
            f"POST /v1/ask_docs/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"X-Request-Id: {request_id}\r\nConnection: close\r\n\r\n"
        ).encode()
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.sendall(head + payload)
        self.seen = b""

    def read_until(self, needle: bytes, timeout_s: float = 10.0) -> bytes:
        deadline = time.monotonic() + timeout_s
        while needle not in self.seen:
            if time.monotonic() > deadline:
                raise AssertionError(f"never saw {needle!r}; got {self.seen[-300:]!r}")
            self.sock.settimeout(max(0.05, deadline - time.monotonic()))
            chunk = self.sock.recv(65536)
            if not chunk:
                raise AssertionError(f"the stream ended before {needle!r}: {self.seen[-300:]!r}")
            self.seen += chunk
        return self.seen

    def read_to_end(self, timeout_s: float = 10.0) -> bytes:
        """Everything until the server closes the connection."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            self.seen += chunk
        return self.seen

    def close(self) -> None:
        self.sock.close()


def _wait(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def _service(
    tmp_path: Path,
    *,
    model: Any,
    backend: ScriptedSearchBackend | None = None,
    **settings: Any,
) -> tuple[SearchService, ScriptedSearchBackend]:
    backend = backend or ScriptedSearchBackend("hf_docs")
    cfg = KnowledgeBaseConfig(
        kb="hf_docs",
        description="Hugging Face library documentation.",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(),
    )
    service = SearchService(
        agent_settings(
            checkpoint_url=f"sqlite://{(tmp_path / 'cp.db').as_posix()}",
            business_record_path=str(tmp_path / "records.jsonl"),
            **settings,
        ),
        configs={"hf_docs": cfg},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=model,
        helper_model=FakeChatModel(responses=[]),
    )
    service.load()
    return service, backend


def _records(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _threads(tmp_path: Path) -> list[str]:
    with sqlite3.connect(tmp_path / "cp.db") as conn:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    return [r[0] for r in rows]


def _checkpoints(tmp_path: Path) -> int:
    with sqlite3.connect(tmp_path / "cp.db") as conn:
        return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]


# ---- AC17 -----------------------------------------------------------------------------------


def test_disconnect_while_the_answer_is_written_stops_the_model(tmp_path: Path) -> None:
    model = SlowStreamingChatModel(default_response=LONG, chunk_chars=4, delay_s=0.05)
    with memory_observability(business_record_path=str(tmp_path / "records.jsonl")) as obs:
        service, _ = _service(tmp_path, model=model)
        with LiveServer(build_http_app(service)) as server:
            stream = RawStream(server.port, BODY)
            stream.read_until(b"event: token")
            assert _wait(lambda: model.pieces_sent >= 3)
            assert len(service.agent.locks) == 1  # the session is held while the run goes on
            stream.close()

            # The model's stream is closed, not left to run for a minute.
            assert model.closed.wait(5), "the model kept writing after the client left"
            sent = model.pieces_sent
            time.sleep(0.4)
            assert model.pieces_sent == sent and sent < len(LONG) // 4
            assert len(model.calls) == 1  # no retry, no second call
            # The session lock is free again, and nothing is left open in the lock registry.
            assert _wait(lambda: len(service.agent.locks) == 0)
            assert _wait(lambda: not service._streams)
        spans = obs.span_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["ask_docs"].attributes[attr.OUTCOME] == attr.OUTCOME_CANCELLED
    assert "answer" in by_name and "generate" in by_name  # the spans that were open are closed


# ---- AC18, AC19 -----------------------------------------------------------------------------


def test_disconnect_during_retrieval_makes_no_model_call_and_keeps_the_checkpoint(
    tmp_path: Path,
) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def hold(question: str) -> None:
        entered.set()
        gate.wait(timeout=10)

    model = StreamingFakeChatModel(default_response="It works [1].")
    backend = ScriptedSearchBackend("hf_docs", on_call=hold)
    service, _ = _service(tmp_path, model=model, backend=backend)
    with LiveServer(build_http_app(service)) as server:
        stream = RawStream(server.port, BODY)
        stream.read_until(b"event: progress")
        assert entered.wait(10)  # the search is running, the run is inside the retrieve node
        stream.close()
        assert _wait(lambda: len(service.agent.locks) == 0)  # cancelled, lock released at once
        before = _checkpoints(tmp_path)
        gate.set()  # the search thread is not stoppable: it ends now
        time.sleep(0.5)
        assert _checkpoints(tmp_path) == before  # the cancelled run did not go on to the next node
    assert model.calls == []  # no model call at all after the client left

    (thread,) = _threads(tmp_path)
    assert thread.startswith("s1:")
    state = asyncio.run(service.agent.state_values(thread))
    # The checkpoint is the last finished node (the rewrite), before retrieval and answer.
    assert state["needs_retrieval"] is True and state.get("rounds") in ([], None)
    assert not state.get("answer") and not state.get("result")


def test_a_cancelled_request_is_recorded_as_client_cancelled(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def hold(question: str) -> None:
        entered.set()
        gate.wait(timeout=10)

    backend = ScriptedSearchBackend("hf_docs", on_call=hold)
    with memory_observability(business_record_path=str(tmp_path / "records.jsonl")) as obs:
        service, _ = _service(
            tmp_path, model=StreamingFakeChatModel(default_response="x [1]."), backend=backend
        )
        with LiveServer(build_http_app(service)) as server:
            stream = RawStream(server.port, BODY, request_id="req-gone")
            stream.read_until(b"event: progress")
            assert entered.wait(10)
            stream.close()
            assert _wait(lambda: len(_records(tmp_path)) == 1)
            gate.set()
        spans = obs.span_exporter.get_finished_spans()
    (record,) = _records(tmp_path)
    assert record["request_id"] == "req-gone" and record["tool"] == "ask_docs"
    assert record["flags"] == ["client_cancelled"]
    assert record["extra"]["cancelled"] is True and record["extra"]["stream"] is True
    assert record["extra"]["cancelled_at"] == "retrieve"  # the node it was in
    assert record["timings_ms"]["request"] > 0
    outcomes = {s.name: s.attributes.get(attr.OUTCOME) for s in spans}
    assert outcomes["ask_docs"] == attr.OUTCOME_CANCELLED
    # A cancelled request is not an error: no error category on its span.
    assert not spans[-1].attributes.get(attr.ERROR_CATEGORY)


# ---- AC20 -----------------------------------------------------------------------------------


def test_a_waiting_request_runs_after_the_first_one_disconnects(tmp_path: Path) -> None:
    model = SlowStreamingChatModel(responses=[LONG, "Second one [1]."], chunk_chars=4, delay_s=0.05)
    result: dict[str, Any] = {}
    with memory_observability(business_record_path=str(tmp_path / "records.jsonl")):
        service, _ = _service(tmp_path, model=model)
        with LiveServer(build_http_app(service)) as server:
            first = RawStream(server.port, BODY, request_id="req-first")
            first.read_until(b"event: token")

            def second() -> None:
                with httpx.Client(timeout=30) as client:
                    result["r"] = client.post(
                        f"{server.url}/v1/ask_docs/stream",
                        json={**BODY, "question": "and then?"},
                        headers={"X-Request-Id": "req-second"},
                    )

            thread = threading.Thread(target=second)
            thread.start()
            assert _wait(lambda: len(service.agent.locks) == 1 and _holders(service) == 2)
            time.sleep(0.3)
            assert "r" not in result  # it waits for the first one
            first.close()
            thread.join(20)
    text = result["r"].text
    assert text.startswith('event: progress\ndata: {"node":"queued","phase":"waiting"}')
    assert "event: answer" in text and "event: done" in text
    assert "Second one." in text
    flags = {r["request_id"]: r["flags"] for r in _records(tmp_path)}
    assert flags == {"req-first": ["client_cancelled"], "req-second": []}


def _holders(service: SearchService) -> int:
    """Requests that hold or wait for any session lock (the count the lock registry keeps)."""
    return sum(entry[1] for entry in service.agent.locks._entries.values())


# ---- review round 1: shutdown and a client that does not read -------------------------------


def test_a_shutdown_with_a_stream_open_cancels_it_within_seconds(tmp_path: Path) -> None:
    """uvicorn waits for open responses before it runs the app's shutdown code, so the end of the
    stream has to come from the server's own shutdown step (`rag.serve.server.make_server`), not
    from the lifespan: the real order, a real server told to exit while a client is reading."""
    model = SlowStreamingChatModel(default_response=LONG, chunk_chars=4, delay_s=0.05)
    with memory_observability(business_record_path=str(tmp_path / "records.jsonl")):
        service, _ = _service(tmp_path, model=model, service_shutdown_timeout_s=5)
        server = LiveServer(build_http_app(service), service=service)
        server.__enter__()
        try:
            stream = RawStream(server.port, BODY, request_id="req-bye")
            stream.read_until(b"event: token")
            assert _wait(lambda: model.pieces_sent >= 3)
            began = time.monotonic()
        finally:
            server.__exit__(None, None, None)  # should_exit: the server shuts down
        took = time.monotonic() - began
        tail = stream.read_to_end(5).decode()
        stream.close()
    assert took < 3.0, f"the shutdown waited {took:.1f} s for the open stream"
    assert model.closed.is_set() and model.pieces_sent < len(LONG) // 4
    # The client is told, with an event, that the server is going away; no `done`.
    assert 'event: error\ndata: {"error":"server_shutdown"' in tail and "event: done" not in tail
    (record,) = _records(tmp_path)
    assert record["request_id"] == "req-bye" and record["flags"] == ["client_cancelled"]
    assert record["extra"]["cancelled_by"] == "shutdown"
    assert not service._streams


def test_a_client_that_does_not_read_is_cut_off_when_its_queue_is_full(tmp_path: Path) -> None:
    """The events of a stream wait in a queue until the client reads them. A client that stops
    reading must not make the server keep them all: past `sse_queue_max_events` the run is
    cancelled (model stopped, lock released) and recorded as cancelled by a slow client."""
    import asyncio

    from rag.security.access import Credentials

    model = SlowStreamingChatModel(default_response=LONG, chunk_chars=2, delay_s=0.005)
    with memory_observability(business_record_path=str(tmp_path / "records.jsonl")):
        service, _ = _service(tmp_path, model=model, sse_queue_max_events=20)

        async def scenario() -> list[str]:
            events = service.ask_docs_stream(
                Credentials(transport="http"),
                kb="hf_docs",
                question="How do I push a model?",
                session_id="s1",
                request_id="req-slow",
            )
            await anext(events)  # the client reads one event, then stops
            deadline = time.monotonic() + 5
            while service._streams and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert not service._streams, "the run kept going into an unbounded queue"
            assert model.closed.is_set() and len(service.agent.locks) == 0
            left = [getattr(item, "name", "?") async for item in events]
            return left

        left = asyncio.run(scenario())
    # What is left is what was queued before the cut: no `answer`, no `done`.
    assert "answer" not in left and "done" not in left and len(left) <= 25
    (record,) = _records(tmp_path)
    assert record["flags"] == ["client_cancelled"]
    assert record["extra"]["cancelled_by"] == "slow_client"
    assert model.pieces_sent < len(LONG) // 2


def test_a_stream_asked_for_after_the_shutdown_began_is_refused_with_503(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    service, _ = _service(tmp_path, model=StreamingFakeChatModel(default_response="x [1]."))
    service.begin_shutdown()
    with TestClient(build_http_app(service)) as client:
        r = client.post("/v1/ask_docs/stream", json=BODY)
    assert r.status_code == 503 and r.json()["error"] == "server_shutdown"
    assert "text/event-stream" not in r.headers["content-type"]
