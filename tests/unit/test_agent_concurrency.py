"""Spec P3-01 AC8 (through `ask_docs`), AC18, AC19: concurrent requests, cold databases, the
session key, and what a request leaves behind (lock, checkpoint thread) however it ends."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime, ask_docs
from rag.observe import shutdown_observability
from rag.testing import StreamingFakeChatModel
from rag.testing.agent import (
    HangingChatModel,
    ScriptedSearchBackend,
    agent_settings,
    kb_config,
    pages_result,
)
from rag.types import AskResult

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class Gauge:
    """Counts how many searches run at the same time (they run in worker threads)."""

    def __init__(self, hold_s: float = 0.05) -> None:
        self.hold_s = hold_s
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, question: str) -> None:
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.hold_s)
        with self._lock:
            self.now -= 1


def _ask(runtime: AgentRuntime, backend: Any, question: str, **kwargs: Any) -> Any:
    kwargs.setdefault("model", StreamingFakeChatModel(default_response="Answer [1]."))
    cfg = kwargs.pop("cfg", None) or kb_config(KB)
    return ask_docs(
        question,
        cfg=cfg,
        retriever=backend,
        settings=runtime.settings,
        runtime=runtime,
        **kwargs,
    )


def test_cold_sqlite_databases_survive_concurrent_requests(tmp_path: Path) -> None:
    # Fresh data directories: the checkpoint database and the session database do not exist
    # yet when twelve requests (six sessions, two each) arrive together.
    async def scenario(data: Path) -> list[AskResult]:
        settings = agent_settings(
            checkpoint_url=f"sqlite://{(data / 'checkpoints' / 'cp.db').as_posix()}",
            session_store_url=f"sqlite://{(data / 'sessions' / 's.db').as_posix()}",
        )
        runtime = AgentRuntime.from_settings(settings)
        backend = ScriptedSearchBackend(KB)
        results = await asyncio.gather(
            *[
                _ask(runtime, backend, f"q{i}", session_id=f"s{i % 6}", request_id=f"r{i}")
                for i in range(12)
            ]
        )
        for n in range(6):
            key = runtime.session_key("local", KB, f"s{n}")
            assert len(await runtime.store.load(key)) == 2
        return list(results)

    for attempt in range(4):  # the failure is a race: try several cold starts
        results = asyncio.run(scenario(tmp_path / f"data{attempt}"))
        assert all(r.found for r in results)


def test_same_session_runs_one_at_a_time_through_ask_docs() -> None:
    gauge = Gauge()
    backend = ScriptedSearchBackend(KB, on_call=gauge)
    runtime = AgentRuntime.from_settings(agent_settings())

    async def main() -> list[AskResult]:
        tasks = [
            asyncio.create_task(_ask(runtime, backend, f"q{i}", session_id="s1")) for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        turns = await runtime.store.load(runtime.session_key("local", KB, "s1"))
        # Turns are stored in the order the requests were made, each request saw the earlier ones.
        assert [t.question for t in turns] == [f"q{i}" for i in range(5)]
        assert [r.used_history for r in results] == [False, True, True, True, True]
        return list(results)

    asyncio.run(main())
    assert gauge.peak == 1
    assert len(runtime.locks) == 0  # nothing is kept for a finished session


def test_different_sessions_overlap_through_ask_docs() -> None:
    gauge = Gauge(hold_s=0.15)
    backend = ScriptedSearchBackend(KB, on_call=gauge)
    runtime = AgentRuntime.from_settings(agent_settings())

    async def main() -> None:
        await asyncio.gather(
            *[_ask(runtime, backend, f"q{i}", session_id=f"s{i}") for i in range(3)]
        )

    asyncio.run(main())
    assert gauge.peak >= 2
    assert len(runtime.locks) == 0


def test_lock_is_released_however_a_request_ends() -> None:
    runtime = AgentRuntime.from_settings(agent_settings())
    good = ScriptedSearchBackend(KB)

    async def next_request_runs(name: str) -> None:
        # The next request of the session must get the lock at once, not wait for its deadline.
        began = time.monotonic()
        result = await _ask(
            runtime, good, name, session_id="s1", deadline_ms=5000, request_id=f"after-{name}"
        )
        assert result.found and time.monotonic() - began < 2
        assert len(runtime.locks) == 0

    async def main() -> None:
        # 1. an exception in the run
        boom = ScriptedSearchBackend(KB, raises=RuntimeError("search broke"))
        with pytest.raises(RuntimeError):
            await _ask(runtime, boom, "q", session_id="s1")
        await next_request_runs("exception")

        # 2. the caller cancels the request while the model is working
        hanging = HangingChatModel(responses=["x [1]."], hang_s=30.0)
        task = asyncio.create_task(_ask(runtime, good, "q", session_id="s1", model=hanging))
        for _ in range(200):
            if hanging.started.is_set():
                break
            await asyncio.sleep(0.01)
        assert hanging.started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await next_request_runs("cancel")

        # 3. the hard deadline cuts the run
        hanging = HangingChatModel(responses=["x [1]."], hang_s=30.0)
        cut = await _ask(
            runtime, good, "q", session_id="s1", model=hanging, deadline_ms=400, request_id="cut"
        )
        assert cut.stop_reason == "deadline"
        await next_request_runs("deadline")

    asyncio.run(main())


def test_runs_with_the_same_request_id_do_not_share_a_checkpoint_thread() -> None:
    # Two callers send the same X-Request-Id (no session). One run is cut by the hard deadline
    # and rebuilt from its checkpoint: it must see its own chunks, not the other run's.
    backend = ScriptedSearchBackend(
        KB, by_question={"qa": pages_result(KB, [1, 2]), "qb": pages_result(KB, [9])}
    )
    runtime = AgentRuntime.from_settings(agent_settings())

    async def main() -> tuple[AskResult, AskResult]:
        slow = HangingChatModel(responses=["x [1]."], hang_s=30.0)
        cut = asyncio.create_task(
            _ask(runtime, backend, "qa", request_id="dup", model=slow, deadline_ms=600)
        )
        for _ in range(200):
            if slow.started.is_set():
                break
            await asyncio.sleep(0.01)
        other = await _ask(runtime, backend, "qb", request_id="dup")
        return await cut, other

    cut, other = asyncio.run(main())
    assert cut.request_id == other.request_id == "dup"
    assert cut.thread_id != other.thread_id
    assert cut.stop_reason == "deadline"
    assert [c.chunk_id for c in cut.sources] == ["hf_docs:p1.md#0", "hf_docs:p2.md#0"]
    assert [c.chunk_id for c in other.sources] == ["hf_docs:p9.md#0"]


def test_history_is_not_shared_between_callers_or_knowledge_bases() -> None:
    backend = ScriptedSearchBackend(KB)
    other_kb_backend = ScriptedSearchBackend("other")
    runtime = AgentRuntime.from_settings(agent_settings())

    async def main() -> None:
        first = await _ask(runtime, backend, "q1", session_id="s1", caller_id="alice")
        assert first.used_history is False
        # The same caller, session and KB: history.
        same = await _ask(runtime, backend, "q2", session_id="s1", caller_id="alice")
        assert same.used_history is True
        # Another caller with the same session id: no history, and no waiting for alice's lock.
        bob = await _ask(runtime, backend, "q3", session_id="s1", caller_id="bob")
        assert bob.used_history is False
        # The same caller and session id in another knowledge base: no history.
        other = await _ask(
            runtime,
            other_kb_backend,
            "q4",
            session_id="s1",
            caller_id="alice",
            cfg=kb_config("other"),
        )
        assert other.used_history is False
        alice = await runtime.store.load(runtime.session_key("alice", KB, "s1"))
        assert [t.question for t in alice] == ["q1", "q2"]
        keys = {runtime.session_key(c, k, "s1") for c in ("alice", "bob") for k in (KB, "other")}
        assert len(keys) == 4

    asyncio.run(main())

    # Their locks are separate too: bob's request is not held up by alice's running one.
    gauge = Gauge(hold_s=0.15)
    slow = ScriptedSearchBackend(KB, on_call=gauge)

    async def overlap() -> None:
        await asyncio.gather(
            _ask(runtime, slow, "a", session_id="s2", caller_id="alice"),
            _ask(runtime, slow, "b", session_id="s2", caller_id="bob"),
        )

    asyncio.run(overlap())
    assert gauge.peak == 2
