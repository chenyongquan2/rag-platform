"""Spec P3-01 AC7 (store), AC8 (locks), AC14 (store factory): chat history per session and the
lock that makes requests of one session run one after another."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag.agent import HistoryTurn
from rag.factories import (
    LockTimeoutError,
    make_lock_provider,
    make_session_locks,
    make_session_store,
)


def _turn(n: int) -> HistoryTurn:
    return HistoryTurn(request_id=f"r{n}", question=f"q{n}", answer=f"a{n}")


def _urls(tmp_path: Path) -> list[str]:
    return ["memory://", f"sqlite://{(tmp_path / 'sessions.db').as_posix()}"]


def test_session_store_keeps_last_turns(tmp_path: Path) -> None:
    async def scenario(url: str) -> None:
        store = make_session_store(url, keep_turns=2)
        assert await store.load("s1") == []
        for n in (1, 2, 3):
            await store.append("s1", _turn(n))
        await store.append("s2", _turn(9))
        assert [t.question for t in await store.load("s1")] == ["q2", "q3"]  # oldest first
        assert [t.answer for t in await store.load("s2")] == ["a9"]
        assert await store.load("unknown") == []

    for url in _urls(tmp_path):
        asyncio.run(scenario(url))

    # The SQLite store is a file: a second store object on the same URL sees the turns.
    url = _urls(tmp_path)[1]
    reopened = make_session_store(url, keep_turns=2)
    assert [t.question for t in asyncio.run(reopened.load("s1"))] == ["q2", "q3"]

    async def disabled() -> None:
        store = make_session_store("memory://", keep_turns=0)
        await store.append("s1", _turn(1))
        assert await store.load("s1") == []

    asyncio.run(disabled())


def test_session_store_factory(tmp_path: Path) -> None:
    store = make_session_store(_urls(tmp_path)[1], keep_turns=3)
    assert type(store).__name__ == "SqliteSessionStore"
    assert type(make_session_store("memory://", keep_turns=3)).__name__ == "MemorySessionStore"
    with pytest.raises(NotImplementedError):
        make_session_store("postgresql://db/rag", keep_turns=3)
    with pytest.raises(ValueError):
        make_session_store("no-scheme", keep_turns=3)
    with pytest.raises(NotImplementedError):
        make_session_locks("redis://127.0.0.1:6379/0")
    assert make_session_locks("memory://") is not None


def test_same_session_runs_one_after_another() -> None:
    locks = make_session_locks("memory://")
    log: list[str] = []

    async def job(name: str) -> None:
        async with locks.hold("s1"):
            log.append(f"start {name}")
            await asyncio.sleep(0.05)
            log.append(f"end {name}")

    async def main() -> None:
        first = asyncio.create_task(job("a"))
        await asyncio.sleep(0.01)
        await asyncio.gather(first, job("b"), job("c"))

    asyncio.run(main())
    assert log == ["start a", "end a", "start b", "end b", "start c", "end c"]


def test_different_sessions_run_in_parallel() -> None:
    locks = make_session_locks("memory://")
    log: list[str] = []

    async def job(session: str) -> None:
        async with locks.hold(session):
            log.append(f"start {session}")
            await asyncio.sleep(0.05)
            log.append(f"end {session}")

    async def main() -> None:
        await asyncio.gather(job("s1"), job("s2"))

    asyncio.run(main())
    assert log[:2] == ["start s1", "start s2"]
    assert sorted(log[2:]) == ["end s1", "end s2"]
    # Nothing is kept for a finished session: the registry does not grow with session ids.
    assert len(locks) == 0


def test_file_lock_is_held_while_running(tmp_path: Path) -> None:
    url = f"file://{(tmp_path / 'locks').as_posix()}"
    locks = make_session_locks(url)
    other_process = make_lock_provider(url)  # what a second server process would use

    async def main() -> None:
        async with locks.hold("s1"):
            with pytest.raises(LockTimeoutError), other_process.lock("session:s1", timeout=0.2):
                pass
            # Another session is not blocked.
            with other_process.lock("session:s2", timeout=1.0):
                pass
        with other_process.lock("session:s1", timeout=2.0):  # released with the block
            pass

    asyncio.run(main())
