"""Chat history per session and the lock for one session's requests (design s13-3, s16-7).

The history is not LangGraph state: every request is its own checkpoint thread
(`session_id:request_id`, so one request never sees another's retrieved chunks) and the turns of
a session (what was asked, what was answered, nothing else) are stored here, per session id, and
handed to the graph as input. Both stores are created from a connection string by
`rag.factories.make_session_store` and `make_session_locks`.

`SessionLocks.hold(session_id)`: a coding assistant often sends several tool calls at once and
LangGraph does not stop two runs on one thread (LangGraph page, measured), and a session's turns
must be read and written in order, so requests of one session run one after another; other
sessions are not held up. Inside one process an asyncio lock per session id queues the requests
first-come first-served; with a lock provider (`file://<dir>`) the process also holds a lock
across processes on this machine. The provider's locks are thread locks, so one helper thread
takes and releases it (a file lock must be released by the thread that took it).
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from rag.factories import LockProvider, LockTimeoutError

# Longest question or answer text kept per stored turn: history is context for rewriting, not an
# archive (the business record and the trace keep the details).
TURN_TEXT_CHARS = 8000

_PROVIDER_POLL_S = 0.5


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class HistoryTurn(BaseModel):
    """One finished request of a session: the words only, never chunks or ids of chunks."""

    request_id: str
    question: str
    answer: str
    at: str = Field(default_factory=_now)


class SessionStore(Protocol):
    async def load(self, session_id: str) -> list[HistoryTurn]:
        """The session's last turns, oldest first (empty for an unknown session)."""
        ...

    async def append(self, session_id: str, turn: HistoryTurn) -> None:
        """Add a turn and drop the oldest ones beyond the number kept."""
        ...


def _clip(turn: HistoryTurn) -> HistoryTurn:
    return turn.model_copy(
        update={
            "question": turn.question[:TURN_TEXT_CHARS],
            "answer": turn.answer[:TURN_TEXT_CHARS],
        }
    )


class MemorySessionStore:
    """Turns in process memory (tests, single run scripts)."""

    def __init__(self, keep_turns: int) -> None:
        self.keep_turns = keep_turns
        self._turns: dict[str, list[HistoryTurn]] = {}

    async def load(self, session_id: str) -> list[HistoryTurn]:
        if self.keep_turns <= 0:
            return []
        return list(self._turns.get(session_id, [])[-self.keep_turns :])

    async def append(self, session_id: str, turn: HistoryTurn) -> None:
        if self.keep_turns <= 0:
            return
        turns = self._turns.setdefault(session_id, [])
        turns.append(_clip(turn))
        del turns[: -self.keep_turns]


class SqliteSessionStore:
    """Turns in one SQLite file (single instance, design s16-7); Postgres comes with several
    instances. A connection per operation, run in a worker thread so the event loop is not held
    up; WAL mode lets other sessions' requests write at the same time."""

    def __init__(self, path: Path, keep_turns: int) -> None:
        self.path = path
        self.keep_turns = keep_turns

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "request_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, "
            "at TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS turns_session ON turns (session_id, id)")
        return conn

    def _load(self, session_id: str) -> list[HistoryTurn]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT request_id, question, answer, at FROM turns WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, self.keep_turns),
            ).fetchall()
        return [
            HistoryTurn(request_id=r[0], question=r[1], answer=r[2], at=r[3])
            for r in reversed(rows)
        ]

    def _append(self, session_id: str, turn: HistoryTurn) -> None:
        turn = _clip(turn)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO turns (session_id, request_id, question, answer, at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, turn.request_id, turn.question, turn.answer, turn.at),
            )
            conn.execute(
                "DELETE FROM turns WHERE session_id = ? AND id NOT IN "
                "(SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self.keep_turns),
            )

    async def load(self, session_id: str) -> list[HistoryTurn]:
        if self.keep_turns <= 0:
            return []
        return await asyncio.to_thread(self._load, session_id)

    async def append(self, session_id: str, turn: HistoryTurn) -> None:
        if self.keep_turns <= 0:
            return
        await asyncio.to_thread(self._append, session_id, turn)


@asynccontextmanager
async def _hold_provider_lock(provider: LockProvider, name: str) -> AsyncIterator[None]:
    """Hold `provider.lock(name)` for the block, taken and released by one helper thread. A
    wait that is cancelled (the request's deadline) stops the helper's polling."""
    loop = asyncio.get_running_loop()
    acquired: asyncio.Future[None] = loop.create_future()
    release = threading.Event()
    abandoned = threading.Event()

    def settle(setter: Any, value: Any) -> None:
        def apply() -> None:
            if not acquired.done():
                setter(value)

        loop.call_soon_threadsafe(apply)

    def worker() -> None:
        while not abandoned.is_set():
            try:
                with provider.lock(name, timeout=_PROVIDER_POLL_S):
                    settle(acquired.set_result, None)
                    release.wait()
                    return
            except LockTimeoutError:
                continue
            except BaseException as exc:  # a lock that cannot be taken at all
                settle(acquired.set_exception, exc)
                return

    thread = threading.Thread(target=worker, name=f"lock-{name[:40]}", daemon=True)
    thread.start()
    try:
        await acquired
    except BaseException:
        abandoned.set()
        release.set()
        raise
    try:
        yield
    finally:
        release.set()
        await loop.run_in_executor(None, thread.join, 5.0)


class SessionLocks:
    """One lock per session id; see the module doc. A session's lock exists only while someone
    holds or waits for it, so the registry does not grow with the number of sessions."""

    def __init__(self, provider: LockProvider | None = None) -> None:
        self._provider = provider
        # session id -> [asyncio lock, holders and waiters]
        self._entries: dict[str, list[Any]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        entry = self._entries.get(session_id)
        if entry is None:
            entry = self._entries[session_id] = [asyncio.Lock(), 0]
        entry[1] += 1
        try:
            async with entry[0]:
                if self._provider is None:
                    yield
                else:
                    async with _hold_provider_lock(self._provider, f"session:{session_id}"):
                        yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                with contextlib.suppress(KeyError):
                    del self._entries[session_id]
