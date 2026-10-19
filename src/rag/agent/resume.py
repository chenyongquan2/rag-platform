"""Resume tokens: the keys to runs that stopped to ask the caller (design s13-3; spec P3-02).

A run that stops at the clarify step is a checkpoint thread waiting for an answer. The caller
gets an opaque `resume_token`, not the thread id, and the server keeps, per token (as a hash):
who it was issued to (caller id), for which knowledge base and session, which thread it opens
and until when. `ask_docs(..., resume_token=..., answer=...)` takes the token with `claim`:

- it must exist, be unused and not expired;
- the caller, the knowledge base and the session id must be the ones it was issued for;
- it is used once: `claim` marks it used in the same step that checks it, so two calls with
  one token cannot both continue the run.

Anything else is `ResumeTokenInvalid` (category `invalid_resume_token`), one category and one
fixed message for every reason, so a caller cannot find out which tokens exist; the reason is
only in the exception text, which goes to the log. A call with someone else's token fails
without using it up: the owner can still answer. The token is never stored or logged in clear.

The store is made from `session_store_url` by `rag.factories.make_resume_store` (SQLite file
next to the session history, or process memory).
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.agent.sqlite_db import init_agent_db
from rag.factories import init_sqlite_once
from rag.types import ErrorCategory


class ResumeTokenInvalid(Exception):  # noqa: N818 - named after its category
    """The token cannot continue a run; `str(exc)` is the reason, for the log only."""

    error_category = ErrorCategory.INVALID_RESUME_TOKEN


@dataclass(frozen=True)
class ResumeRecord:
    caller_id: str
    kb: str
    session_id: str | None
    thread_id: str
    expires_at: float


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    """A URL-safe token with 192 random bits: it cannot be guessed."""
    return secrets.token_urlsafe(24)


def _problem(
    record: ResumeRecord | None,
    status: str,
    *,
    caller_id: str,
    kb: str,
    session_id: str | None,
    now: float,
) -> str | None:
    """Why the token cannot be used by this call, or None."""
    if record is None:
        return "unknown token"
    if (record.caller_id, record.kb, record.session_id) != (caller_id, kb, session_id):
        return "token was issued to another caller, knowledge base or session"
    if status != "pending":
        return "token already used"
    if record.expires_at <= now:
        return "token expired"
    return None


class ResumeStore(Protocol):
    def prepare(self) -> None:
        """Create the storage before the first request (idempotent)."""
        ...

    async def create(
        self, *, caller_id: str, kb: str, session_id: str | None, thread_id: str
    ) -> str:
        """Issue a token for a run waiting at `thread_id`."""
        ...

    async def purge_expired(self) -> list[str]:
        """Drop the tokens that expired and return their checkpoint threads, which nobody can
        resume any more (the caller deletes them)."""
        ...

    async def claim(
        self, token: str, *, caller_id: str, kb: str, session_id: str | None
    ) -> ResumeRecord:
        """Use the token (once). `ResumeTokenInvalid` when it cannot be used by this call."""
        ...


@dataclass
class _Row:
    record: ResumeRecord
    status: str = "pending"


class MemoryResumeStore:
    """Tokens in process memory (tests, single run scripts)."""

    def __init__(self, ttl_s: float, now: Callable[[], float] = time.time) -> None:
        self.ttl_s = ttl_s
        self._now = now
        self._rows: dict[str, _Row] = {}
        self._lock = threading.Lock()

    def prepare(self) -> None:
        pass

    async def create(
        self, *, caller_id: str, kb: str, session_id: str | None, thread_id: str
    ) -> str:
        token = new_token()
        now = self._now()
        record = ResumeRecord(caller_id, kb, session_id, thread_id, now + self.ttl_s)
        with self._lock:
            self._rows[hash_token(token)] = _Row(record)
        return token

    async def purge_expired(self) -> list[str]:
        now = self._now()
        with self._lock:
            stale = [k for k, row in self._rows.items() if row.record.expires_at <= now]
            return [self._rows.pop(k).record.thread_id for k in stale]

    async def claim(
        self, token: str, *, caller_id: str, kb: str, session_id: str | None
    ) -> ResumeRecord:
        with self._lock:
            row = self._rows.get(hash_token(token))
            problem = _problem(
                row.record if row else None,
                row.status if row else "",
                caller_id=caller_id,
                kb=kb,
                session_id=session_id,
                now=self._now(),
            )
            if problem is not None or row is None:
                raise ResumeTokenInvalid(problem or "unknown token")
            row.status = "used"
            return row.record


class SqliteResumeStore:
    """Tokens in the session SQLite file. A connection per operation in a worker thread;
    `claim` checks and marks the token in one write transaction."""

    def __init__(self, path: Path, ttl_s: float, now: Callable[[], float] = time.time) -> None:
        self.path = path
        self.ttl_s = ttl_s
        self._now = now

    def prepare(self) -> None:
        init_sqlite_once(self.path, init_agent_db)

    def _connect(self) -> sqlite3.Connection:
        self.prepare()
        return sqlite3.connect(self.path, timeout=5.0)

    def _create(
        self, token: str, caller_id: str, kb: str, session_id: str | None, thread: str
    ) -> None:
        now = self._now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO resumes (token_hash, caller_id, kb, session_id, thread_id, status, "
                "expires_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (hash_token(token), caller_id, kb, session_id, thread, now + self.ttl_s),
            )

    def _purge(self) -> list[str]:
        now = self._now()
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT thread_id FROM resumes WHERE expires_at <= ?", (now,)
            ).fetchall()
            conn.execute("DELETE FROM resumes WHERE expires_at <= ?", (now,))
        return [r[0] for r in rows]

    def _claim(self, token: str, caller_id: str, kb: str, session_id: str | None) -> ResumeRecord:
        with closing(self._connect()) as conn:
            conn.isolation_level = None  # explicit transaction: the read and the update are one
            conn.execute("BEGIN IMMEDIATE")
            try:
                found = conn.execute(
                    "SELECT caller_id, kb, session_id, thread_id, expires_at, status FROM resumes "
                    "WHERE token_hash = ?",
                    (hash_token(token),),
                ).fetchone()
                record = ResumeRecord(*found[:5]) if found else None
                problem = _problem(
                    record,
                    found[5] if found else "",
                    caller_id=caller_id,
                    kb=kb,
                    session_id=session_id,
                    now=self._now(),
                )
                if problem is not None or record is None:
                    raise ResumeTokenInvalid(problem or "unknown token")
                conn.execute(
                    "UPDATE resumes SET status = 'used' WHERE token_hash = ?", (hash_token(token),)
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return record

    async def create(
        self, *, caller_id: str, kb: str, session_id: str | None, thread_id: str
    ) -> str:
        token = new_token()
        await asyncio.to_thread(self._create, token, caller_id, kb, session_id, thread_id)
        return token

    async def purge_expired(self) -> list[str]:
        return await asyncio.to_thread(self._purge)

    async def claim(
        self, token: str, *, caller_id: str, kb: str, session_id: str | None
    ) -> ResumeRecord:
        return await asyncio.to_thread(self._claim, token, caller_id, kb, session_id)
