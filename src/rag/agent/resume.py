"""Resume tokens: the keys to runs that stopped to ask the caller (design s13-3; spec P3-02).

A run that stops at the clarify step is a checkpoint thread waiting for an answer. The caller
gets an opaque `resume_token`, not the thread id, and the server keeps, per token (as a hash):
who it was issued to (caller id), for which knowledge base and session, which thread it opens
and until when. `ask_docs(..., resume_token=..., answer=...)` takes the token with `claim`:

- it must exist, be unused, not in use by another call and not expired;
- the caller, the knowledge base and the session id must be the ones it was issued for;
- it can continue the run once at a time: `claim` marks it `resuming` in the same step that
  checks it, so two calls with one token cannot both continue the run.

A token is `pending` (can be claimed), `resuming` (a call is running the thread) or `used`. The
call that claimed it ends the claim: `finish` (the run ended, with an answer or cut off: used
for good) or `release` (the run raised or was cancelled: the caller may try again, the thread
goes on from where it stopped, and the tokens the attempt spent come back with the next claim).
Every claim has its own id, and only the call holding the current claim can end it: a call that
is slow past its lease and ends late does not touch the claim of the call that took over.
A call that dies without either (the process is killed) leaves `resuming`; after `lease_s`
seconds the claim lapses and the token can be claimed again.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from rag.agent.sqlite_db import init_agent_db
from rag.factories import init_sqlite_once
from rag.types import ErrorCategory

DEFAULT_LEASE_S = 120.0


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
    claim_id: str = ""
    """Who holds the claim (set by `claim`)."""
    tokens: int = 0
    """Model tokens spent by earlier, failed attempts (set by `claim`)."""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    """A URL-safe token with 192 random bits: it cannot be guessed."""
    return secrets.token_urlsafe(24)


def new_claim_id() -> str:
    return secrets.token_hex(8)


def in_flight(status: str, claimed_at: float | None, lease_s: float, now: float) -> bool:
    """A call is running the thread: it claimed the token and its lease has not lapsed."""
    return status == "resuming" and (claimed_at or 0.0) + lease_s > now


def _problem(
    record: ResumeRecord | None,
    status: str,
    claimed_at: float | None,
    *,
    caller_id: str,
    kb: str,
    session_id: str | None,
    now: float,
    lease_s: float,
) -> str | None:
    """Why the token cannot be used by this call, or None."""
    if record is None:
        return "unknown token"
    if (record.caller_id, record.kb, record.session_id) != (caller_id, kb, session_id):
        return "token was issued to another caller, knowledge base or session"
    if status == "used":
        return "token already used"
    if in_flight(status, claimed_at, lease_s, now):
        return "token in use by a running call"
    if record.expires_at <= now:
        return "token expired"
    return None


class ResumeStore(Protocol):
    lease_s: float

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
        resume any more (the caller deletes them). A token whose run is in flight stays."""
        ...

    async def claim(
        self, token: str, *, caller_id: str, kb: str, session_id: str | None
    ) -> ResumeRecord:
        """Take the token for one running call. `ResumeTokenInvalid` when this call cannot."""
        ...

    async def finish(self, token: str, claim_id: str) -> None:
        """The call that holds claim `claim_id` ended the run: the token is used up."""
        ...

    async def release(self, token: str, claim_id: str, tokens: int = 0) -> None:
        """The call that holds claim `claim_id` failed or was cancelled, after spending `tokens`
        in all on this request: the token can be claimed again."""
        ...


@dataclass
class _Row:
    record: ResumeRecord
    status: str = "pending"
    claimed_at: float | None = None
    claim_id: str = ""
    tokens: int = 0


class MemoryResumeStore:
    """Tokens in process memory (tests, single run scripts)."""

    def __init__(
        self,
        ttl_s: float,
        now: Callable[[], float] = time.time,
        lease_s: float = DEFAULT_LEASE_S,
    ) -> None:
        self.ttl_s = ttl_s
        self.lease_s = lease_s
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
            stale = [
                key
                for key, row in self._rows.items()
                if row.record.expires_at <= now
                and not in_flight(row.status, row.claimed_at, self.lease_s, now)
            ]
            return [self._rows.pop(key).record.thread_id for key in stale]

    async def claim(
        self, token: str, *, caller_id: str, kb: str, session_id: str | None
    ) -> ResumeRecord:
        with self._lock:
            row = self._rows.get(hash_token(token))
            now = self._now()
            problem = _problem(
                row.record if row else None,
                row.status if row else "",
                row.claimed_at if row else None,
                caller_id=caller_id,
                kb=kb,
                session_id=session_id,
                now=now,
                lease_s=self.lease_s,
            )
            if problem is not None or row is None:
                raise ResumeTokenInvalid(problem or "unknown token")
            row.status, row.claimed_at, row.claim_id = "resuming", now, new_claim_id()
            return replace(row.record, claim_id=row.claim_id, tokens=row.tokens)

    async def finish(self, token: str, claim_id: str) -> None:
        with self._lock:
            row = self._rows.get(hash_token(token))
            if row is not None and row.status == "resuming" and row.claim_id == claim_id:
                row.status = "used"

    async def release(self, token: str, claim_id: str, tokens: int = 0) -> None:
        with self._lock:
            row = self._rows.get(hash_token(token))
            if row is not None and row.status == "resuming" and row.claim_id == claim_id:
                row.status, row.claimed_at, row.tokens = "pending", None, max(row.tokens, tokens)


class SqliteResumeStore:
    """Tokens in the session SQLite file. A connection per operation in a worker thread;
    `claim` checks and marks the token in one write transaction."""

    def __init__(
        self,
        path: Path,
        ttl_s: float,
        now: Callable[[], float] = time.time,
        lease_s: float = DEFAULT_LEASE_S,
    ) -> None:
        self.path = path
        self.ttl_s = ttl_s
        self.lease_s = lease_s
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
        in_flight_since = now - self.lease_s
        where = (
            "expires_at <= ? AND NOT (status = 'resuming' AND COALESCE(claimed_at, 0) > ?)",
            (now, in_flight_since),
        )
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(f"SELECT thread_id FROM resumes WHERE {where[0]}", where[1])
            threads = [r[0] for r in rows.fetchall()]
            conn.execute(f"DELETE FROM resumes WHERE {where[0]}", where[1])
        return threads

    def _claim(self, token: str, caller_id: str, kb: str, session_id: str | None) -> ResumeRecord:
        with closing(self._connect()) as conn:
            conn.isolation_level = None  # explicit transaction: the read and the update are one
            conn.execute("BEGIN IMMEDIATE")
            try:
                found = conn.execute(
                    "SELECT caller_id, kb, session_id, thread_id, expires_at, status, claimed_at, "
                    "tokens FROM resumes WHERE token_hash = ?",
                    (hash_token(token),),
                ).fetchone()
                record = ResumeRecord(*found[:5]) if found else None
                now = self._now()
                problem = _problem(
                    record,
                    found[5] if found else "",
                    found[6] if found else None,
                    caller_id=caller_id,
                    kb=kb,
                    session_id=session_id,
                    now=now,
                    lease_s=self.lease_s,
                )
                if problem is not None or record is None:
                    raise ResumeTokenInvalid(problem or "unknown token")
                claim_id = new_claim_id()
                conn.execute(
                    "UPDATE resumes SET status = 'resuming', claimed_at = ?, claim_id = ? "
                    "WHERE token_hash = ?",
                    (now, claim_id, hash_token(token)),
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return replace(record, claim_id=claim_id, tokens=int(found[7] or 0))

    def _end(self, sql: str, args: tuple[object, ...]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(sql, args)

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

    async def finish(self, token: str, claim_id: str) -> None:
        await asyncio.to_thread(
            self._end,
            "UPDATE resumes SET status = 'used' "
            "WHERE token_hash = ? AND status = 'resuming' AND claim_id = ?",
            (hash_token(token), claim_id),
        )

    async def release(self, token: str, claim_id: str, tokens: int = 0) -> None:
        await asyncio.to_thread(
            self._end,
            "UPDATE resumes SET status = 'pending', claimed_at = NULL, "
            "tokens = MAX(COALESCE(tokens, 0), ?) "
            "WHERE token_hash = ? AND status = 'resuming' AND claim_id = ?",
            (tokens, hash_token(token), claim_id),
        )
