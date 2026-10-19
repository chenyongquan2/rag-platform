"""Spec P3-02 AC14, AC17: the resume token store (memory and SQLite behave the same)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rag.agent.resume import MemoryResumeStore, ResumeTokenInvalid, SqliteResumeStore
from rag.factories import make_resume_store
from rag.types import ErrorCategory


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(params=["memory", "sqlite"])
def store_and_clock(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Any, Clock]:
    clock = Clock()
    if request.param == "memory":
        return MemoryResumeStore(60, now=clock), clock
    return SqliteResumeStore(tmp_path / "sessions.db", 60, now=clock), clock


def _create(store: Any, **overrides: Any) -> str:
    args: dict[str, Any] = {
        "caller_id": "alice",
        "kb": "hf_docs",
        "session_id": "s1",
        "thread_id": "s1:run1",
    }
    return asyncio.run(store.create(**{**args, **overrides}))


def _claim(store: Any, token: str, **overrides: Any) -> Any:
    args: dict[str, Any] = {"caller_id": "alice", "kb": "hf_docs", "session_id": "s1"}
    return asyncio.run(store.claim(token, **{**args, **overrides}))


def test_token_is_claimed_once_by_its_owner(store_and_clock: tuple[Any, Clock]) -> None:
    store, _ = store_and_clock
    token = _create(store)
    assert len(token) >= 30 and _create(store) != token  # random, not derived from anything
    record = _claim(store, token)
    assert (record.thread_id, record.caller_id, record.kb, record.session_id) == (
        "s1:run1",
        "alice",
        "hf_docs",
        "s1",
    )
    with pytest.raises(ResumeTokenInvalid, match="in use"):  # a call is running the thread
        _claim(store, token)
    asyncio.run(store.finish(token, record.claim_id))
    with pytest.raises(ResumeTokenInvalid, match="already used"):
        _claim(store, token)
    assert ResumeTokenInvalid.error_category == ErrorCategory.INVALID_RESUME_TOKEN


def test_wrong_owner_is_refused_and_does_not_use_the_token(
    store_and_clock: tuple[Any, Clock],
) -> None:
    store, _ = store_and_clock
    token = _create(store)
    for wrong in (
        {"caller_id": "bob"},
        {"kb": "other"},
        {"session_id": "s2"},
        {"session_id": None},
    ):
        with pytest.raises(ResumeTokenInvalid, match="another caller"):
            _claim(store, token, **wrong)
    assert _claim(store, token).thread_id == "s1:run1"  # still unused for its owner

    # A token made for a call without a session only opens a call without one.
    bare = _create(store, session_id=None, thread_id="-:run2")
    with pytest.raises(ResumeTokenInvalid):
        _claim(store, bare, session_id="s1")
    assert _claim(store, bare, session_id=None).thread_id == "-:run2"


def test_unknown_and_expired_tokens_are_refused(store_and_clock: tuple[Any, Clock]) -> None:
    store, clock = store_and_clock
    with pytest.raises(ResumeTokenInvalid, match="unknown"):
        _claim(store, "nope")
    token = _create(store, thread_id="s1:old")
    live = _create(store, thread_id="s1:live")
    clock.now += 30
    fresh = _create(store, thread_id="s1:fresh")
    clock.now += 31  # the first two are 61 s old, the last 31 s
    with pytest.raises(ResumeTokenInvalid, match="expired"):
        _claim(store, token)
    # Purging drops what expired and says which checkpoint threads nobody can resume now.
    assert sorted(asyncio.run(store.purge_expired())) == ["s1:live", "s1:old"]
    assert asyncio.run(store.purge_expired()) == []
    with pytest.raises(ResumeTokenInvalid, match="unknown"):
        _claim(store, live)
    assert _claim(store, fresh).thread_id == "s1:fresh"


def test_concurrent_claims_of_one_token_have_one_winner(
    store_and_clock: tuple[Any, Clock],
) -> None:
    store, _ = store_and_clock
    token = _create(store)

    async def race() -> list[Any]:
        calls = [
            store.claim(token, caller_id="alice", kb="hf_docs", session_id="s1") for _ in range(12)
        ]
        return await asyncio.gather(*calls, return_exceptions=True)

    outcomes = asyncio.run(race())
    winners = [o for o in outcomes if not isinstance(o, Exception)]
    assert len(winners) == 1
    assert all(isinstance(o, ResumeTokenInvalid) for o in outcomes if o not in winners)


def test_sqlite_store_keeps_only_a_hash_and_is_made_from_the_url(tmp_path: Path) -> None:
    url = f"sqlite://{(tmp_path / 'sessions.db').as_posix()}"
    store = make_resume_store(url, ttl_s=60)
    assert isinstance(store, SqliteResumeStore) and isinstance(
        make_resume_store("memory://", ttl_s=60), MemoryResumeStore
    )
    token = _create(store)
    raw = (tmp_path / "sessions.db").read_bytes() + b"".join(
        p.read_bytes() for p in tmp_path.glob("sessions.db-*")
    )
    assert token.encode() not in raw
    with pytest.raises(NotImplementedError, match="resume store"):
        make_resume_store("redis://localhost", ttl_s=60)


# ---- review round 1: in flight, finish, release, lease ------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def leased(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Any, Clock]:
    clock = Clock()
    if request.param == "memory":
        return MemoryResumeStore(60, now=clock, lease_s=20), clock
    return SqliteResumeStore(tmp_path / "sessions.db", 60, now=clock, lease_s=20), clock


def test_a_claimed_token_is_in_flight_until_it_is_finished_or_released(
    leased: tuple[Any, Clock],
) -> None:
    store, _clock = leased
    token = _create(store)
    first = _claim(store, token)
    with pytest.raises(ResumeTokenInvalid, match="in use"):
        _claim(store, token)
    asyncio.run(store.release(token, first.claim_id))  # the run failed: usable again
    second = _claim(store, token)
    assert second.claim_id != first.claim_id
    asyncio.run(store.finish(token, second.claim_id))  # the run ended: used up for good
    asyncio.run(store.release(token, second.claim_id))  # (a late release changes nothing)
    with pytest.raises(ResumeTokenInvalid, match="already used"):
        _claim(store, token)


def test_a_lapsed_lease_frees_the_token_of_a_run_that_never_ended(
    leased: tuple[Any, Clock],
) -> None:
    store, clock = leased
    token = _create(store)
    _claim(store, token)
    clock.now += 19
    with pytest.raises(ResumeTokenInvalid, match="in use"):
        _claim(store, token)
    clock.now += 2  # 21 s after the claim: the process that held it is gone
    assert _claim(store, token).thread_id == "s1:run1"


def test_purge_keeps_the_thread_of_a_run_in_flight(leased: tuple[Any, Clock]) -> None:
    store, clock = leased
    store.lease_s = 40
    flying = _create(store, thread_id="s1:flying")
    _create(store, thread_id="s1:idle")
    done = _create(store, thread_id="s1:done")
    clock.now += 30
    _claim(store, flying)
    asyncio.run(store.finish(done, _claim(store, done).claim_id))
    clock.now += 31  # 61 s: all three tokens have expired (60 s), the flying run's lease has not
    assert sorted(asyncio.run(store.purge_expired())) == ["s1:done", "s1:idle"]
    clock.now += 10  # the lease (40 s from the claim) has lapsed: nobody is running it
    assert asyncio.run(store.purge_expired()) == ["s1:flying"]


def test_an_old_sqlite_file_without_the_lease_column_is_upgraded(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "sessions.db"
    with sqlite3.connect(path) as conn:  # the table as the first P3-02 build made it
        conn.execute(
            "CREATE TABLE resumes (token_hash TEXT PRIMARY KEY, caller_id TEXT NOT NULL, "
            "kb TEXT NOT NULL, session_id TEXT, thread_id TEXT NOT NULL, status TEXT NOT NULL, "
            "expires_at REAL NOT NULL)"
        )
    store = SqliteResumeStore(path, 60)
    token = _create(store)
    assert _claim(store, token).thread_id == "s1:run1"


# ---- review round 2: only the call that holds the claim can end it ------------------------------


def test_a_late_release_or_finish_of_a_lapsed_claim_changes_nothing(
    leased: tuple[Any, Clock],
) -> None:
    store, clock = leased
    token = _create(store)
    first = _claim(store, token)
    clock.now += 21  # the first call is slow: its lease lapsed
    second = _claim(store, token)  # another call took the token over
    assert second.claim_id != first.claim_id
    asyncio.run(store.release(token, first.claim_id))  # the first call fails late ...
    with pytest.raises(ResumeTokenInvalid, match="in use"):  # ... the second still holds it
        _claim(store, token)
    asyncio.run(store.finish(token, first.claim_id))  # ... or ends late: not its claim any more
    with pytest.raises(ResumeTokenInvalid, match="in use"):
        _claim(store, token)
    asyncio.run(store.finish(token, second.claim_id))  # the holder ends it
    with pytest.raises(ResumeTokenInvalid, match="already used"):
        _claim(store, token)


def test_tokens_spent_by_a_failed_attempt_come_back_with_the_next_claim(
    leased: tuple[Any, Clock],
) -> None:
    store, _clock = leased
    token = _create(store)
    first = _claim(store, token)
    assert first.tokens == 0
    asyncio.run(store.release(token, first.claim_id, tokens=14_000))
    assert _claim(store, token).tokens == 14_000
