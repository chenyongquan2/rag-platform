"""The exact cache (L1) around one computation (design s12-1, s16-8; spec P5-01).

    cache = ExactCache(backend, config)
    run = cache.get_or_compute(build, compute, codec=codec, decode=decode, adapt=adapt)
    run.result       # the cached result, a shared one, or the one `compute` made
    run.outcome      # hit | shared | miss | bypass | error

What one call does:

1. Builds the key: reads the knowledge base's data generation from the store, then
   `build(generation)` makes the key fields (`rag.cache.keys`). A generation that cannot be read
   means no safe key: the request is computed without the cache (`error` or `bypass`).
2. Single flight: the first request for a key is the leader, identical requests that arrive
   while it works wait for its result (an in-process future) instead of computing again. A
   follower waits at most `wait_s`, and a leader that failed or was cancelled hands the work to
   the next follower; nobody sees another request's error.
3. The leader reads the entry; a hit ends it. On a miss it takes a short Redis lock
   (`SET NX PX`), so a second process that misses at the same time waits for the entry rather
   than computing it again (it polls the key for at most `lock_wait_ms`, then computes).
4. Computes, and writes the result when `codec.verdict` says it is complete and the value is
   not larger than `max_value_bytes`. Entries expire (`ttl_s`, a not-found result sooner).

Failure policy: any failure of the store is `CacheUnavailable` inside this class and becomes an
`error` outcome (or `bypass` while the breaker is open); the computation still runs and its
result is returned. Nothing here raises for the store. Exceptions of `compute` propagate.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, TypeVar

from rag.cache.keys import CacheKeyFields, cache_key
from rag.cache.rules import Entry, Verdict, decode_entry, encode_entry
from rag.cache.store import CacheBackend, CacheUnavailable, CircuitBreaker

T = TypeVar("T")

FLIGHT_ATTEMPTS = 3
"""How often a follower tries to follow a leader before it computes for itself."""


@dataclass(frozen=True)
class CacheConfig:
    ttl_s: int = 3600
    not_found_ttl_s: int = 600
    max_value_bytes: int = 262_144
    lock_ttl_ms: int = 30_000
    lock_wait_ms: int = 3_000
    lock_poll_ms: int = 25
    breaker_failures: int = 5
    breaker_open_s: float = 5.0


@dataclass(frozen=True)
class Codec[T]:
    """What the cache needs to know about a kind of result."""

    to_payload: Callable[[T], dict[str, Any]]
    verdict: Callable[[T], Verdict]
    not_found: Callable[[T], bool]


@dataclass
class CacheRun[T]:
    """How one request went through the cache."""

    result: T
    outcome: str
    reason: str = ""
    key: str | None = None
    generation: int | None = None
    stored: bool = False
    not_stored_reason: str = ""
    age_s: float | None = None
    lookup_ms: float = 0.0
    store_ms: float = 0.0
    elapsed_ms: float = 0.0


class _LeaderFailed(Exception):  # noqa: N818 - private signal, not an error of the caller
    """The leader of a flight ended without a result (it raised or was cancelled)."""


class _Flights:
    """The computations in progress, one future per key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[str, Future[CacheRun[Any]]] = {}

    def begin(self, key: str) -> tuple[Future[CacheRun[Any]], bool]:
        with self._lock:
            future = self._open.get(key)
            if future is not None:
                return future, False
            future = Future()
            self._open[key] = future
            return future, True

    def finish(
        self, key: str, future: Future[CacheRun[Any]], run: CacheRun[Any] | None = None
    ) -> None:
        """Remove the flight, then wake the followers (so a follower that retries never finds
        a flight that is over)."""
        with self._lock:
            if self._open.get(key) is future:
                del self._open[key]
        if run is not None:
            future.set_result(run)
        else:
            future.set_exception(_LeaderFailed())


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


class ExactCache:
    def __init__(
        self,
        backend: CacheBackend,
        config: CacheConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or CacheConfig()
        self._clock = clock
        self.breaker = breaker or CircuitBreaker(
            failures=self.config.breaker_failures, open_s=self.config.breaker_open_s
        )
        self._flights = _Flights()

    # ---- steps (blocking; the async flow runs them in a thread) ------------------------------

    def _guard(self, fn: Callable[..., Any], *args: Any) -> Any:
        """One call of the store, through the breaker; only `CacheUnavailable` leaves."""
        if not self.breaker.allow():
            raise CacheUnavailable("breaker_open")
        try:
            value = fn(*args)
        except CacheUnavailable:
            self.breaker.failure()
            raise
        except Exception as exc:  # a store that fails in a way we did not expect
            self.breaker.failure()
            raise CacheUnavailable(type(exc).__name__) from exc
        self.breaker.success()
        return value

    def _prepare(
        self, build: Callable[[int], CacheKeyFields], kb: str | None
    ) -> tuple[CacheKeyFields, str]:
        kb = kb if kb is not None else build(0).kb
        generation = int(self._guard(self.backend.generation, kb))
        fields = build(generation)
        return fields, cache_key(fields)

    def _read(self, key: str) -> Entry | None:
        text = self._guard(self.backend.get, key)
        return decode_entry(text) if text is not None else None

    def _try_lock(self, key: str, token: str) -> bool | None:
        """True: the lock is ours; False: another process holds it; None: the store failed."""
        try:
            return bool(
                self._guard(self.backend.set_nx, key + ":lock", token, self.config.lock_ttl_ms)
            )
        except CacheUnavailable:
            return None

    def _release(self, key: str, token: str) -> None:
        with contextlib.suppress(CacheUnavailable):  # the lock expires by itself
            self._guard(self.backend.release, key + ":lock", token)

    def _store(self, run: CacheRun[T], fields: CacheKeyFields, codec: Codec[T]) -> None:
        began = time.perf_counter()
        verdict = codec.verdict(run.result)
        if not verdict.ok:
            run.not_stored_reason = verdict.reason
            return
        text = encode_entry(fields, codec.to_payload(run.result), created_at=self._clock())
        if len(text.encode("utf-8")) > self.config.max_value_bytes:
            run.not_stored_reason = "too_large"
            return
        ttl = self.config.not_found_ttl_s if codec.not_found(run.result) else self.config.ttl_s
        try:
            self._guard(self.backend.set, run.key, text, ttl)
        except CacheUnavailable as exc:
            run.not_stored_reason = exc.reason
            return
        finally:
            run.store_ms = _ms(began)
        run.stored = True

    @staticmethod
    def _hit(
        entry: Entry, decode: Callable[[Entry], T], key: str, now: float
    ) -> CacheRun[T] | None:
        try:
            result = decode(entry)
        except Exception:  # an entry another version of the code wrote: a miss
            return None
        return CacheRun(result, "hit", key=key, age_s=max(0.0, now - entry.created_at))

    def _skipped(self, exc: CacheUnavailable) -> tuple[str, str]:
        return ("bypass" if exc.reason == "breaker_open" else "error"), exc.reason

    @staticmethod
    def _direct(result: T, key: str, read: bool) -> CacheRun[T]:
        """A result computed outside any flight: by a call that may not read the cache
        (`bypass`, and its result is still written when complete), or by a follower that gave
        up waiting (`miss`)."""
        if read:
            return CacheRun(result, "miss", key=key)
        return CacheRun(result, "bypass", reason="read_off", key=key)

    @staticmethod
    def _follow(shared: CacheRun[T], adapt: Callable[[T], T] | None, began: float) -> CacheRun[T]:
        result = adapt(shared.result) if adapt is not None else shared.result
        outcome = shared.outcome if shared.outcome == "hit" else "shared"
        run = CacheRun(
            result,
            outcome,
            key=shared.key,
            generation=shared.generation,
            age_s=shared.age_s,
            lookup_ms=shared.lookup_ms,
        )
        run.elapsed_ms = _ms(began)
        return run

    # ---- the call ----------------------------------------------------------------------------

    def get_or_compute(
        self,
        build: Callable[[int], CacheKeyFields],
        compute: Callable[[], T],
        *,
        codec: Codec[T],
        decode: Callable[[Entry], T],
        adapt: Callable[[T], T] | None = None,
        wait_s: float | None = None,
        read: bool = True,
        kb: str | None = None,
    ) -> CacheRun[T]:
        """See the module doc. `read=False` computes without looking at the cache and without a
        flight, but still writes a complete result (a call that must not be served from the cache
        may still fill it). `adapt` gives a follower its own copy of the leader's result (its
        request id)."""
        began = time.perf_counter()
        try:
            fields, key = self._prepare(build, kb)
        except CacheUnavailable as exc:
            outcome, reason = self._skipped(exc)
            run = CacheRun(compute(), outcome, reason=reason)
            run.lookup_ms = run.elapsed_ms = _ms(began)
            return run
        wait = wait_s if wait_s is not None else self.config.lock_wait_ms / 1000.0
        if read:
            for _ in range(FLIGHT_ATTEMPTS):
                flight, leader = self._flights.begin(key)
                if leader:
                    return self._lead(flight, key, fields, compute, codec, decode, wait, began)
                try:
                    return self._follow(flight.result(timeout=wait), adapt, began)
                except FutureTimeout:
                    break
                except _LeaderFailed:
                    continue
        run = self._direct(compute(), key, read)
        run.generation = fields.data_generation
        self._store(run, fields, codec)
        run.elapsed_ms = _ms(began)
        return run

    def _lead(
        self,
        flight: Future[CacheRun[T]],
        key: str,
        fields: CacheKeyFields,
        compute: Callable[[], T],
        codec: Codec[T],
        decode: Callable[[Entry], T],
        wait: float,
        began: float,
    ) -> CacheRun[T]:
        try:
            run = self._lead_inner(key, fields, compute, codec, decode, wait, began)
            run.generation = fields.data_generation
        except BaseException:
            self._flights.finish(key, flight)
            raise
        self._flights.finish(key, flight, run)
        return run

    def _lead_inner(
        self,
        key: str,
        fields: CacheKeyFields,
        compute: Callable[[], T],
        codec: Codec[T],
        decode: Callable[[Entry], T],
        wait: float,
        began: float,
    ) -> CacheRun[T]:
        try:
            entry = self._read(key)
        except CacheUnavailable as exc:
            outcome, reason = self._skipped(exc)
            run = CacheRun(compute(), outcome, reason=reason, key=key)
            run.lookup_ms = run.elapsed_ms = _ms(began)
            return run
        if entry is not None:
            hit = self._hit(entry, decode, key, self._clock())
            if hit is not None:
                hit.lookup_ms = hit.elapsed_ms = _ms(began)
                return hit
        lookup_ms = _ms(began)
        token = uuid.uuid4().hex
        locked = self._try_lock(key, token)
        try:
            if locked is False:  # another process computes it: wait for its entry
                waited = self._poll(key, wait, decode)
                if waited is not None:
                    waited.outcome = "shared"
                    waited.lookup_ms, waited.elapsed_ms = lookup_ms, _ms(began)
                    return waited
            run = CacheRun(compute(), "miss", key=key, lookup_ms=lookup_ms)
            self._store(run, fields, codec)
        finally:
            if locked:
                self._release(key, token)
        run.elapsed_ms = _ms(began)
        return run

    def _poll(self, key: str, wait: float, decode: Callable[[Entry], T]) -> CacheRun[T] | None:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(self.config.lock_poll_ms / 1000.0)
            try:
                entry = self._read(key)
            except CacheUnavailable:
                return None
            if entry is not None:
                return self._hit(entry, decode, key, self._clock())
        return None

    # ---- the same, for a computation that is a coroutine -------------------------------------

    async def aget_or_compute(
        self,
        build: Callable[[int], CacheKeyFields],
        compute: Callable[[], Awaitable[T]],
        *,
        codec: Codec[T],
        decode: Callable[[Entry], T],
        adapt: Callable[[T], T] | None = None,
        wait_s: float | None = None,
        read: bool = True,
        kb: str | None = None,
    ) -> CacheRun[T]:
        """`get_or_compute` for `ask_docs`: the store is called in a worker thread (a time-out
        must not stop the event loop), the flight is awaited."""
        began = time.perf_counter()
        try:
            fields, key = await asyncio.to_thread(self._prepare, build, kb)
        except CacheUnavailable as exc:
            outcome, reason = self._skipped(exc)
            run = CacheRun(await compute(), outcome, reason=reason)
            run.lookup_ms = run.elapsed_ms = _ms(began)
            return run
        wait = wait_s if wait_s is not None else self.config.lock_wait_ms / 1000.0
        if read:
            for _ in range(FLIGHT_ATTEMPTS):
                flight, leader = self._flights.begin(key)
                if leader:
                    return await self._alead(
                        flight, key, fields, compute, codec, decode, wait, began
                    )
                try:
                    shared = await asyncio.wait_for(asyncio.wrap_future(flight), timeout=wait)
                except TimeoutError:
                    break
                except _LeaderFailed:
                    continue
                return self._follow(shared, adapt, began)
        run = self._direct(await compute(), key, read)
        run.generation = fields.data_generation
        await asyncio.to_thread(self._store, run, fields, codec)
        run.elapsed_ms = _ms(began)
        return run

    async def _alead(
        self,
        flight: Future[CacheRun[T]],
        key: str,
        fields: CacheKeyFields,
        compute: Callable[[], Awaitable[T]],
        codec: Codec[T],
        decode: Callable[[Entry], T],
        wait: float,
        began: float,
    ) -> CacheRun[T]:
        try:
            run = await self._alead_inner(key, fields, compute, codec, decode, wait, began)
            run.generation = fields.data_generation
        except BaseException:
            self._flights.finish(key, flight)
            raise
        self._flights.finish(key, flight, run)
        return run

    async def _alead_inner(
        self,
        key: str,
        fields: CacheKeyFields,
        compute: Callable[[], Awaitable[T]],
        codec: Codec[T],
        decode: Callable[[Entry], T],
        wait: float,
        began: float,
    ) -> CacheRun[T]:
        try:
            entry = await asyncio.to_thread(self._read, key)
        except CacheUnavailable as exc:
            outcome, reason = self._skipped(exc)
            run = CacheRun(await compute(), outcome, reason=reason, key=key)
            run.lookup_ms = run.elapsed_ms = _ms(began)
            return run
        if entry is not None:
            hit = self._hit(entry, decode, key, self._clock())
            if hit is not None:
                hit.lookup_ms = hit.elapsed_ms = _ms(began)
                return hit
        lookup_ms = _ms(began)
        token = uuid.uuid4().hex
        locked = await asyncio.to_thread(self._try_lock, key, token)
        try:
            if locked is False:
                waited = await self._apoll(key, wait, decode)
                if waited is not None:
                    waited.outcome = "shared"
                    waited.lookup_ms, waited.elapsed_ms = lookup_ms, _ms(began)
                    return waited
            run = CacheRun(await compute(), "miss", key=key, lookup_ms=lookup_ms)
            await asyncio.to_thread(self._store, run, fields, codec)
        finally:
            if locked:
                await asyncio.shield(asyncio.to_thread(self._release, key, token))
        run.elapsed_ms = _ms(began)
        return run

    async def _apoll(
        self, key: str, wait: float, decode: Callable[[Entry], T]
    ) -> CacheRun[T] | None:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            await asyncio.sleep(self.config.lock_poll_ms / 1000.0)
            try:
                entry = await asyncio.to_thread(self._read, key)
            except CacheUnavailable:
                return None
            if entry is not None:
                return self._hit(entry, decode, key, self._clock())
        return None

    # ---- admin -------------------------------------------------------------------------------

    def drop_prefix(self, prefix: str) -> int:
        """Remove the entries (and locks) whose key starts with `prefix`; raises
        `CacheUnavailable`, an admin wants to know."""
        return int(self.backend.delete_prefix(prefix))
