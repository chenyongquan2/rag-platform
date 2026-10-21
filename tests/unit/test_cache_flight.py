"""Spec P5-01 AC4: single flight. N identical requests that arrive together make one
computation (threads, asyncio, and two cache instances on one store standing for two processes);
a leader that fails or is cancelled does not take the followers down with it."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from rag.cache.exact import CacheConfig, CacheRun, Codec, ExactCache
from rag.cache.keys import CacheKeyFields, make_fields
from rag.cache.rules import Entry, search_cacheable, search_from_payload, search_payload
from rag.cache.store import CacheUnavailable, MemoryCacheBackend
from rag.testing.agent import pages_result
from rag.types import RetrievalResult


def _codec() -> Codec[RetrievalResult]:
    return Codec(to_payload=search_payload, verdict=search_cacheable, not_found=lambda r: False)


def _decode(entry: Entry) -> RetrievalResult:
    return search_from_payload(entry.payload, request_id="cached", data_generation=0, timings_ms={})


def _build(question: str = "How do I push?") -> Callable[[int], CacheKeyFields]:
    def build(generation: int) -> CacheKeyFields:
        return make_fields(
            "search",
            kb="hf_docs",
            question=question,
            acl_groups=["public"],
            top_k=3,
            data_generation=generation,
            config_version="cfg",
        )

    return build


def _adapt(request_id: str) -> Callable[[RetrievalResult], RetrievalResult]:
    return lambda r: r.model_copy(update={"request_id": request_id})


class SlowCompute:
    """A computation of `seconds` that counts how often it ran and can be told to fail."""

    def __init__(self, seconds: float = 0.15, *, fail_first: bool = False) -> None:
        self.seconds = seconds
        self.calls = 0
        self.fail_first = fail_first
        self._lock = threading.Lock()

    def _begin(self) -> int:
        with self._lock:
            self.calls += 1
            return self.calls

    def _result(self, n: int) -> RetrievalResult:
        return pages_result("hf_docs", [1, 2, 3]).model_copy(update={"request_id": f"leader-{n}"})

    def __call__(self) -> RetrievalResult:
        n = self._begin()
        time.sleep(self.seconds)
        if self.fail_first and n == 1:
            raise RuntimeError("leader failed")
        return self._result(n)

    async def acall(self) -> RetrievalResult:
        n = self._begin()
        await asyncio.sleep(self.seconds)
        if self.fail_first and n == 1:
            raise RuntimeError("leader failed")
        return self._result(n)


def _cache(backend: MemoryCacheBackend | None = None, **config: Any) -> ExactCache:
    return ExactCache(backend or MemoryCacheBackend(), CacheConfig(**config))


def _many(cache: ExactCache, compute: SlowCompute, n: int, **kwargs: Any) -> list[Any]:
    barrier = threading.Barrier(n)

    def one(i: int) -> CacheRun[RetrievalResult] | Exception:
        barrier.wait()
        try:
            return cache.get_or_compute(
                _build(),
                compute,
                codec=_codec(),
                decode=_decode,
                adapt=_adapt(f"req-{i}"),
                **kwargs,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def test_n_concurrent_identical_requests_compute_once_threads() -> None:
    compute = SlowCompute()
    runs = _many(_cache(), compute, 20)
    assert compute.calls == 1
    assert all(isinstance(r, CacheRun) for r in runs)
    outcomes = sorted(r.outcome for r in runs)
    assert outcomes.count("miss") == 1 and set(outcomes) <= {"miss", "shared", "hit"}
    assert all(len(r.result.chunks) == 3 for r in runs)
    # Each follower's result belongs to its own request.
    followers = [r for r in runs if r.outcome == "shared"]
    assert followers and all(r.result.request_id.startswith("req-") for r in followers)


def test_n_concurrent_identical_async_requests_compute_once() -> None:
    cache, compute = _cache(), SlowCompute()

    async def one(i: int) -> CacheRun[RetrievalResult]:
        return await cache.aget_or_compute(
            _build(), compute.acall, codec=_codec(), decode=_decode, adapt=_adapt(f"req-{i}")
        )

    async def main() -> list[CacheRun[RetrievalResult]]:
        return list(await asyncio.gather(*(one(i) for i in range(20))))

    runs = asyncio.run(main())
    assert compute.calls == 1
    assert sorted(r.outcome for r in runs).count("miss") == 1
    assert all(r.result.chunks for r in runs)


def test_different_keys_do_not_wait_for_each_other() -> None:
    cache = _cache()
    compute = SlowCompute(0.3)
    began = time.perf_counter()

    def one(q: str) -> CacheRun[RetrievalResult]:
        return cache.get_or_compute(_build(q), compute, codec=_codec(), decode=_decode)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, ["first question", "second question"]))
    assert compute.calls == 2
    assert time.perf_counter() - began < 0.55  # parallel, not 0.6


def test_two_processes_on_one_store_compute_once_through_the_redis_lock() -> None:
    backend = MemoryCacheBackend()
    one, two = _cache(backend, lock_poll_ms=5), _cache(backend, lock_poll_ms=5)
    compute = SlowCompute(0.2)
    barrier = threading.Barrier(2)
    runs: list[CacheRun[RetrievalResult]] = []

    def go(cache: ExactCache) -> None:
        barrier.wait()
        runs.append(cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode))

    threads = [threading.Thread(target=go, args=(c,)) for c in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert compute.calls == 1
    assert sorted(r.outcome for r in runs) in (["miss", "shared"], ["hit", "miss"])
    assert not [k for k in backend.live_keys() if k.endswith(":lock")]  # the lock is gone


def test_a_failed_leader_hands_over_to_one_follower_and_only_the_leader_sees_the_error() -> None:
    compute = SlowCompute(0.1, fail_first=True)
    runs = _many(_cache(), compute, 6)
    errors = [r for r in runs if isinstance(r, Exception)]
    done = [r for r in runs if isinstance(r, CacheRun)]
    assert len(errors) == 1 and "leader failed" in str(errors[0])
    assert len(done) == 5 and all(r.result.chunks for r in done)
    assert compute.calls == 2  # the leader, then one follower that became the new leader


def test_a_cancelled_async_leader_does_not_take_the_followers_down() -> None:
    cache, compute = _cache(), SlowCompute(0.2)

    async def one(i: int) -> CacheRun[RetrievalResult]:
        return await cache.aget_or_compute(
            _build(), compute.acall, codec=_codec(), decode=_decode, adapt=_adapt(f"req-{i}")
        )

    async def main() -> list[Any]:
        leader = asyncio.create_task(one(0))
        await asyncio.sleep(0.02)  # the leader has the flight
        followers = [asyncio.create_task(one(i)) for i in range(1, 4)]
        await asyncio.sleep(0.02)
        leader.cancel()
        results = await asyncio.gather(leader, *followers, return_exceptions=True)
        return results

    results = asyncio.run(main())
    assert isinstance(results[0], asyncio.CancelledError)
    assert all(isinstance(r, CacheRun) and r.result.chunks for r in results[1:])
    assert compute.calls == 2


def test_a_follower_that_waits_too_long_computes_for_itself() -> None:
    cache = _cache()
    compute = SlowCompute(0.4)
    leader_done = threading.Event()

    def leader() -> None:
        cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)
        leader_done.set()

    t = threading.Thread(target=leader)
    t.start()
    time.sleep(0.05)
    run = cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode, wait_s=0.05)
    t.join()
    assert run.outcome == "miss" and compute.calls == 2


def test_a_lock_that_cannot_be_taken_does_not_stop_the_computation() -> None:
    backend = MemoryCacheBackend()
    cache = _cache(backend)

    def broken(key: str, token: str, ttl_ms: int) -> bool:
        raise CacheUnavailable("TimeoutError")

    backend.set_nx = broken  # type: ignore[method-assign]
    compute = SlowCompute(0.0)
    run = cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)
    assert run.outcome == "miss" and compute.calls == 1


def test_the_lock_is_released_when_the_computation_fails() -> None:
    backend = MemoryCacheBackend()
    cache = _cache(backend)

    def boom() -> RetrievalResult:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cache.get_or_compute(_build(), boom, codec=_codec(), decode=_decode)
    assert not [k for k in backend.live_keys() if k.endswith(":lock")]
    run = cache.get_or_compute(_build(), SlowCompute(0.0), codec=_codec(), decode=_decode)
    assert run.outcome == "miss"
