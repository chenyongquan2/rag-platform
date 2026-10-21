"""Spec P5-01 AC5 (and parts of AC2/AC3): the exact cache core: hit and miss, what is stored, and
that a store that fails never fails the request."""

from __future__ import annotations

import time
from typing import Any

import pytest

from rag.cache.exact import CacheConfig, CacheRun, Codec, ExactCache
from rag.cache.keys import CacheKeyFields, make_fields
from rag.cache.rules import (
    Entry,
    Verdict,
    search_cacheable,
    search_from_payload,
    search_payload,
)
from rag.cache.store import CacheUnavailable, CircuitBreaker, MemoryCacheBackend, RedisCacheBackend
from rag.factories import make_redis_client
from rag.testing.agent import pages_result
from rag.types import RetrievalResult


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _codec() -> Codec[RetrievalResult]:
    return Codec(
        to_payload=search_payload,
        verdict=search_cacheable,
        not_found=lambda r: r.not_found,
    )


def _decode(entry: Entry) -> RetrievalResult:
    return search_from_payload(entry.payload, request_id="hit", data_generation=0, timings_ms={})


def _build(question: str = "How do I push?", **over: Any) -> Any:
    def build(generation: int) -> CacheKeyFields:
        return make_fields(
            "search",
            kb="hf_docs",
            question=question,
            acl_groups=["public"],
            top_k=3,
            data_generation=generation,
            config_version="cfg",
            **over,
        )

    return build


class Counter:
    def __init__(self, result: RetrievalResult | None = None, **over: Any) -> None:
        self.calls = 0
        self.result = result or pages_result("hf_docs", [1, 2, 3])
        self.over = over

    def __call__(self) -> RetrievalResult:
        self.calls += 1
        return self.result.model_copy(update={"request_id": f"fresh-{self.calls}", **self.over})


def _cache(
    backend: Any = None,
    *,
    clock: Clock | None = None,
    breaker: CircuitBreaker | None = None,
    **config: Any,
) -> tuple[ExactCache, MemoryCacheBackend]:
    clock = clock or Clock()
    backend = backend or MemoryCacheBackend(clock=clock)
    cache = ExactCache(
        backend,
        CacheConfig(**config),
        clock=clock,
        breaker=breaker or CircuitBreaker(failures=3, open_s=5.0, clock=clock),
    )
    return cache, backend


def _run(cache: ExactCache, compute: Any, build: Any = None, **kwargs: Any) -> CacheRun[Any]:
    return cache.get_or_compute(
        build or _build(), compute, codec=_codec(), decode=_decode, **kwargs
    )


def test_miss_then_hit_and_the_second_request_does_not_compute() -> None:
    cache, _ = _cache()
    compute = Counter()
    first = _run(cache, compute)
    second = _run(cache, compute)
    assert (first.outcome, first.stored) == ("miss", True)
    assert (second.outcome, second.result.request_id) == ("hit", "hit")
    assert compute.calls == 1
    assert [c.chunk_id for c in second.result.chunks] == [c.chunk_id for c in first.result.chunks]
    assert second.age_s is not None and second.age_s >= 0


def test_a_hit_carries_created_at_and_the_key_fields_in_the_value() -> None:
    clock = Clock(5000.0)
    cache, backend = _cache(clock=clock)
    _run(cache, Counter())
    (key,) = backend.live_keys()
    import json

    entry = json.loads(backend.get(key) or "")
    assert entry["created_at"] == 5000.0
    assert entry["key_fields"]["kb"] == "hf_docs" and entry["key_fields"]["acl"] == ["public"]
    assert entry["key_fields"]["data_generation"] == backend.generation("hf_docs")
    clock.now += 42
    assert _run(cache, Counter()).age_s == pytest.approx(42)


def test_equivalent_questions_share_an_entry_and_different_ones_do_not() -> None:
    cache, _ = _cache()
    compute = Counter()
    _run(cache, compute, _build("How do I push a model?"))
    assert _run(cache, compute, _build("how do i push a model")).outcome == "hit"
    assert _run(cache, compute, _build("how do i pull a model")).outcome == "miss"


def test_bumping_the_data_generation_misses() -> None:
    cache, backend = _cache()
    compute = Counter()
    _run(cache, compute)
    assert _run(cache, compute).outcome == "hit"
    backend.bump_generation("hf_docs")
    after = _run(cache, compute)
    assert after.outcome == "miss" and compute.calls == 2
    assert _run(cache, compute).outcome == "hit"


@pytest.mark.parametrize(
    ("over", "reason"),
    [({"degraded": True}, "degraded"), ({"rerank_timeout": True}, "rerank_timeout")],
)
def test_an_incomplete_result_is_returned_but_not_stored(over: dict[str, Any], reason: str) -> None:
    cache, backend = _cache()
    compute = Counter(**over)
    first = _run(cache, compute)
    assert first.outcome == "miss" and first.stored is False and first.not_stored_reason == reason
    assert backend.live_keys() == []
    assert _run(cache, compute).outcome == "miss" and compute.calls == 2


def test_a_value_over_the_size_limit_is_not_stored() -> None:
    cache, backend = _cache(max_value_bytes=200)
    run = _run(cache, Counter())
    assert run.stored is False and run.not_stored_reason == "too_large"
    assert backend.live_keys() == []


def test_ttl_and_the_shorter_ttl_of_a_not_found_result() -> None:
    clock = Clock()
    cache, _ = _cache(clock=clock, ttl_s=100, not_found_ttl_s=10)
    found = Counter()
    gone = Counter(pages_result("hf_docs", [1], not_found=True))
    _run(cache, found, _build("a question"))
    _run(cache, gone, _build("a missing thing"))
    clock.now += 11
    assert _run(cache, found, _build("a question")).outcome == "hit"
    assert _run(cache, gone, _build("a missing thing")).outcome == "miss"
    clock.now += 100
    assert _run(cache, found, _build("a question")).outcome == "miss"


def test_a_value_nobody_can_read_is_a_miss_and_is_overwritten() -> None:
    cache, backend = _cache()
    compute = Counter()
    _run(cache, compute)
    (key,) = backend.live_keys()
    backend.set(key, "{broken", ttl_s=100)
    run = _run(cache, compute)
    assert run.outcome == "miss" and run.stored
    assert _run(cache, compute).outcome == "hit"


def test_read_false_computes_and_stores_without_reading() -> None:
    cache, backend = _cache()
    compute = Counter()
    _run(cache, compute)
    gets = backend.calls.get("get", 0)
    run = _run(cache, compute, read=False)
    assert (run.outcome, run.reason) == ("bypass", "read_off")
    assert run.stored and compute.calls == 2
    assert backend.calls.get("get", 0) == gets


# ---- the store fails -------------------------------------------------------------------------


def test_a_store_that_is_down_never_fails_the_request() -> None:
    cache, backend = _cache()
    backend.fail_with = CacheUnavailable("ConnectionError")
    compute = Counter()
    run = _run(cache, compute)
    assert run.outcome == "error" and run.reason == "ConnectionError"
    assert run.result.request_id == "fresh-1" and compute.calls == 1
    assert run.stored is False


def test_a_generation_that_cannot_be_read_is_an_error_not_a_guess() -> None:
    cache, backend = _cache()
    original = backend.generation

    def broken(kb: str) -> int:
        raise CacheUnavailable("TimeoutError")

    backend.generation = broken  # type: ignore[method-assign]
    compute = Counter()
    run = _run(cache, compute)
    assert run.outcome == "error" and compute.calls == 1 and backend.live_keys() == []
    backend.generation = original  # type: ignore[method-assign]
    assert _run(cache, compute).outcome == "miss"


def test_a_write_that_fails_still_returns_the_result() -> None:
    cache, backend = _cache()

    def fail(key: str, value: str, ttl_s: int) -> None:
        raise CacheUnavailable("TimeoutError")

    backend.set = fail  # type: ignore[method-assign]
    run = _run(cache, Counter())
    assert run.outcome == "miss" and run.stored is False and run.not_stored_reason == "TimeoutError"


def test_the_breaker_stops_calls_to_a_dead_store_and_recovers() -> None:
    clock = Clock()
    cache, backend = _cache(clock=clock)
    backend.fail_with = CacheUnavailable("ConnectionError")
    compute = Counter()
    outcomes = [_run(cache, compute).outcome for _ in range(3)]
    assert outcomes == ["error", "error", "error"]
    touched = sum(backend.calls.values())
    run = _run(cache, compute)
    assert run.outcome == "bypass" and run.reason == "breaker_open"
    assert sum(backend.calls.values()) == touched  # no call reached the store
    assert compute.calls == 4
    backend.fail_with = None
    clock.now += 5.1
    assert _run(cache, compute).outcome == "miss"  # the probe works: closed again
    assert _run(cache, compute).outcome == "hit"


def test_a_closed_port_costs_little_and_never_raises() -> None:
    """Spec: Redis down is simulated by pointing the client at a port nobody listens on."""
    client = make_redis_client("redis://127.0.0.1:1/0", socket_timeout=0.05)
    backend = RedisCacheBackend(client, key_prefix="p501test:")
    cache = ExactCache(backend, CacheConfig())
    compute = Counter()
    began = time.perf_counter()
    runs = [_run(cache, compute) for _ in range(6)]
    elapsed = time.perf_counter() - began
    assert all(r.outcome in {"error", "bypass"} for r in runs)
    assert compute.calls == 6 and elapsed < 5
    assert runs[-1].outcome == "bypass"  # the breaker opened after the failures
    client.close()


def test_the_lookup_reports_its_own_time() -> None:
    cache, _ = _cache()
    run = _run(cache, Counter())
    assert run.lookup_ms >= 0 and run.elapsed_ms >= run.lookup_ms


def test_verdict_function_reason_is_reported() -> None:
    codec = Codec(
        to_payload=search_payload,
        verdict=lambda r: Verdict(False, "my_reason"),
        not_found=lambda r: False,
    )
    cache, _ = _cache()
    run = cache.get_or_compute(_build(), Counter(), codec=codec, decode=_decode)
    assert run.not_stored_reason == "my_reason"
