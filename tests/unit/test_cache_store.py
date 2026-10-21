"""Spec P5-01 AC2: cache backends (memory, Redis-shaped) and the circuit breaker."""

from __future__ import annotations

import fnmatch
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from rag.cache.store import (
    CacheUnavailable,
    CircuitBreaker,
    MemoryCacheBackend,
    NullCacheBackend,
    RedisCacheBackend,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeRedis:
    """The few commands the backend uses; `keys` and `flushdb` fail the test."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, float] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.raises: Exception | None = None

    def _note(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if self.raises is not None:
            raise self.raises

    def get(self, key: str) -> str | None:
        self._note("get", key)
        return self.data.get(key)

    def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._note("set", key, value, ex=ex, px=px, nx=nx)
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def eval(self, script: str, numkeys: int, *args: Any) -> int:
        self._note("eval", script, numkeys, *args)
        key, token = args
        if self.data.get(key) == token:
            del self.data[key]
            return 1
        return 0

    def scan_iter(self, match: str | None = None, count: int | None = None) -> Any:
        self._note("scan_iter", match=match, count=count)
        yield from [k for k in list(self.data) if match is None or fnmatch.fnmatch(k, match)]

    def unlink(self, *keys: str) -> int:
        self._note("unlink", *keys)
        for key in keys:
            self.data.pop(key, None)
        return len(keys)

    def keys(self, *args: Any) -> list[str]:
        raise AssertionError("KEYS must never be used on the cache Redis")

    def flushdb(self) -> None:
        raise AssertionError("FLUSHDB must never be used on the cache Redis")

    def flushall(self) -> None:
        raise AssertionError("FLUSHALL must never be used on the cache Redis")


def test_memory_backend_ttl_with_a_fake_clock() -> None:
    clock = Clock()
    backend = MemoryCacheBackend(clock=clock)
    backend.set("k", "v", ttl_s=10)
    assert backend.get("k") == "v"
    clock.now += 9.9
    assert backend.get("k") == "v"
    clock.now += 0.2
    assert backend.get("k") is None
    assert backend.get("never") is None


def test_memory_backend_set_nx_and_release_with_the_owner_token() -> None:
    clock = Clock()
    backend = MemoryCacheBackend(clock=clock)
    assert backend.set_nx("lock", "a", ttl_ms=1000) is True
    assert backend.set_nx("lock", "b", ttl_ms=1000) is False
    backend.release("lock", "b")  # not the owner: the lock stays
    assert backend.set_nx("lock", "c", ttl_ms=1000) is False
    backend.release("lock", "a")
    assert backend.set_nx("lock", "c", ttl_ms=1000) is True
    clock.now += 1.1  # the lock of a crashed owner expires
    assert backend.set_nx("lock", "d", ttl_ms=1000) is True


def test_memory_backend_delete_prefix_and_generation() -> None:
    backend = MemoryCacheBackend()
    for key in ("hf:search:1", "hf:ask:2", "other:search:3"):
        backend.set(key, "v", ttl_s=60)
    assert backend.delete_prefix("hf:") == 2
    assert backend.get("other:search:3") == "v" and backend.get("hf:search:1") is None
    first = backend.generation("hf")
    assert backend.generation("hf") == first
    assert backend.bump_generation("hf") == first + 1 == backend.generation("hf")


def test_memory_backend_can_be_made_to_fail_and_to_be_slow() -> None:
    backend = MemoryCacheBackend()
    backend.fail_with = CacheUnavailable("down")
    for call in (
        lambda: backend.get("k"),
        lambda: backend.set("k", "v", ttl_s=1),
        lambda: backend.set_nx("k", "t", ttl_ms=1),
        lambda: backend.generation("kb"),
    ):
        with pytest.raises(CacheUnavailable):
            call()


def test_null_backend_is_always_unavailable() -> None:
    backend = NullCacheBackend()
    with pytest.raises(CacheUnavailable):
        backend.get("k")
    with pytest.raises(CacheUnavailable):
        backend.generation("kb")


def test_redis_backend_uses_expiry_nx_and_scan() -> None:
    redis = FakeRedis()
    backend = RedisCacheBackend(redis, key_prefix="p501test:", generation=lambda kb: 5)  # type: ignore[arg-type]
    backend.set("hf:search:1", "v", ttl_s=30)
    assert redis.data == {"p501test:hf:search:1": "v"}
    assert redis.calls[-1][2]["ex"] == 30
    assert backend.get("hf:search:1") == "v"
    assert backend.set_nx("hf:lock:1", "tok", ttl_ms=2500) is True
    assert redis.calls[-1][2]["px"] == 2500 and redis.calls[-1][2]["nx"] is True
    assert backend.set_nx("hf:lock:1", "tok2", ttl_ms=2500) is False
    backend.release("hf:lock:1", "wrong")
    assert "p501test:hf:lock:1" in redis.data
    backend.release("hf:lock:1", "tok")
    assert "p501test:hf:lock:1" not in redis.data
    assert backend.generation("hf") == 5


def test_redis_backend_drops_a_prefix_with_scan_and_unlink_only() -> None:
    redis = FakeRedis()
    backend = RedisCacheBackend(redis, key_prefix="p501test:", generation=lambda kb: 1)  # type: ignore[arg-type]
    for i in range(1200):
        backend.set(f"hf:search:{i}", "v", ttl_s=60)
    backend.set("other:search:1", "v", ttl_s=60)
    dropped = backend.delete_prefix("hf:", batch=500)
    assert dropped == 1200
    assert list(redis.data) == ["p501test:other:search:1"]
    names = [c[0] for c in redis.calls]
    assert "scan_iter" in names and "unlink" in names
    scan = next(c for c in redis.calls if c[0] == "scan_iter")
    assert scan[2]["match"] == "p501test:hf:*"
    assert max(len(c[1]) for c in redis.calls if c[0] == "unlink") <= 500


def test_a_prefix_cannot_widen_the_drop_with_glob_characters() -> None:
    redis = FakeRedis()
    backend = RedisCacheBackend(redis, key_prefix="p:", generation=lambda kb: 1)  # type: ignore[arg-type]
    backend.set("a:search:1", "v", ttl_s=60)
    assert backend.delete_prefix("*") == 0  # `*` is not a wildcard here
    assert backend.delete_prefix("a[") == 0
    assert list(redis.data) == ["p:a:search:1"]


@pytest.mark.parametrize(
    "error", [RedisConnectionError("refused"), RedisTimeoutError("slow"), OSError("x")]
)
def test_redis_errors_become_cache_unavailable(error: Exception) -> None:
    redis = FakeRedis()
    redis.raises = error
    backend = RedisCacheBackend(redis, key_prefix="p:", generation=lambda kb: 1)  # type: ignore[arg-type]
    for call in (
        lambda: backend.get("k"),
        lambda: backend.set("k", "v", ttl_s=1),
        lambda: backend.set_nx("k", "t", ttl_ms=1),
        lambda: backend.release("k", "t"),
        lambda: backend.delete_prefix("k"),
    ):
        with pytest.raises(CacheUnavailable):
            call()


def test_breaker_opens_after_n_failures_in_a_row_then_probes_once() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failures=3, open_s=5.0, clock=clock)
    assert breaker.state == "closed" and breaker.allow()
    breaker.failure()
    breaker.failure()
    breaker.success()  # a success resets the count
    breaker.failure()
    breaker.failure()
    assert breaker.state == "closed" and breaker.allow()
    breaker.failure()
    assert breaker.state == "open" and not breaker.allow()
    clock.now += 4.9
    assert not breaker.allow()
    clock.now += 0.2
    assert breaker.allow()  # one probe
    assert breaker.state == "half_open"
    assert not breaker.allow()  # the others wait for the probe
    breaker.failure()  # the probe failed: open again for the full time
    assert breaker.state == "open" and not breaker.allow()
    clock.now += 5.1
    assert breaker.allow()
    breaker.success()
    assert breaker.state == "closed" and breaker.allow() and breaker.allow()


def test_breaker_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failures=0, open_s=1.0)
    with pytest.raises(ValueError):
        CircuitBreaker(failures=1, open_s=0)
