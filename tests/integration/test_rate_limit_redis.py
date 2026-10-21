"""Spec P8-01 AC4 against the compose Redis: the counters, their expiry, and Redis being down.

Keys live under `test:<run>:rl:` and are deleted at the end.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from redis import Redis

from rag.serve.limits import RateLimiter, RedisWindowCounter, make_rate_limiter
from rag.testing import observe_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def prefix(redis_client: Redis, redis_key_prefix: str) -> Iterator[str]:
    value = f"{redis_key_prefix}rl:"
    yield value
    keys = list(redis_client.scan_iter(match=f"{value}*"))
    if keys:
        redis_client.delete(*keys)


def test_counts_per_caller_in_redis_and_the_keys_expire(redis_client: Redis, prefix: str) -> None:
    limiter = RateLimiter(
        RedisWindowCounter(redis_client, prefix=prefix),
        per_minute=3,
        per_caller={},
        clock=lambda: 6030.0,  # minute 100, 30 s in
    )
    assert [limiter.admit("alice").allowed for _ in range(4)] == [True, True, True, False]
    assert limiter.admit("bob").allowed
    keys = sorted(redis_client.scan_iter(match=f"{prefix}*"))
    assert keys == [f"{prefix}alice:100", f"{prefix}bob:100"]
    assert redis_client.get(f"{prefix}alice:100") == "4"
    assert all(0 < redis_client.ttl(k) <= 120 for k in keys)


def test_the_settings_build_a_limiter_on_the_compose_redis(prefix: str) -> None:
    settings = observe_settings(rate_limit_mode="redis", rate_limit_key_prefix=prefix)
    limiter = make_rate_limiter(settings)
    assert limiter is not None
    admission = limiter.admit("carol")
    assert admission.allowed and not admission.counter_error and admission.count == 1


def test_redis_down_lets_requests_through() -> None:
    settings = observe_settings(
        rate_limit_mode="redis",
        redis_url="redis://127.0.0.1:9/0",  # nothing listens there
        rate_limit_per_minute=1,
    )
    limiter = make_rate_limiter(settings)
    assert limiter is not None
    admissions = [limiter.admit("alice") for _ in range(3)]
    assert all(a.allowed and a.counter_error for a in admissions)
