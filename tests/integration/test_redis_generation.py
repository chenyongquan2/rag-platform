from __future__ import annotations

import time

import pytest
from redis import Redis

from rag.index.generation import get_data_generation, incr_data_generation

pytestmark = pytest.mark.integration


def test_redis_is_an_lru_cache(redis_client: Redis) -> None:
    config = redis_client.config_get("maxmemory*")
    assert config["maxmemory"] == str(256 * 1024 * 1024)
    assert config["maxmemory-policy"] == "allkeys-lru"


def _server_ms(redis_client: Redis) -> int:
    seconds, microseconds = redis_client.time()
    return seconds * 1000 + microseconds // 1000


def test_data_generation_counter(
    redis_client: Redis, redis_key_prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = f"{redis_key_prefix}data_generation:"
    # The seed must come from the Redis server clock, the one clock all processes share;
    # a client whose clock is far behind must not be able to move the counter back.
    monkeypatch.setattr(time, "time_ns", lambda: 0)
    monkeypatch.setattr(time, "time", lambda: 0.0)
    started_ms = _server_ms(redis_client)

    first = get_data_generation(redis_client, "hf_docs", key_prefix=prefix)
    # A missing counter starts at the server time in ms, so it is above every earlier value.
    assert first >= started_ms
    assert get_data_generation(redis_client, "hf_docs", key_prefix=prefix) == first
    assert incr_data_generation(redis_client, "hf_docs", key_prefix=prefix) == first + 1
    assert get_data_generation(redis_client, "hf_docs", key_prefix=prefix) == first + 1

    other = get_data_generation(redis_client, "mt5", key_prefix=prefix)
    assert incr_data_generation(redis_client, "mt5", key_prefix=prefix) == other + 1
    assert get_data_generation(redis_client, "hf_docs", key_prefix=prefix) == first + 1

    # The key is lost (evicted by allkeys-lru, or Redis restarted without persistence):
    # the counter must not go back, or old cache entries would match again.
    time.sleep(0.01)
    redis_client.delete(f"{prefix}hf_docs")
    reseeded = get_data_generation(redis_client, "hf_docs", key_prefix=prefix)
    assert reseeded > first + 1
    time.sleep(0.01)
    redis_client.delete(f"{prefix}hf_docs")
    assert incr_data_generation(redis_client, "hf_docs", key_prefix=prefix) > reseeded + 1

    with pytest.raises(ValueError, match="knowledge base"):
        get_data_generation(redis_client, "bad kb", key_prefix=prefix)
