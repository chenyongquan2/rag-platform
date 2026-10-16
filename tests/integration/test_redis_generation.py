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


def test_data_generation_counter(redis_client: Redis, redis_key_prefix: str) -> None:
    prefix = f"{redis_key_prefix}data_generation:"
    started_ms = time.time_ns() // 1_000_000

    first = get_data_generation(redis_client, "hf_docs", key_prefix=prefix)
    # A missing counter starts at the current time in ms, so it is above every earlier value.
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
