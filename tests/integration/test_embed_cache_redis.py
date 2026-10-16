"""Spec P0-05 AC17: the L0 query cache against the compose Redis (`redis_url`).

Keys carry a prefix unique to this run and are deleted at the end; other keys are not touched.
"""

from __future__ import annotations

import uuid

import pytest
from redis import Redis
from redis.exceptions import RedisError

from rag.config import ServiceSettings
from rag.config.settings import mask_url
from rag.embed import RedisEmbeddingCache

pytestmark = pytest.mark.integration


def test_redis_cache_round_trip() -> None:
    url = ServiceSettings().redis_url
    client = Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
    try:
        client.ping()
    except RedisError as exc:
        pytest.skip(f"Redis not reachable at {mask_url(url)} ({exc}); start docker compose")
    prefix = f"test:{uuid.uuid4().hex[:8]}:l0"
    cache = RedisEmbeddingCache(client, ttl_s=60, prefix=prefix)
    vector = [0.5, -0.25, 1.0, 0.0]
    try:
        cache.put_many("m@4", {"h1": vector})
        assert cache.get_many("m@4", ["h1", "h2"]) == {"h1": vector}
        assert 0 < client.ttl(f"{prefix}:m@4:h1") <= 60
    finally:
        keys = list(client.scan_iter(match=f"{prefix}:*"))
        if keys:
            client.delete(*keys)
        client.close()
