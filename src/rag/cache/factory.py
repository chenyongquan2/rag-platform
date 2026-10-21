"""Build the result cache from the settings and a connection string (design s16-7; spec P5-01).

    cache = make_result_cache(settings)      # None when `cache_mode` is `off`

The store is the cache Redis (`cache_url`, empty: `redis_url`) through the factory's client with
the short `cache_timeout_ms` for connecting and for every command: a Redis that is down or slow
is a bypass, not a failed request. The client does not connect until the first command, so a
service starts whether or not Redis is up.
"""

from __future__ import annotations

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.store import RedisCacheBackend
from rag.config import ServiceSettings
from rag.factories import make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX


def cache_config(settings: ServiceSettings) -> CacheConfig:
    return CacheConfig(
        ttl_s=settings.cache_ttl_s,
        not_found_ttl_s=settings.cache_not_found_ttl_s,
        max_value_bytes=settings.cache_max_value_bytes,
        lock_ttl_ms=settings.cache_lock_ttl_ms,
        lock_wait_ms=settings.cache_lock_wait_ms,
        lock_poll_ms=settings.cache_lock_poll_ms,
        breaker_failures=settings.cache_breaker_failures,
        breaker_open_s=settings.cache_breaker_open_s,
    )


def make_cache_backend(settings: ServiceSettings) -> RedisCacheBackend:
    client = make_redis_client(
        settings.cache_url or settings.redis_url, socket_timeout=settings.cache_timeout_ms / 1000.0
    )
    return RedisCacheBackend(
        client,
        key_prefix=settings.cache_key_prefix,
        # The counter ingestion bumps lives under the same prefix the search chains read it with.
        generation_key_prefix=f"{DEFAULT_KEY_PREFIX}{settings.search_index_prefix}",
    )


def make_result_cache(settings: ServiceSettings) -> ResultCache | None:
    """The L1 cache (and the shadow recorder when `cache_mode` is `l1_shadow`), or None."""
    if settings.cache_mode == "off":
        return None
    exact = ExactCache(make_cache_backend(settings), cache_config(settings))
    shadow = None
    if settings.cache_mode == "l1_shadow":
        from rag.cache.semantic import make_shadow_recorder

        shadow = make_shadow_recorder(settings)
    return ResultCache(exact, settings, shadow=shadow)
