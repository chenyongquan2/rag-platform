"""Spec P5-01 AC11: the exact cache against the compose Redis.

Every key carries the prefix unique to this run (`redis_key_prefix`) and is deleted at the end by
the fixture; the data generation counters used here are under the same prefix, never the real
ones of the services.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from redis import Redis

from rag.cache.exact import CacheConfig, Codec, ExactCache
from rag.cache.keys import make_fields
from rag.cache.layer import ResultCache
from rag.cache.rules import Entry, search_cacheable, search_from_payload, search_payload
from rag.cache.store import RedisCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.factories import make_redis_client
from rag.index.generation import incr_data_generation
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.agent import pages_result
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline
from rag.types import RetrievalResult

pytestmark = pytest.mark.integration


@pytest.fixture
def backend(redis_client: Redis, redis_key_prefix: str) -> RedisCacheBackend:
    return RedisCacheBackend(
        redis_client,
        key_prefix=f"{redis_key_prefix}l1:{uuid.uuid4().hex[:6]}:",
        generation_key_prefix=f"{redis_key_prefix}data_generation:",
    )


def _codec() -> Codec[RetrievalResult]:
    return Codec(
        to_payload=search_payload, verdict=search_cacheable, not_found=lambda r: r.not_found
    )


def _decode(entry: Entry) -> RetrievalResult:
    return search_from_payload(entry.payload, request_id="hit", data_generation=0, timings_ms={})


def _build(question: str = "how do i push a model") -> Any:
    return lambda generation: make_fields(
        "search", kb="hf_docs", question=question, top_k=3, data_generation=generation
    )


def test_set_get_ttl_nx_and_release(backend: RedisCacheBackend, redis_client: Redis) -> None:
    backend.set("hf_docs:search:a", "value", ttl_s=60)
    assert backend.get("hf_docs:search:a") == "value" and backend.get("nope") is None
    assert 0 < redis_client.ttl(backend.key_prefix + "hf_docs:search:a") <= 60
    assert backend.set_nx("hf_docs:search:a:lock", "one", ttl_ms=5000) is True
    assert backend.set_nx("hf_docs:search:a:lock", "two", ttl_ms=5000) is False
    backend.release("hf_docs:search:a:lock", "two")  # not the owner
    assert backend.get("hf_docs:search:a:lock") == "one"
    backend.release("hf_docs:search:a:lock", "one")
    assert backend.get("hf_docs:search:a:lock") is None
    assert backend.set_nx("hf_docs:search:a:lock", "three", ttl_ms=50) is True
    time.sleep(0.12)
    assert backend.get("hf_docs:search:a:lock") is None  # a crashed owner's lock expires


def test_delete_prefix_scans_a_larger_set_and_leaves_other_keys(
    backend: RedisCacheBackend,
) -> None:
    for i in range(700):
        backend.set(f"hf_docs:search:{i}", "v", ttl_s=60)
    backend.set("mt5:search:1", "v", ttl_s=60)
    assert backend.delete_prefix("hf_docs:", batch=100) == 700
    assert backend.get("hf_docs:search:5") is None and backend.get("mt5:search:1") == "v"
    assert backend.delete_prefix("*") == 0  # a glob character is not a wildcard
    assert backend.get("mt5:search:1") == "v"


def test_generation_comes_from_the_counter_ingestion_bumps(
    backend: RedisCacheBackend, redis_client: Redis, redis_key_prefix: str
) -> None:
    first = backend.generation("hf_docs")
    assert backend.generation("hf_docs") == first
    incr_data_generation(redis_client, "hf_docs", key_prefix=f"{redis_key_prefix}data_generation:")
    assert backend.generation("hf_docs") == first + 1


def test_miss_hit_and_a_new_generation_misses(
    backend: RedisCacheBackend, redis_client: Redis, redis_key_prefix: str
) -> None:
    cache = ExactCache(backend, CacheConfig(ttl_s=30))
    calls = []

    def compute() -> RetrievalResult:
        calls.append(1)
        return pages_result("hf_docs", [1, 2, 3])

    run = cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)
    again = cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)
    assert (run.outcome, again.outcome) == ("miss", "hit") and len(calls) == 1
    assert again.age_s is not None and again.age_s < 5
    incr_data_generation(redis_client, "hf_docs", key_prefix=f"{redis_key_prefix}data_generation:")
    assert cache.get_or_compute(_build(), compute, codec=_codec(), decode=_decode).outcome == "miss"
    # the stored value is JSON with created_at and the key fields, and it expires
    (key,) = [k for k in redis_client.scan_iter(match=f"{backend.key_prefix}hf_docs:search:*")][:1]
    assert 0 < redis_client.ttl(key) <= 30
    import json

    value = json.loads(redis_client.get(key))
    assert value["key_fields"]["kb"] == "hf_docs" and value["created_at"] > 0


def test_two_processes_on_real_redis_compute_once(backend: RedisCacheBackend) -> None:
    one = ExactCache(backend, CacheConfig(lock_poll_ms=10))
    two = ExactCache(backend, CacheConfig(lock_poll_ms=10))
    calls: list[int] = []
    barrier = threading.Barrier(2)

    def compute() -> RetrievalResult:
        calls.append(1)
        time.sleep(0.4)
        return pages_result("hf_docs", [1, 2, 3])

    def go(cache: ExactCache) -> str:
        barrier.wait()
        return cache.get_or_compute(
            _build("two process question"), compute, codec=_codec(), decode=_decode
        ).outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(go, [one, two]))
    assert len(calls) == 1 and outcomes in (["miss", "shared"], ["hit", "miss"])


def test_a_closed_port_never_fails_a_request_and_costs_little(redis_key_prefix: str) -> None:
    client = make_redis_client("redis://127.0.0.1:1/0", socket_timeout=0.05)
    cache = ExactCache(RedisCacheBackend(client, key_prefix=redis_key_prefix), CacheConfig())
    began = time.perf_counter()
    runs = [
        cache.get_or_compute(
            _build(), lambda: pages_result("hf_docs", [1]), codec=_codec(), decode=_decode
        )
        for _ in range(8)
    ]
    assert time.perf_counter() - began < 5
    assert all(r.result.chunks for r in runs)
    assert runs[0].outcome == "error" and runs[-1].outcome == "bypass"
    client.close()


def test_a_slow_redis_is_a_bypass_not_a_failure(redis_key_prefix: str) -> None:
    """A client whose time-out is shorter than the server's answer: simulated with a time-out so
    small that connecting cannot finish."""
    client = make_redis_client("redis://10.255.255.1:6379/0", socket_timeout=0.02)
    cache = ExactCache(RedisCacheBackend(client, key_prefix=redis_key_prefix), CacheConfig())
    began = time.perf_counter()
    run = cache.get_or_compute(
        _build(), lambda: pages_result("hf_docs", [1]), codec=_codec(), decode=_decode
    )
    assert run.outcome == "error" and run.result.chunks
    assert time.perf_counter() - began < 1.0
    client.close()


def test_the_service_with_real_redis_serves_a_repeat_from_the_cache(
    redis_client: Redis, redis_key_prefix: str, service_settings: Any
) -> None:
    settings = observe_settings(
        cache_key_prefix=f"{redis_key_prefix}svc:",
        redis_url=service_settings.redis_url,
        search_index_prefix=f"{redis_key_prefix.replace(':', '_')}",
    )
    backend = RedisCacheBackend(
        redis_client,
        key_prefix=settings.cache_key_prefix,
        generation_key_prefix=f"{redis_key_prefix}data_generation:",
    )
    pipeline = FakeSearchPipeline("hf_docs")
    service = SearchService(
        settings,
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": pipeline}),
        cache=ResultCache(ExactCache(backend, CacheConfig(ttl_s=30)), settings),
    )
    service.load()
    creds = Credentials(transport="http")
    first = service.search_docs(creds, kb="hf_docs", query="how do i push a model")
    second = service.search_docs(creds, kb="hf_docs", query="How do I push a model?")
    assert (first.cache.outcome, second.cache.outcome) == ("miss", "hit")
    assert len(pipeline.calls) == 1 and second.chunks == first.chunks
    assert service.cache is not None and service.cache.drop_kb("hf_docs") >= 1
    assert service.search_docs(
        creds, kb="hf_docs", query="how do i push a model"
    ).cache.outcome == ("miss")
