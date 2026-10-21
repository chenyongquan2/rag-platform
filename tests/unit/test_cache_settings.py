"""Spec P5-01: the cache settings are explicit, checked, and build the layer from connection
strings (design s16-7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.cache.factory import cache_config, make_cache_backend, make_result_cache
from rag.cache.semantic import EsSemanticStore, MemorySemanticStore, ShadowRecorder
from rag.cache.store import RedisCacheBackend
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.serve import FakePipelineBuilder


def test_defaults_are_explicit() -> None:
    s = observe_settings()
    assert s.cache_mode == "l1_shadow" and s.cache_url == "" and s.cache_key_prefix == "rag:l1:"
    assert s.cache_timeout_ms == 50 and s.cache_ttl_s == 3600 and s.cache_not_found_ttl_s == 600
    assert s.cache_max_value_bytes == 262_144
    assert (s.cache_breaker_failures, s.cache_breaker_open_s) == (5, 5.0)
    assert (s.cache_lock_ttl_ms, s.cache_lock_wait_ms, s.cache_lock_poll_ms) == (30_000, 3_000, 25)
    assert s.cache_semantic_store == "es" and s.cache_semantic_lexical_guard is True
    assert 0 < s.cache_semantic_threshold <= 0.5


@pytest.mark.parametrize(
    "bad",
    [
        {"cache_mode": "on"},
        {"cache_timeout_ms": 0},
        {"cache_ttl_s": 0},
        {"cache_max_value_bytes": 0},
        {"cache_breaker_failures": 0},
        {"cache_breaker_open_s": 0},
        {"cache_semantic_threshold": 1.5},
        {"cache_semantic_store": "redis"},
    ],
)
def test_bad_values_are_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        observe_settings(**bad)


def test_the_config_follows_the_settings() -> None:
    cfg = cache_config(
        observe_settings(cache_ttl_s=7, cache_lock_wait_ms=9, cache_breaker_failures=2)
    )
    assert (cfg.ttl_s, cfg.lock_wait_ms, cfg.breaker_failures) == (7, 9, 2)


def test_the_backend_comes_from_the_connection_string_with_a_short_timeout() -> None:
    backend = make_cache_backend(
        observe_settings(
            cache_url="redis://127.0.0.1:1/3", cache_timeout_ms=40, cache_key_prefix="x:"
        )
    )
    assert isinstance(backend, RedisCacheBackend) and backend.key_prefix == "x:"
    kwargs = backend.client.connection_pool.connection_kwargs
    assert kwargs["db"] == 3 and kwargs["port"] == 1
    assert kwargs["socket_timeout"] == pytest.approx(0.04)
    assert kwargs["socket_connect_timeout"] == pytest.approx(0.04)
    backend.client.close()


def test_an_empty_cache_url_uses_the_redis_url() -> None:
    backend = make_cache_backend(observe_settings(redis_url="redis://127.0.0.1:2/5"))
    assert backend.client.connection_pool.connection_kwargs["port"] == 2
    backend.client.close()


def test_mode_off_makes_no_layer_and_the_modes_make_the_expected_one(tmp_path: object) -> None:
    assert make_result_cache(observe_settings(cache_mode="off")) is None
    l1 = make_result_cache(observe_settings(cache_mode="l1"))
    assert l1 is not None and l1.shadow is None
    l1.close()
    shadow = make_result_cache(
        observe_settings(cache_mode="l1_shadow", cache_semantic_store="memory", cache_shadow_dir="")
    )
    assert shadow is not None and isinstance(shadow.shadow, ShadowRecorder)
    assert isinstance(shadow.shadow.store, MemorySemanticStore)
    assert shadow.shadow.writer is None
    shadow.close()
    es = make_result_cache(observe_settings(cache_mode="l1_shadow", cache_semantic_store="es"))
    assert es is not None and es.shadow is not None
    assert isinstance(es.shadow.store, EsSemanticStore)
    es.close()


def test_the_bare_service_has_no_cache_and_from_settings_builds_one() -> None:
    bare = SearchService(observe_settings(), configs={}, builder=FakePipelineBuilder({}))
    assert bare.cache is None
    built = SearchService.from_settings(
        observe_settings(cache_mode="l1"), configs={}, builder=FakePipelineBuilder({})
    )
    assert built.cache is not None
    built.close()
    off = SearchService.from_settings(
        observe_settings(cache_mode="off"), configs={}, builder=FakePipelineBuilder({})
    )
    assert off.cache is None
