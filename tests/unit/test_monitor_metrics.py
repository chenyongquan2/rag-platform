"""Spec P5-02: metric labels the alert rules rely on."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag.cache.exact import CacheRun
from rag.cache.layer import ResultCache
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.testing import memory_observability, metric_points


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _cache_points(obs: object) -> set[tuple[str, str]]:
    return {
        (p[0]["rag.cache.outcome"], p[0].get("rag.cache.reason", ""))
        for p in metric_points(obs, attr.METRIC_CACHE)  # type: ignore[arg-type]
    }


def test_a_bypass_because_the_breaker_is_open_carries_its_reason() -> None:
    """The `CacheBreakerOpen` alert counts bypasses with reason `breaker_open`."""
    with memory_observability() as obs:
        ResultCache.observe(
            None,
            kb="hf_docs",
            tool="search_docs",
            run=CacheRun("r", "bypass", reason="breaker_open"),
        )
        ResultCache.observe(
            None, kb="hf_docs", tool="search_docs", run=CacheRun("r", "bypass", reason="read_off")
        )
        ResultCache.observe(
            None, kb="hf_docs", tool="search_docs", run=CacheRun("r", "error", reason="Timeout")
        )
        assert _cache_points(obs) == {
            ("bypass", "breaker_open"),
            ("bypass", "read_off"),
            ("error", "Timeout"),
        }


def test_hits_and_misses_have_no_reason_label() -> None:
    """A miss that was not stored already has its own counter; the label set of hit and miss
    stays what P5-01 made it."""
    with memory_observability() as obs:
        ResultCache.observe(
            None,
            kb="hf_docs",
            tool="search_docs",
            run=CacheRun("r", "miss", not_stored_reason="degraded"),
        )
        ResultCache.observe(None, kb="hf_docs", tool="search_docs", run=CacheRun("r", "hit"))
        assert _cache_points(obs) == {("miss", ""), ("hit", "")}
