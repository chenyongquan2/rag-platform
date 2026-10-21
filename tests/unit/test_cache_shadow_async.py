"""Spec P5-01 review round 1 (item 5): the semantic cache runs beside the request, not inside it:
a slow store does not slow a request, a full queue drops work, and a store that fails is tried
less and less and logged once, without a traceback."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.semantic import MemorySemanticStore, ShadowRecorder
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import attributes as attr
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import memory_observability, metric_points, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

CREDS = Credentials(transport="http")


class VectorPipeline(FakeSearchPipeline):
    def query_vector(self, question: str) -> list[float] | None:
        return [1.0, 0.0, 0.0, 0.0]


def _service(recorder: ShadowRecorder, **layer: Any) -> tuple[SearchService, ResultCache]:
    settings = observe_settings()
    cache = ResultCache(
        ExactCache(MemoryCacheBackend(), CacheConfig()), settings, shadow=recorder, **layer
    )
    service = SearchService(
        settings,
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": VectorPipeline("hf_docs")}),
        cache=cache,
    )
    service.load()
    return service, cache


def _search(service: SearchService, n: int) -> Any:
    return service.search_docs(CREDS, kb="hf_docs", query=f"question number {n} about models")


class SlowStore(MemorySemanticStore):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.searches = 0

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.searches += 1
        time.sleep(self.seconds)
        return super().search(*args, **kwargs)


class FailingStore(MemorySemanticStore):
    def __init__(self) -> None:
        super().__init__()
        self.searches = 0

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.searches += 1
        raise ConnectionError("elasticsearch is down")


def test_a_slow_store_does_not_slow_the_request() -> None:
    store = SlowStore(0.4)
    service, cache = _service(ShadowRecorder(store, None, threshold=0.2, ttl_s=60))
    began = time.perf_counter()
    response = _search(service, 1)
    elapsed = time.perf_counter() - began
    assert response.chunks and elapsed < 0.3  # not 0.4 s plus
    assert cache.flush_shadow(5.0)  # the work still happens, beside the request
    assert store.searches == 1


def test_a_full_queue_drops_work_instead_of_waiting() -> None:
    release = threading.Event()

    class Blocked(MemorySemanticStore):
        def __init__(self) -> None:
            super().__init__()
            self.searches = 0

        def search(self, *args: Any, **kwargs: Any) -> Any:
            self.searches += 1
            release.wait(10)
            return super().search(*args, **kwargs)

    store = Blocked()
    with memory_observability() as obs:
        service, cache = _service(
            ShadowRecorder(store, None, threshold=0.2, ttl_s=60), shadow_max_pending=2
        )
        began = time.perf_counter()
        for n in range(8):
            _search(service, n)
        assert time.perf_counter() - began < 2.0  # nobody waited for the store
        release.set()
        assert cache.flush_shadow(5.0)
        assert store.searches == 2  # the two that fit; the other six were dropped
        points = metric_points(obs, attr.METRIC_CACHE_SHADOW)
        dropped = sum(p[1].value for p in points if p[0].get(attr.CACHE_SHADOW) == "dropped")
        assert dropped == 6


def test_a_failing_store_is_tried_less_and_logged_once_without_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FailingStore()
    service, cache = _service(ShadowRecorder(store, None, threshold=0.2, ttl_s=60))
    with caplog.at_level(logging.DEBUG, logger="rag.cache"):
        for n in range(30):
            response = _search(service, n)
            assert response.chunks and response.cache.outcome == "miss"
            cache.flush_shadow(5.0)
    assert store.searches <= 6  # five failures open the breaker; nothing is sent after that
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None
    assert "ConnectionError" in warnings[0].getMessage()


def test_the_result_is_still_the_same_with_the_queue_in_use(tmp_path: Any) -> None:
    store = SlowStore(0.05)
    service, cache = _service(ShadowRecorder(store, None, threshold=0.2, ttl_s=60))
    plain = SearchService(
        observe_settings(),
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": VectorPipeline("hf_docs")}),
    )
    plain.load()
    for n in range(5):
        a = _search(service, n).model_dump(mode="json")
        b = _search(plain, n).model_dump(mode="json")
        for name in ("request_id", "timings_ms", "cache"):
            a.pop(name)
            b.pop(name)
        assert a == b
    cache.flush_shadow(5.0)
