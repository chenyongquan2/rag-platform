"""Spec P5-02, review round 1 issue 4: the service recognises the probe's requests (request id
`probe-<hex>`). They do not fill the caches (no L0 query-embedding write, no L1 write, no L2
shadow row) and their metrics carry `rag.request.probe=1`, so user-facing panels can leave them
out."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.semantic import MemorySemanticStore, ShadowRecorder
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.embed.cache import CachedEmbedding, InMemoryEmbeddingCache
from rag.observe import attributes as attr
from rag.observe import bind_request, shutdown_observability
from rag.observe.context import PROBE_REQUEST_PREFIX, current_request_is_probe, is_probe_request
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import FakeEmbedding, memory_observability, metric_points, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

CREDS = Credentials(transport="http")
PROBE_ID = PROBE_REQUEST_PREFIX + "0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class VectorPipeline(FakeSearchPipeline):
    def query_vector(self, question: str) -> list[float] | None:
        return [1.0, 0.0, 0.0, 0.0]


class CountingStore(MemorySemanticStore):
    def __init__(self) -> None:
        super().__init__()
        self.searches = 0

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.searches += 1
        return super().search(*args, **kwargs)


def _service() -> tuple[SearchService, ResultCache, CountingStore]:
    settings = observe_settings()
    store = CountingStore()
    cache = ResultCache(
        ExactCache(MemoryCacheBackend(), CacheConfig()),
        settings,
        shadow=ShadowRecorder(store, None, threshold=0.2, ttl_s=60),
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
    return service, cache, store


def test_a_probe_request_id_is_recognised() -> None:
    assert (
        is_probe_request(PROBE_ID)
        and not is_probe_request("0123abcd")
        and not is_probe_request(None)
    )
    assert not current_request_is_probe()
    with bind_request(PROBE_ID):
        assert current_request_is_probe()
    with bind_request("user-request"):
        assert not current_request_is_probe()


def test_a_probe_search_neither_reads_nor_writes_the_result_cache() -> None:
    service, cache, store = _service()
    query = "how to push a model to the hub"
    probe = service.search_docs(CREDS, kb="hf_docs", query=query, request_id=PROBE_ID)
    assert probe.chunks
    assert probe.cache is not None
    assert (probe.cache.outcome, probe.cache.hit) == ("bypass", False)
    assert cache.flush_shadow(5.0)
    assert store.searches == 0  # no L2 shadow row
    # Nothing was written: the same question from a user is a miss, and that one is stored.
    user = service.search_docs(CREDS, kb="hf_docs", query=query, request_id="u1")
    assert user.cache is not None and user.cache.outcome == "miss"
    assert cache.flush_shadow(5.0) and store.searches == 1
    again = service.search_docs(CREDS, kb="hf_docs", query=query, request_id="u2")
    assert again.cache is not None and again.cache.outcome == "hit"


def test_probe_metrics_carry_the_probe_label_and_user_metrics_do_not() -> None:
    with memory_observability() as obs:
        service, _, _ = _service()
        service.search_docs(CREDS, kb="hf_docs", query="probe question", request_id=PROBE_ID)
        service.search_docs(CREDS, kb="hf_docs", query="user question", request_id="u1")
        requests = metric_points(obs, attr.METRIC_REQUESTS)
        cache = metric_points(obs, attr.METRIC_CACHE)
        steps = metric_points(obs, attr.METRIC_STEP_DURATION)
    probe_requests = [p for a, p in requests if a.get(attr.REQUEST_PROBE) == "1"]
    user_requests = [p for a, p in requests if attr.REQUEST_PROBE not in a]
    assert [p.value for p in probe_requests] == [1] and [p.value for p in user_requests] == [1]
    assert any(
        a.get(attr.REQUEST_PROBE) == "1" and a.get(attr.CACHE_REASON) == "probe" for a, _ in cache
    )
    assert any(a.get(attr.REQUEST_PROBE) == "1" for a, _ in steps)
    assert any(attr.REQUEST_PROBE not in a for a, _ in steps)


def test_a_probe_query_embedding_is_not_written_to_the_embedding_cache() -> None:
    """Each probe run asks a new question: written, every run would leave a 5 KB vector in Redis
    for 30 days (about 227 MB at one run a minute, next to a 256 MB Redis)."""
    store = InMemoryEmbeddingCache()
    embed = CachedEmbedding(FakeEmbedding(dim=8), store, cache_model="fake@8")
    with bind_request(PROBE_ID):
        vector = embed.get_query_embedding("probe question zq1a2b3c")
    assert len(vector) == 8
    assert embed.cached_query_embedding("probe question zq1a2b3c") is None
    with bind_request("user-request"):
        embed.get_query_embedding("user question")
    assert embed.cached_query_embedding("user question") is not None
