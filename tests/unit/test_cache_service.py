"""Spec P5-01 AC6: the exact cache in `search_docs` (design s12-1): a hit skips the pipeline, other
filters and other ACL groups never share an entry, a new data generation misses, an incomplete
result is not stored, a broken store never fails a request, and the cache shows in the response,
the span, the metrics and the business record."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.store import CacheUnavailable, MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.security.access import Caller, Credentials
from rag.serve.schema import SearchResponse
from rag.serve.service import SearchService
from rag.testing import memory_observability, metric_points, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline, retrieval_result

CREDS = Credentials(transport="http")


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class ByClient:
    """Callers by client id, each with its own ACL groups."""

    def __init__(self, groups: dict[str, tuple[str, ...]]) -> None:
        self.groups = groups

    def resolve(self, credentials: Credentials) -> Caller:
        client = credentials.client_id or "local"
        return Caller(client, self.groups.get(client, ("public",)))


def _cfg(kb: str = "hf_docs") -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        kb=kb, description=f"{kb} pages", language="en", access=AccessConfig(callers=["*"])
    )


def _layer(
    backend: MemoryCacheBackend | None = None, **settings: Any
) -> tuple[ResultCache, MemoryCacheBackend]:
    backend = backend or MemoryCacheBackend()
    s = observe_settings(**settings)
    return ResultCache(ExactCache(backend, CacheConfig()), s), backend


def _service(
    cache: ResultCache | None,
    *,
    pipeline: FakeSearchPipeline | None = None,
    resolver: Any = None,
    **settings: Any,
) -> tuple[SearchService, FakeSearchPipeline]:
    pipeline = pipeline or FakeSearchPipeline("hf_docs")
    service = SearchService(
        observe_settings(**settings),
        configs={"hf_docs": _cfg()},
        builder=FakePipelineBuilder({"hf_docs": pipeline}),
        cache=cache,
        resolver=resolver,
    )
    service.load()
    return service, pipeline


def _search(service: SearchService, query: str = "How do I push a model?", **kw: Any) -> Any:
    return service.search_docs(CREDS, kb="hf_docs", query=query, **kw)


def test_a_repeated_search_is_served_from_the_cache() -> None:
    cache, backend = _layer()
    service, pipeline = _service(cache)
    first = _search(service, request_id="r1")
    second = _search(service, request_id="r2")
    assert len(pipeline.calls) == 1
    assert (first.cache.outcome, first.cache.hit) == ("miss", False)
    assert (second.cache.outcome, second.cache.hit, second.cache.layer) == ("hit", True, "l1")
    assert second.request_id == "r2" and first.request_id == "r1"
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]
    assert second.data_generation == backend.generation("hf_docs")
    assert second.config_version == first.config_version
    assert second.timings_ms["request"] >= 0


def test_a_service_without_a_cache_layer_answers_as_before() -> None:
    service, pipeline = _service(None)
    one, two = _search(service), _search(service)
    assert len(pipeline.calls) == 2 and one.cache is None and two.cache is None
    assert "cache" in one.model_dump()  # the field exists, it is null


def test_a_user_with_other_acl_groups_never_gets_the_entry_of_another() -> None:
    cache, _ = _layer()
    resolver = ByClient({"alice": ("public",), "bob": ("public", "hr")})
    service, pipeline = _service(cache, resolver=resolver)
    alice = Credentials(transport="http", client_id="alice")
    bob = Credentials(transport="http", client_id="bob")
    service.search_docs(alice, kb="hf_docs", query="salary bands")
    assert len(pipeline.calls) == 1
    again = service.search_docs(bob, kb="hf_docs", query="salary bands")
    assert again.cache.outcome == "miss" and len(pipeline.calls) == 2
    assert pipeline.calls[1]["acl_groups"] == ("hr", "public") or set(
        pipeline.calls[1]["acl_groups"]
    ) == {"public", "hr"}
    # And each group set hits its own entry afterwards.
    assert service.search_docs(alice, kb="hf_docs", query="salary bands").cache.outcome == "hit"
    assert service.search_docs(bob, kb="hf_docs", query="salary bands").cache.outcome == "hit"
    assert len(pipeline.calls) == 2


def test_other_filters_do_not_share_an_entry_and_equal_filters_do() -> None:
    cache, _ = _layer()
    service, pipeline = _service(cache)
    _search(service)
    assert _search(service, filters={"language": "en"}).cache.outcome == "miss"
    assert _search(service, filters={"language": "zh"}).cache.outcome == "miss"
    assert _search(service, filters={"doc_type": ["md", "mdx"]}).cache.outcome == "miss"
    assert _search(service, filters={"doc_type": ["mdx", "md"]}).cache.outcome == "hit"
    assert _search(service, filters={"language": "en"}).cache.outcome == "hit"
    assert len(pipeline.calls) == 4


def test_top_k_is_part_of_the_key() -> None:
    cache, _ = _layer()
    service, pipeline = _service(cache)
    assert _search(service, top_k=2).cache.outcome == "miss"
    assert _search(service, top_k=3).cache.outcome == "miss"
    hit = _search(service, top_k=2)
    assert hit.cache.outcome == "hit" and len(hit.chunks) == 2
    assert len(pipeline.calls) == 2


def test_a_new_data_generation_misses_and_returns_the_new_content() -> None:
    cache, backend = _layer()
    pipeline = FakeSearchPipeline("hf_docs", result=retrieval_result("hf_docs", n=2))
    service, _ = _service(cache, pipeline=pipeline)
    old = _search(service)
    assert _search(service).cache.outcome == "hit"
    backend.bump_generation("hf_docs")  # ingestion wrote and refreshed
    pipeline.result = retrieval_result("hf_docs", n=3)  # the page changed
    fresh = _search(service)
    assert fresh.cache.outcome == "miss" and len(fresh.chunks) == 3 and len(old.chunks) == 2
    again = _search(service)
    assert again.cache.outcome == "hit" and len(again.chunks) == 3
    assert again.data_generation == backend.generation("hf_docs") > old.data_generation


def test_equivalent_questions_share_an_entry_and_different_ones_do_not() -> None:
    cache, _ = _layer()
    service, pipeline = _service(cache)
    _search(service, "How do I push a model?")
    assert _search(service, "  how do i   push a model ").cache.outcome == "hit"
    assert _search(service, "how do i pull a model").cache.outcome == "miss"
    assert _search(service, "What does push_to_hub do?").cache.outcome == "miss"
    assert _search(service, "What does push_to_hub_fastai do?").cache.outcome == "miss"
    assert len(pipeline.calls) == 4


@pytest.mark.parametrize("flag", ["degraded", "rerank_timeout"])
def test_an_incomplete_result_is_returned_but_never_stored(flag: str) -> None:
    cache, backend = _layer()
    result = retrieval_result("hf_docs").model_copy(update={flag: True})
    service, pipeline = _service(cache, pipeline=FakeSearchPipeline("hf_docs", result=result))
    one, two = _search(service), _search(service)
    assert getattr(one, flag) is True and one.cache.outcome == "miss"
    assert two.cache.outcome == "miss" and len(pipeline.calls) == 2
    assert backend.live_keys() == []


def test_a_complete_not_found_result_is_cached() -> None:
    cache, _ = _layer()
    result = retrieval_result("hf_docs", not_found=True)
    service, pipeline = _service(cache, pipeline=FakeSearchPipeline("hf_docs", result=result))
    _search(service)
    again = _search(service)
    assert again.cache.outcome == "hit" and again.not_found is True and again.chunks == []
    assert len(again.below_threshold) == 3 and len(pipeline.calls) == 1


def test_invalid_requests_are_refused_before_the_cache_is_touched() -> None:
    from rag.serve.errors import ServiceError

    cache, backend = _layer()
    service, _ = _service(cache)
    with pytest.raises(ServiceError):
        _search(service, filters={"author": "x"})
    with pytest.raises(ServiceError):
        _search(service, "")
    assert backend.calls == {}


def test_a_store_that_is_down_never_fails_the_request() -> None:
    cache, backend = _layer()
    service, pipeline = _service(cache)
    backend.fail_with = CacheUnavailable("ConnectionError")
    one = _search(service)
    assert one.cache.outcome == "error" and one.cache.hit is False and one.chunks
    for _ in range(5):
        _search(service)
    last = _search(service)
    assert last.cache.outcome == "bypass"  # the breaker is open: no call reaches the store
    touched = sum(backend.calls.values())
    _search(service)
    assert sum(backend.calls.values()) == touched
    assert len(pipeline.calls) == 8


def test_the_cache_shows_in_the_span_the_metrics_and_the_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    cache, _ = _layer()
    with memory_observability(business_record_path=str(path)) as obs:
        service, _ = _service(cache, business_record_path=str(path))
        _search(service, request_id="miss-1")
        _search(service, request_id="hit-1")
        spans = [s for s in obs.span_exporter.get_finished_spans() if s.name == "search_docs"]
        by_request = {s.attributes[attr.REQUEST_ID]: s for s in spans}
        assert by_request["miss-1"].attributes[attr.CACHE_OUTCOME] == "miss"
        assert by_request["hit-1"].attributes[attr.CACHE_OUTCOME] == "hit"
        assert by_request["hit-1"].attributes[attr.CACHE_LAYER] == "l1"
        points = metric_points(obs, attr.METRIC_CACHE)
        counted = {(p[0]["rag.cache.outcome"], p[0]["rag.kb"], p[1].value) for p in points}
        assert ("miss", "hf_docs", 1) in counted and ("hit", "hf_docs", 1) in counted
    records = {
        r["request_id"]: r for r in (json.loads(x) for x in path.read_text("utf-8").splitlines())
    }
    assert records["hit-1"]["extra"]["cache"]["outcome"] == "hit"
    assert records["miss-1"]["extra"]["cache"]["outcome"] == "miss"
    assert records["hit-1"]["extra"]["cache"]["layer"] == "l1"
    assert [h["chunk_id"] for h in records["hit-1"]["final"]] == [
        h["chunk_id"] for h in records["miss-1"]["final"]
    ]


def test_a_cache_error_is_in_the_record_and_counted() -> None:
    cache, backend = _layer()
    backend.fail_with = CacheUnavailable("TimeoutError")
    with memory_observability() as obs:
        service, _ = _service(cache)
        _search(service)
        points = metric_points(obs, attr.METRIC_CACHE)
        assert any(p[0]["rag.cache.outcome"] == "error" for p in points)


def test_the_response_schema_accepts_the_cache_field() -> None:
    cache, _ = _layer()
    service, _ = _service(cache)
    response = _search(service)
    again = SearchResponse.model_validate(response.model_dump(mode="json"))
    assert again.cache is not None and again.cache.layer == "l1"
    assert set(again.cache.model_dump()) == {"layer", "hit", "outcome", "age_s"}


def test_concurrent_identical_searches_compute_once() -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    cache, _ = _layer()
    pipeline = FakeSearchPipeline("hf_docs")
    original = pipeline.retrieve

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)
        return original(*args, **kwargs)

    pipeline.retrieve = slow  # type: ignore[method-assign]
    service, _ = _service(cache, pipeline=pipeline)
    barrier = threading.Barrier(12)

    def one(i: int) -> Any:
        barrier.wait()
        return _search(service, request_id=f"c{i}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(one, range(12)))
    assert len(pipeline.calls) == 1
    assert sorted(r.request_id for r in responses) == sorted(f"c{i}" for i in range(12))
    assert all(r.chunks for r in responses)
    assert sorted(r.cache.outcome for r in responses).count("miss") == 1


def test_a_knowledge_base_can_opt_out_of_the_cache() -> None:
    cache, backend = _layer()
    private = KnowledgeBaseConfig(
        kb="hf_docs", language="en", access=AccessConfig(callers=["*"]), cache=False
    )
    pipeline = FakeSearchPipeline("hf_docs")
    service = SearchService(
        observe_settings(),
        configs={"hf_docs": private},
        builder=FakePipelineBuilder({"hf_docs": pipeline}),
        cache=cache,
    )
    service.load()
    one, two = _search(service), _search(service)
    assert len(pipeline.calls) == 2 and one.cache is None and two.cache is None
    assert backend.live_keys() == [] and backend.calls == {}


def test_the_opt_out_does_not_change_any_config_version() -> None:
    from rag.cache.keys import answer_config_version
    from rag.retrieve.pipeline import search_config_version

    settings = observe_settings()
    on = KnowledgeBaseConfig(kb="hf_docs")
    off = KnowledgeBaseConfig(kb="hf_docs", cache=False)
    assert search_config_version(on, settings, "m") == search_config_version(off, settings, "m")
    assert answer_config_version(on, settings) == answer_config_version(off, settings)
