"""Spec P5-01 AC9: the semantic cache in shadow mode: partitions, distance, entity guard, records,
and above all that the returned result is the same with the layer on and off."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.semantic import (
    EsSemanticStore,
    MemorySemanticStore,
    Partition,
    ShadowRecorder,
    StoredQuery,
    question_hash,
)
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe.rotating import RotatingJsonlWriter
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.agent import pages_result
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline
from rag.types import RetrievalResult

DIM = 8


def vec(*parts: float) -> list[float]:
    out = list(parts) + [0.0] * (DIM - len(parts))
    return out


def near(base: list[float], angle: float) -> list[float]:
    """A unit vector at `angle` radians from the unit vector `base` (cosine distance
    1 - cos(angle))."""
    b = np.asarray(base, dtype=float)
    b = b / np.linalg.norm(b)
    other = np.zeros(DIM)
    other[DIM - 1] = 1.0
    return list(np.cos(angle) * b + np.sin(angle) * other)


PART = Partition("hf_docs", "f1", "a1", "cfg", 7, 8, "en")


def _entry(question: str, ids: tuple[str, ...] = ("c1", "c2"), at: float = 1000.0) -> StoredQuery:
    return StoredQuery(question_hash(question), question, ids, tuple(i[:1] for i in ids), at)


# ---- the memory store ------------------------------------------------------------------------


def test_memory_store_returns_the_nearest_first_with_cosine_distance() -> None:
    store = MemorySemanticStore()
    base = vec(1.0)
    store.add(PART, near(base, 0.5), _entry("far"), ttl_s=100)
    store.add(PART, near(base, 0.1), _entry("close"), ttl_s=100)
    found = store.search(PART, base, k=2, now=1001.0)
    assert [c.entry.question for c in found] == ["close", "far"]
    assert found[0].distance == pytest.approx(1 - np.cos(0.1), abs=1e-6)
    assert found[1].distance == pytest.approx(1 - np.cos(0.5), abs=1e-6)


@pytest.mark.parametrize(
    "other",
    [
        Partition("mt5", "f1", "a1", "cfg", 7, 8, "en"),
        Partition("hf_docs", "f2", "a1", "cfg", 7, 8, "en"),
        Partition("hf_docs", "f1", "a2", "cfg", 7, 8, "en"),
        Partition("hf_docs", "f1", "a1", "cfg2", 7, 8, "en"),
        Partition("hf_docs", "f1", "a1", "cfg", 8, 8, "en"),
        Partition("hf_docs", "f1", "a1", "cfg", 7, 9, "en"),
        Partition("hf_docs", "f1", "a1", "cfg", 7, 8, "zh"),
    ],
    ids=["kb", "filters", "acl", "config", "generation", "top_k", "language"],
)
def test_memory_store_never_matches_across_partitions(other: Partition) -> None:
    store = MemorySemanticStore()
    store.add(PART, vec(1.0), _entry("q"), ttl_s=100)
    assert store.search(other, vec(1.0), k=3, now=1001.0) == []
    assert len(store.search(PART, vec(1.0), k=3, now=1001.0)) == 1


def test_memory_store_expiry_and_size_limit() -> None:
    store = MemorySemanticStore(max_entries=2)
    store.add(PART, vec(1.0), _entry("old", at=1000.0), ttl_s=10)
    assert store.search(PART, vec(1.0), k=1, now=1005.0)
    assert store.search(PART, vec(1.0), k=1, now=1011.0) == []
    store.add(PART, vec(0.0, 1.0), _entry("b", at=1000.0), ttl_s=100)
    store.add(PART, vec(0.0, 0.0, 1.0), _entry("c", at=1000.0), ttl_s=100)
    assert store.size() == 2  # the oldest was dropped


# ---- the recorder ----------------------------------------------------------------------------


def _fresh(*pages: int) -> RetrievalResult:
    return pages_result("hf_docs", list(pages) or [1, 2, 3])


class Clock:
    now = 5000.0

    def __call__(self) -> float:
        return self.now


def _recorder(tmp_path: Path, **kw: Any) -> tuple[ShadowRecorder, MemorySemanticStore, Path]:
    store = MemorySemanticStore()
    path = tmp_path / "shadow.jsonl"
    recorder = ShadowRecorder(
        store,
        RotatingJsonlWriter(path, max_bytes=10_000_000, backups=1),
        threshold=kw.pop("threshold", 0.2),
        ttl_s=100,
        clock=Clock(),
        **kw,
    )
    return recorder, store, path


def _observe(
    recorder: ShadowRecorder,
    question: str,
    vector: list[float] | None,
    fresh: RetrievalResult | None = None,
    **over: Any,
) -> Any:
    args: dict[str, Any] = {
        "kb": "hf_docs",
        "question": question,
        "filters": None,
        "acl_groups": ["public"],
        "top_k": 8,
        "config_version": "cfg",
        "data_generation": 7,
        "fresh": fresh or _fresh(),
        "vector": vector,
    }
    args.update(over)
    return recorder.observe(**args)


def test_without_a_vector_nothing_is_looked_up_and_nothing_stored(tmp_path: Path) -> None:
    recorder, store, _ = _recorder(tmp_path)
    record = _observe(recorder, "how do i push a model", None)
    assert record.status == "no_vector" and store.size() == 0


def test_first_question_finds_nothing_and_is_stored(tmp_path: Path) -> None:
    recorder, store, _ = _recorder(tmp_path)
    record = _observe(recorder, "how do i push a model", vec(1.0))
    assert record.status == "empty" and store.size() == 1


def test_a_close_question_with_the_same_entities_would_hit(tmp_path: Path) -> None:
    recorder, store, path = _recorder(tmp_path)
    _observe(recorder, "how do i push a model with push_to_hub", vec(1.0))
    record = _observe(
        recorder, "what is the way to push a model with push_to_hub", near(vec(1.0), 0.1)
    )
    assert record.status == "would_hit" and record.guard_ok is True
    assert record.distance == pytest.approx(1 - np.cos(0.1), abs=1e-4)
    assert record.same_top1 is True and record.overlap == 1.0
    assert store.size() == 1  # a live L2 would have served the stored entry and written nothing
    lines = [json.loads(x) for x in path.read_text("utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["empty", "would_hit"]
    assert lines[1]["question_hash"] == question_hash(
        "what is the way to push a model with push_to_hub"
    )
    assert lines[1]["candidate_hash"] == question_hash("how do i push a model with push_to_hub")


def test_a_close_question_with_another_identifier_is_rejected_by_the_guard(tmp_path: Path) -> None:
    recorder, store, _ = _recorder(tmp_path)
    _observe(recorder, "What does push_to_hub do?", vec(1.0))
    record = _observe(recorder, "What does push_to_hub_fastai do?", near(vec(1.0), 0.02))
    assert record.status == "guard_rejected" and record.guard_reason == "identifiers"
    assert record.distance == pytest.approx(1 - np.cos(0.02), abs=1e-4)
    assert store.size() == 2  # it is a new question for the store


def test_a_question_beyond_the_threshold_is_too_far(tmp_path: Path) -> None:
    recorder, _, _ = _recorder(tmp_path, threshold=0.05)
    _observe(recorder, "how do i push a model with push_to_hub", vec(1.0))
    record = _observe(
        recorder, "how do i push a model with push_to_hub please", near(vec(1.0), 0.5)
    )
    assert record.status == "too_far" and record.nearest_distance == pytest.approx(
        1 - np.cos(0.5), abs=1e-4
    )


def test_a_question_without_entities_needs_a_stricter_distance(tmp_path: Path) -> None:
    recorder, _, _ = _recorder(tmp_path, threshold=0.2, strict_factor=0.5)
    _observe(recorder, "how do i upload a model", vec(1.0))
    # distance 0.15: inside 0.2 but outside 0.2 * 0.5
    record = _observe(recorder, "how can i upload a model", near(vec(1.0), np.arccos(0.85)))
    assert record.status == "too_far"
    recorder2, _, _ = _recorder(tmp_path, threshold=0.2, strict_factor=0.5)
    _observe(recorder2, "how do i upload a model", vec(1.0))
    close = _observe(recorder2, "how can i upload a model", near(vec(1.0), np.arccos(0.95)))
    assert close.status == "would_hit" and close.entities is False


def test_a_wrong_hit_has_a_different_top_chunk(tmp_path: Path) -> None:
    recorder, _, _ = _recorder(tmp_path)
    _observe(recorder, "how do i push a model with push_to_hub", vec(1.0), _fresh(1, 2, 3))
    record = _observe(
        recorder,
        "what is the way to push a model with push_to_hub",
        near(vec(1.0), 0.05),
        _fresh(2, 1, 3),
    )
    assert record.status == "would_hit"
    assert record.same_top1 is False and record.overlap == 1.0


def test_filters_acl_and_generation_make_other_partitions(tmp_path: Path) -> None:
    recorder, store, _ = _recorder(tmp_path)
    _observe(recorder, "how do i push a model with push_to_hub", vec(1.0))
    for over in (
        {"filters": {"language": "en"}},
        {"acl_groups": ["public", "hr"]},
        {"data_generation": 8},
        {"top_k": 3},
        {"config_version": "cfg2"},
    ):
        record = _observe(recorder, "how do i push a model with push_to_hub", vec(1.0), **over)
        assert record.status == "empty", over
    assert store.size() == 6


# ---- the result is never changed -------------------------------------------------------------


class VectorPipeline(FakeSearchPipeline):
    """A fake chain that also gives the question's vector (what the L0 cache would)."""

    def query_vector(self, question: str) -> list[float] | None:
        return vec(1.0) if "push" in question else None


def _service(cache: ResultCache | None, pipeline: FakeSearchPipeline) -> SearchService:
    service = SearchService(
        observe_settings(),
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": pipeline}),
        cache=cache,
    )
    service.load()
    return service


def _scrub(response: Any) -> dict[str, Any]:
    data = response.model_dump(mode="json")
    for name in ("request_id", "timings_ms", "cache"):
        data.pop(name)
    return data


def test_the_returned_result_is_byte_identical_with_shadow_mode_on_and_off(tmp_path: Path) -> None:
    questions = [
        "how do i push a model with push_to_hub",
        "what is the way to push a model with push_to_hub",
        "How do I push a model with push_to_hub?",
        "push_to_hub_fastai how to",
        "where are the docs",
    ]
    creds = Credentials(transport="http")
    off = _service(None, VectorPipeline("hf_docs"))
    settings = observe_settings()
    recorder, _, path = _recorder(tmp_path, threshold=0.5)
    shadow = _service(
        ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings, shadow=recorder),
        VectorPipeline("hf_docs"),
    )
    plain = ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings)
    l1_only = _service(plain, VectorPipeline("hf_docs"))
    for q in questions:
        a = _scrub(off.search_docs(creds, kb="hf_docs", query=q))
        b = _scrub(shadow.search_docs(creds, kb="hf_docs", query=q))
        c = _scrub(l1_only.search_docs(creds, kb="hf_docs", query=q))
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert json.dumps(a, sort_keys=True) == json.dumps(c, sort_keys=True)
    statuses = [json.loads(x)["status"] for x in path.read_text("utf-8").splitlines()]
    # One record per question (every question missed L1 and had a complete result): the first
    # finds nothing, the next two are close and have the same entities, the fourth is close but
    # names another function, the last has no vector (it is not in the L0 cache).
    assert statuses == ["empty", "would_hit", "would_hit", "guard_rejected", "no_vector"]


def test_a_failing_shadow_never_changes_or_fails_the_request(tmp_path: Path) -> None:
    class Broken(MemorySemanticStore):
        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

    recorder = ShadowRecorder(Broken(), None, threshold=0.2, ttl_s=10)
    cache = ResultCache(
        ExactCache(MemoryCacheBackend(), CacheConfig()), observe_settings(), shadow=recorder
    )
    service = _service(cache, VectorPipeline("hf_docs"))
    creds = Credentials(transport="http")
    response = service.search_docs(creds, kb="hf_docs", query="how do i push a model")
    assert response.chunks and response.cache.outcome == "miss"


def test_shadow_runs_only_after_a_miss_with_a_complete_result(tmp_path: Path) -> None:
    recorder, store, _ = _recorder(tmp_path)
    cache = ResultCache(
        ExactCache(MemoryCacheBackend(), CacheConfig()), observe_settings(), shadow=recorder
    )
    degraded = pages_result("hf_docs", [1, 2], degraded=True)
    service = _service(cache, VectorPipeline("hf_docs", result=degraded))
    creds = Credentials(transport="http")
    service.search_docs(creds, kb="hf_docs", query="how do i push a model")
    assert store.size() == 0  # an incomplete result is never a reference for later questions
    good = _service(cache, VectorPipeline("hf_docs"))
    good.search_docs(creds, kb="hf_docs", query="how do i push a model")
    good.search_docs(creds, kb="hf_docs", query="how do i push a model")  # L1 hit: no shadow
    assert store.size() == 1


# ---- the Elasticsearch store (query shapes; the real index is in the integration test) --------


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.existing: set[str] = set()

    def exists(self, index: str) -> bool:
        return index in self.existing

    def create(self, index: str, **body: Any) -> None:
        self.created.append((index, body))
        self.existing.add(index)

    def delete(self, index: str, ignore_unavailable: bool = False) -> None:
        self.existing.discard(index)


class FakeEs:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.searches: list[dict[str, Any]] = []
        self.indexed: list[dict[str, Any]] = []
        self.hits: list[dict[str, Any]] = []

    def options(self, **kwargs: Any) -> FakeEs:
        return self

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {"hits": {"hits": self.hits}}

    def index(self, **kwargs: Any) -> None:
        self.indexed.append(kwargs)

    def close(self) -> None:
        pass


def test_es_store_creates_its_index_filters_by_partition_and_converts_the_score() -> None:
    es = FakeEs()
    store = EsSemanticStore(es, dim=DIM, index_prefix="p501test_")  # type: ignore[arg-type]
    store.add(PART, vec(1.0), _entry("q"), ttl_s=100)
    ((index, body),) = es.indices.created
    assert index == "p501test_cache_semantic_hf_docs"
    assert body["mappings"]["dynamic"] == "strict"
    assert body["mappings"]["properties"]["vector"]["dims"] == DIM
    doc = es.indexed[0]["document"]
    assert doc["expires_at"] == 1100.0 and doc["data_generation"] == 7 and doc["top_k"] == 8
    es.hits = [
        {
            "_id": "e1",
            "_score": 0.95,  # cosine 0.9
            "_source": {
                "question": "q",
                "top_chunk_ids": ["c1"],
                "top_doc_ids": ["d1"],
                "created_at": 1000.0,
            },
        }
    ]
    found = store.search(PART, vec(1.0), k=3, now=1001.0)
    assert found[0].distance == pytest.approx(0.1, abs=1e-9)
    knn = es.searches[0]["knn"]
    terms = {next(iter(f["term"].items())) for f in knn["filter"] if "term" in f}
    assert ("data_generation", 7) in terms and ("acl_hash", "a1") in terms
    assert ("top_k", 8) in terms and ("language", "en") in terms
    assert ("config_version", "cfg") in terms and ("filters_hash", "f1") in terms
    assert any("range" in f for f in knn["filter"])  # expiry
    assert "term" in {k for f in knn["filter"] for k in f} and es.searches[0]["index"] == index


def test_the_question_vector_comes_from_the_l0_cache_and_never_from_the_model() -> None:
    from rag.config import RetrievalConfig
    from rag.embed.cache import CachedEmbedding, InMemoryEmbeddingCache
    from rag.retrieve.hybrid import EsHybridRetriever
    from rag.retrieve.search import HybridSearch
    from rag.testing import FakeEmbedding
    from rag.testing.retrieval import FakeMsearchClient

    inner = FakeEmbedding(dim=4)
    cached = CachedEmbedding(inner, InMemoryEmbeddingCache(), cache_model="m@4")

    def search_over(model: Any) -> HybridSearch:
        retriever = EsHybridRetriever(
            FakeMsearchClient(),  # type: ignore[arg-type]
            model,
            kb="hf_docs",
            index="kb_hf_docs",
            config=RetrievalConfig(),
            es_reserve_ms=100,
            es_margin_ms=200,
        )
        return HybridSearch(retriever, acl_groups=["public"], deadline_ms=1000)

    search = search_over(cached)
    assert search.query_vector("how do i push") is None  # not embedded yet: no model call
    vector = cached.get_query_embedding("how do i push")  # what a search does
    assert search.query_vector("how do i push") == pytest.approx(vector)
    assert search.query_vector("another question") is None
    assert search_over(inner).query_vector("how do i push") is None  # a model with no L0 cache
