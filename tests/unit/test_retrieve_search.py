"""Spec P0-07 AC11-AC12: HybridSearch returns RetrievalResult; config and factory checks."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings
from rag.eval.questions import LabeledQuestion, load_question_set, save_question_set
from rag.eval.retrieval import Provenance, Retriever, run_retrieval_eval
from rag.retrieve.deadline import Deadline
from rag.retrieve.hybrid import BM25, VECTOR, EsHybridRetriever
from rag.retrieve.search import HybridSearch, IndexEmbedModelError, make_hybrid_search
from rag.testing import FakeEmbedding
from rag.testing.retrieval import FakeMsearchClient, es_hit, search_error, search_response
from rag.testing.services import FakeElasticsearch
from rag.types import ErrorCategory, RouteHit

KB = "hf_docs"
INDEX = "kb_hf_docs_v1"
A = "hf_docs:guide/install.md"
B = "hf_docs:guide/other.md"
C = "hf_docs:api/install.md"


def _search(es: FakeMsearchClient, **kw: object) -> HybridSearch:
    retriever = EsHybridRetriever(
        es,
        FakeEmbedding(dim=4),
        kb=KB,
        index="kb_hf_docs",
        config=RetrievalConfig(),
        es_reserve_ms=100,
        es_margin_ms=200,
    )
    return HybridSearch(retriever, acl_groups=["public"], deadline_ms=5000, **kw)  # type: ignore[arg-type]


SOURCE_A = {
    "content": "Run pip install rag.",
    "title_path": "Guide > Install",
    "source_rel": "guide/install.md",
    "source_url": "https://example.org/guide/install.md",
    "source_version": "rev1",
    "doc_type": "markdown",
    "language": "en",
    "chunk_index": 0,
    "suspicious_instruction": False,
}


def _es() -> FakeMsearchClient:
    return FakeMsearchClient(
        bm25=search_response(
            [es_hit(f"{A}#0", 7.5, **SOURCE_A), es_hit(f"{B}#1", 5.0, content="Unrelated.")]
        ),
        vector=search_response(
            [
                # Same body as A#0 on another page: a duplicate after fusion.
                es_hit(f"{C}#0", 0.9, content="run  PIP install rag."),
                es_hit(f"{A}#0", 0.8, **SOURCE_A),
            ]
        ),
    )


def test_hybrid_search_result() -> None:
    search = _search(_es(), generation=lambda: 42)
    assert isinstance(search, Retriever)
    result = search.retrieve("how to install?", kb=KB, top_k=2, request_id="req-1")

    # Fused: A#0 (1/61 + 1/62), C#0 (1/61), B#1 (1/62). C#0 is a duplicate of A#0 and goes
    # before the top_k cut, so B#1 still makes the list.
    assert [c.chunk_id for c in result.chunks] == [f"{A}#0", f"{B}#1"]
    assert [c.doc_id for c in result.chunks] == [A, B]
    a, b = result.chunks
    assert a.score == pytest.approx(1 / 61 + 1 / 62)
    assert b.score == pytest.approx(1 / 62)
    assert a.kb == KB
    assert a.text == "Run pip install rag."
    assert a.untrusted is True
    assert a.title_path == ["Guide", "Install"]
    assert a.source_rel == "guide/install.md"
    assert a.source_url == "https://example.org/guide/install.md"
    assert a.source_version == "rev1"
    assert a.metadata["route_ranks"] == {BM25: 1, VECTOR: 2}
    assert a.metadata["route_scores"] == {BM25: 7.5, VECTOR: 0.8}
    assert a.metadata["doc_type"] == "markdown"
    assert a.metadata["language"] == "en"
    assert "acl_groups" not in a.metadata
    assert b.metadata["route_ranks"] == {BM25: 2}

    # Each route's own list, before fusion and dedupe.
    assert result.routes == {
        BM25: [
            RouteHit(chunk_id=f"{A}#0", doc_id=A, rank=1, score=7.5),
            RouteHit(chunk_id=f"{B}#1", doc_id=B, rank=2, score=5.0),
        ],
        VECTOR: [
            RouteHit(chunk_id=f"{C}#0", doc_id=C, rank=1, score=0.9),
            RouteHit(chunk_id=f"{A}#0", doc_id=A, rank=2, score=0.8),
        ],
    }
    assert result.request_id == "req-1"
    assert result.kb == KB
    assert result.index_version == INDEX
    assert result.data_generation == 42
    assert result.degraded is False
    assert result.error_categories == []
    assert {"embed", "search", "fuse", "dedupe", "total"} <= set(result.timings_ms)

    # A failed route shows as degraded; the request id is generated when not given.
    es = _es()
    es.responses[VECTOR] = search_error(400, "illegal_argument_exception", "bad vector")
    degraded = _search(es).retrieve("how to install?", kb=KB, top_k=8)
    assert degraded.degraded is True
    assert degraded.error_categories == [ErrorCategory.DEGRADED]
    assert list(degraded.routes) == [BM25]
    assert degraded.request_id
    assert degraded.data_generation == 0  # no generation reader given

    # Caller filters reach both routes; the wrong knowledge base is refused.
    es = _es()
    _search(es).retrieve("q", kb=KB, top_k=2, filters={"language": "en"}, deadline_ms=900)
    bodies = es.calls[0]["searches"][1::2]
    assert {"term": {"language": "en"}} in bodies[0]["query"]["bool"]["filter"]
    assert {"term": {"language": "en"}} in bodies[1]["knn"]["filter"]
    assert es.calls[0]["options"]["request_timeout"] <= 0.9 + 1e-6
    with pytest.raises(ValueError, match="other_kb"):
        _search(_es()).retrieve("q", kb="other_kb", top_k=2)


def test_hybrid_search_in_eval_harness(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    save_question_set(
        [
            LabeledQuestion(qid="q1", question="install?", relevant_pages=[A]),
            LabeledQuestion(qid="q2", question="other?", relevant_pages=[B]),
        ],
        path,
    )
    question_set = load_question_set(path, kb=KB)
    record = run_retrieval_eval(
        _search(_es()),
        question_set,
        kb_config=KnowledgeBaseConfig(kb=KB),
        top_k=3,
        ks=(1, 3),
        provenance=Provenance(),
        now=datetime(2026, 10, 17, tzinfo=UTC),
    )
    assert record.run.lists == ["final", BM25, VECTOR]
    assert record.run.index_versions == [INDEX]
    q1, q2 = record.questions
    assert q1.error is None
    assert q1.metrics["final"]["page"]["hit@1"] == 1.0
    assert q1.metrics[BM25]["page"]["mrr@3"] == 1.0
    assert q1.metrics[VECTOR]["page"]["mrr@3"] == 0.5
    assert q2.metrics["final"]["page"]["mrr@3"] == 0.5
    assert q2.metrics[VECTOR]["page"]["hit@3"] == 0.0
    assert q1.hits[0].route_ranks == {BM25: 1, VECTOR: 2}
    assert record.counts["degraded"] == 0


def test_retrieval_config_defaults() -> None:
    cfg = RetrievalConfig()
    assert cfg.top_k_per_route == 20
    assert cfg.num_candidates == 100
    assert cfg.rrf_k == 60
    assert cfg.title_boost == 1.0  # P0-13b: confirmed on the synthetic dev set
    with pytest.raises(ValidationError, match="num_candidates"):
        RetrievalConfig(top_k_per_route=50, num_candidates=40)
    with pytest.raises(ValidationError):
        RetrievalConfig(num_candidates=10_001)  # Elasticsearch's limit
    with pytest.raises(ValidationError):
        RetrievalConfig(rrf_k=-1)
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.search_deadline_ms == 10_000
    assert settings.search_es_reserve_ms == 200
    assert settings.search_es_reserve_ratio == 0.2
    assert settings.search_es_margin_ms == 200
    with pytest.raises(ValidationError):
        ServiceSettings(_env_file=None, search_es_reserve_ratio=1.0)  # type: ignore[call-arg]
    assert Deadline.after_ms(1000).remaining_ms() <= 1000


def test_factory_refuses_other_embed_model() -> None:
    es = FakeElasticsearch()
    es.indices.create(
        index=INDEX, mappings={"_meta": {"embed_model": "qwen/text-embedding-v4@1024"}}
    )
    es.indices.update_aliases(actions=[{"add": {"index": INDEX, "alias": "kb_hf_docs"}}])
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]
    cfg = KnowledgeBaseConfig(kb=KB)
    with pytest.raises(IndexEmbedModelError, match="fake-embedding@4"):
        make_hybrid_search(
            cfg, settings, acl_groups=["public"], es=es, embed_model=FakeEmbedding(dim=4)
        )

    # Same model: built, bound to the alias, with the settings' deadline.
    es.indices.put_mapping(index=INDEX, meta={"embed_model": "fake-embedding@4"})
    search = make_hybrid_search(
        cfg, settings, acl_groups=["public"], es=es, embed_model=FakeEmbedding(dim=4)
    )
    assert search.retriever.index == "kb_hf_docs"
    assert search.deadline_ms == settings.search_deadline_ms
    assert search.retriever.es_reserve_ms == settings.search_es_reserve_ms
    assert search.retriever.es_reserve_ratio == settings.search_es_reserve_ratio
    # With an index prefix the fake-vector alias is used.
    es.indices.create(
        index="fake_kb_hf_docs_v1", mappings={"_meta": {"embed_model": "fake-embedding@4"}}
    )
    es.indices.update_aliases(
        actions=[{"add": {"index": "fake_kb_hf_docs_v1", "alias": "fake_kb_hf_docs"}}]
    )
    fake = make_hybrid_search(
        cfg,
        settings,
        acl_groups=["public"],
        es=es,
        embed_model=FakeEmbedding(dim=4),
        index_prefix="fake_",
    )
    assert fake.retriever.index == "fake_kb_hf_docs"


def test_data_generation_read_is_bounded_by_the_deadline() -> None:
    def slow_generation() -> int:
        time.sleep(1.5)  # a Redis that does not answer
        return 7

    search = _search(_es(), generation=slow_generation)
    start = time.monotonic()
    result = search.retrieve("q", kb=KB, top_k=2, deadline_ms=300)
    assert time.monotonic() - start < 1.0
    assert result.data_generation == 0  # unknown, the search result is still returned
    assert [c.chunk_id for c in result.chunks] == [f"{A}#0", f"{B}#1"]

    def failing_generation() -> int:
        raise RedisConnectionError("redis down")

    assert (
        _search(_es(), generation=failing_generation).retrieve("q", kb=KB, top_k=2).data_generation
        == 0
    )
    assert _search(_es(), generation=lambda: 9).retrieve("q", kb=KB, top_k=2).data_generation == 9


def test_acl_groups_must_be_a_list() -> None:
    retriever = _search(_es()).retriever
    with pytest.raises(TypeError, match="acl_groups"):
        HybridSearch(retriever, acl_groups="grp", deadline_ms=1000)


def test_query_embedding_thread_sees_the_probe_request() -> None:
    """The query vector is computed in a thread pool; the request context must travel with it,
    otherwise a probe's vector is written to the L0 cache (P5-02 review round 2)."""
    from rag.observe.context import bind_request, current_request_is_probe

    seen: list[bool] = []

    class RecordingEmbedding(FakeEmbedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            seen.append(current_request_is_probe())
            return super()._get_query_embedding(query)

    retriever = EsHybridRetriever(
        _es(),
        RecordingEmbedding(dim=4),
        kb=KB,
        index="kb_hf_docs",
        config=RetrievalConfig(),
        es_reserve_ms=100,
        es_margin_ms=200,
    )
    search = HybridSearch(retriever, acl_groups=["public"], deadline_ms=5000)
    with bind_request("probe-abc"):
        search.retrieve("how to install?", kb=KB, top_k=2, request_id="probe-abc")
    assert seen and all(seen)
