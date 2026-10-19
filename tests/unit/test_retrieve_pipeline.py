"""Spec P0-08 AC12-AC14: SearchPipeline and its factory (design s9-1, s10-1 to s10-3, s16-8)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings, load_kb_config
from rag.config.paths import repo_path
from rag.eval.questions import LabeledQuestion, load_question_set, save_question_set
from rag.eval.retrieval import Provenance, Retriever, run_retrieval_eval
from rag.retrieve import pipeline as pipeline_module
from rag.retrieve.context import ContextExpander, ContextReader
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.hybrid import BM25, VECTOR, EsHybridRetriever
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline, search_config_version
from rag.retrieve.rerank import BatchedTimeoutRerank, RerankRunner
from rag.retrieve.threshold import RerankThreshold
from rag.testing import FakeEmbedding
from rag.testing.rerank import FakeCrossEncoder, ManualClock
from rag.testing.retrieval import FakeMsearchClient, es_hit, search_response
from rag.testing.services import FakeElasticsearch
from rag.types import (
    ContextPiece,
    ErrorCategory,
    RetrievalResult,
    RouteHit,
    ThresholdDecision,
)

KB = "hf_docs"
ALIAS = "kb_hf_docs"
INDEX = "kb_hf_docs_v1"


def _page(n: int) -> str:
    return f"hf_docs:p{n}.md"


def _src(n: int, i: int = 0, body: str | None = None) -> dict[str, Any]:
    return {
        "content": body if body is not None else f"body of page {n} part {i}",
        "title_path": f"Page {n}",
        "source_rel": f"p{n}.md",
        "chunk_index": i,
        "parent_id": _page(n),
        "language": "en",
        "doc_type": "markdown",
    }


def _hit(n: int, score: float, body: str | None = None) -> dict[str, Any]:
    return es_hit(f"{_page(n)}#0", score, **_src(n, body=body))


def _msearch(*, bm25: list[int] | None = None) -> FakeMsearchClient:
    """BM25: p1, p2, p3, p4; vector: p3, p1, p5 (p5 has p1's body: removed by dedupe).
    Fused: p1 (1/61 + 1/62), p3 (1/63 + 1/61), p2 (1/62), p5 (1/63), p4 (1/64)."""
    pages = bm25 if bm25 is not None else [1, 2, 3, 4]
    return FakeMsearchClient(
        bm25=search_response([_hit(n, 10.0 - r) for r, n in enumerate(pages)]),
        vector=search_response(
            []
            if bm25 is not None
            else [_hit(3, 0.9), _hit(1, 0.8), _hit(5, 0.7, body="body of page 1 part 0")]
        ),
    )


def _text(n: int) -> str:
    return f"Page {n}\nbody of page {n} part 0"


def count_tokens(text: str) -> int:
    return len(text.split())


def _expand_es() -> FakeElasticsearch:
    es = FakeElasticsearch()
    es.mappings[INDEX] = {"properties": {}}
    es.aliases[ALIAS] = INDEX
    es.docs[INDEX] = {
        f"{_page(n)}#{i}": {
            **_src(n, i),
            "chunk_id": f"{_page(n)}#{i}",
            "doc_id": _page(n),
            "kb": KB,
            "acl_groups": ["public"],
        }
        for n in range(1, 6)
        for i in range(2)
    }
    return es


def _pipeline(
    msearch: FakeMsearchClient,
    scorer: FakeCrossEncoder,
    *,
    expand: str = "none",
    threshold: float = 0.3,
    rerank_n: int = 30,
    clock: ManualClock | None = None,
) -> SearchPipeline:
    retriever = EsHybridRetriever(
        msearch,
        FakeEmbedding(dim=4),
        kb=KB,
        index=ALIAS,
        config=RetrievalConfig(),
        es_reserve_ms=100,
        es_margin_ms=200,
    )
    runner = RerankRunner(scorer, batch_size=8, max_concurrency=2)
    reader = ContextReader(_expand_es(), kb=KB, index=ALIAS, count_tokens=count_tokens)
    return SearchPipeline(
        retriever,
        reranker=BatchedTimeoutRerank(runner=runner, top_n=rerank_n),
        threshold=RerankThreshold(threshold=threshold),
        expander=ContextExpander(reader=reader, mode=expand, window=1, max_tokens=4000)
        if expand != "none"
        else None,
        reader=reader,
        acl_groups=["public"],
        deadline_ms=5000,
        config_version="cfg1",
        **({"clock": clock} if clock is not None else {}),
    )


SCORES = {_text(2): 0.95, _text(4): 0.6, _text(1): 0.4, _text(3): 0.1}


def test_pipeline_runs_full_chain() -> None:
    pipeline = _pipeline(_msearch(), FakeCrossEncoder(SCORES), expand="neighbors")
    assert isinstance(pipeline, Retriever)
    result = pipeline.retrieve("which page?", kb=KB, top_k=2, request_id="req-1")

    # Dedupe drops p5; rerank orders p2, p4, p1, p3; the threshold drops p3; top_k keeps 2.
    assert [c.chunk_id for c in result.chunks] == [f"{_page(2)}#0", f"{_page(4)}#0"]
    p2, p4 = result.chunks
    assert p2.rerank_score == 0.95 and p4.rerank_score == 0.6
    assert p2.score == pytest.approx(1 / 62) and p4.score == pytest.approx(1 / 64)  # fusion
    assert p2.metadata["route_ranks"] == {BM25: 2}
    assert result.rerank == [
        RouteHit(chunk_id=f"{_page(2)}#0", doc_id=_page(2), rank=1, score=0.95),
        RouteHit(chunk_id=f"{_page(4)}#0", doc_id=_page(4), rank=2, score=0.6),
        RouteHit(chunk_id=f"{_page(1)}#0", doc_id=_page(1), rank=3, score=0.4),
        RouteHit(chunk_id=f"{_page(3)}#0", doc_id=_page(3), rank=4, score=0.1),
    ]
    assert result.threshold == ThresholdDecision(
        threshold=0.3, top_score=0.95, applied=True, passed=True, kept=3, dropped=1
    )
    assert result.not_found is False and result.below_threshold == []
    assert result.rerank_timeout is False and result.degraded is False
    assert result.error_categories == []
    # Expansion runs after the top_k cut, on the kept chunks only.
    assert p2.context == [
        ContextPiece(chunk_id=f"{_page(2)}#1", chunk_index=1, text="body of page 2 part 1")
    ]
    assert [p.chunk_id for p in p4.context] == [f"{_page(4)}#1"]
    assert result.request_id == "req-1"
    assert result.index_version == INDEX
    assert result.config_version == "cfg1"
    assert set(result.routes) == {BM25, VECTOR}
    assert {
        "embed",
        "search",
        "fuse",
        "dedupe",
        "rerank",
        "rerank_queue",
        "threshold",
        "expand",
        "total",
    } <= set(result.timings_ms)
    assert RetrievalResult.model_validate_json(result.model_dump_json()) == result


def test_pipeline_in_eval_harness(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    save_question_set(
        [
            LabeledQuestion(qid="q1", question="which page?", relevant_pages=[_page(2)]),
            LabeledQuestion(qid="q2", question="which page?", relevant_pages=[_page(1)]),
        ],
        path,
    )
    record = run_retrieval_eval(
        _pipeline(_msearch(), FakeCrossEncoder(SCORES)),
        load_question_set(path, kb=KB),
        kb_config=KnowledgeBaseConfig(kb=KB),
        top_k=3,
        ks=(1, 3),
        provenance=Provenance(),
        now=datetime(2026, 10, 17, tzinfo=UTC),
    )
    q1, q2 = record.questions
    assert q1.metrics["final"]["page"]["hit@1"] == 1.0  # rerank moved p2 from fused rank 3 to 1
    assert q2.metrics["final"]["page"]["mrr@3"] == pytest.approx(1 / 3)
    assert q2.metrics[BM25]["page"]["hit@1"] == 1.0  # the routes keep their own lists


def test_pipeline_not_found_result() -> None:
    scores = {_text(n): s for n, s in [(1, 0.2), (2, 0.05), (3, 0.25), (4, 0.01)]}
    result = _pipeline(_msearch(), FakeCrossEncoder(scores)).retrieve("q", kb=KB, top_k=8)
    assert result.chunks == []
    assert result.not_found is True
    assert [(c.chunk_id, c.rerank_score) for c in result.below_threshold] == [
        (f"{_page(3)}#0", 0.25),
        (f"{_page(1)}#0", 0.2),
        (f"{_page(2)}#0", 0.05),
    ]
    assert result.below_threshold[0].text == "body of page 3 part 0"
    assert result.threshold is not None and result.threshold.passed is False
    assert result.error_categories == []  # not found is an answer, not an error


def test_pipeline_rerank_timeout_result() -> None:
    # A manual clock drives the whole request: 12 candidates, 0.05 s per pair, 0.5 s in total.
    # Batch 1 (8 pairs) ends with 0.1 s left, batch 2 (4 pairs) would need 0.2 s: not started.
    clock = ManualClock()
    pages = list(range(1, 13))
    scores = {_text(n): 0.5 + n / 100 for n in pages} | {_text(1): 0.1}
    scorer = FakeCrossEncoder(scores, clock=clock, per_pair_s=0.05)
    pipeline = _pipeline(_msearch(bm25=pages), scorer, clock=clock)
    result = pipeline.retrieve("q", kb=KB, top_k=10, deadline_ms=500)

    assert result.rerank_timeout is True
    assert result.error_categories == [ErrorCategory.RERANK_TIMEOUT]
    assert [len(b) for b in scorer.batches] == [8]
    # Scored above the threshold, then unscored in fusion order; p1 (0.1, below) moves to the
    # end and falls out of the top 10, nothing is dropped by the threshold.
    ids = [c.chunk_id for c in result.chunks]
    assert ids == [f"{_page(n)}#0" for n in (8, 7, 6, 5, 4, 3, 2, 9, 10, 11)]
    assert [c.rerank_score for c in result.chunks[7:]] == [None, None, None]
    assert result.not_found is False
    assert result.threshold is not None
    assert (result.threshold.applied, result.threshold.kept, result.threshold.dropped) == (
        False,
        12,
        0,
    )


def test_make_search_pipeline_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    es = _expand_es()
    es.mappings[INDEX] = {"_meta": {"embed_model": "fake-embedding@4"}, "properties": {}}
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]
    retrieval = RetrievalConfig(
        rerank_n=20,
        final_k=5,
        threshold=0.4,
        expand="neighbors",
        expand_window=2,
        expand_max_tokens=900,
    )
    cfg = KnowledgeBaseConfig(kb=KB, retrieval=retrieval)
    runner = RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1)

    def build(cfg: KnowledgeBaseConfig, **kw: Any) -> SearchPipeline:
        return make_search_pipeline(
            cfg,
            settings,
            acl_groups=["public"],
            es=es,
            embed_model=FakeEmbedding(dim=4),
            count_tokens=count_tokens,
            **kw,
        )

    pipeline = build(cfg, runner=runner)
    dedupe, rerank, threshold = pipeline.postprocessors
    assert isinstance(dedupe, TextDedupe)
    assert isinstance(rerank, BatchedTimeoutRerank)
    assert rerank.top_n == 20 and rerank.runner is runner
    assert isinstance(threshold, RerankThreshold) and threshold.threshold == 0.4
    assert pipeline.expander is not None
    assert (pipeline.expander.mode, pipeline.expander.window) == ("neighbors", 2)
    assert pipeline.expander.max_tokens == 900
    assert pipeline.retriever.index == ALIAS
    assert pipeline.deadline_ms == settings.search_deadline_ms
    assert pipeline.config_version == search_config_version(cfg, settings, "fake-embedding@4")
    assert len(pipeline.config_version) == 16

    # get_context uses the KB's expansion mode and window by default.
    context = pipeline.get_context(f"{_page(3)}#0")
    assert [c.chunk_id for c in context.chunks] == [f"{_page(3)}#0", f"{_page(3)}#1"]
    assert context.mode == "neighbors"

    # Another threshold is another config version; no expansion means no expander.
    other = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(threshold=0.5))
    plain = build(other, runner=runner)
    assert plain.expander is None
    assert plain.config_version != pipeline.config_version
    assert plain.get_context(f"{_page(3)}#1").mode == "chunk"
    other_reranker = settings.model_copy(update={"reranker_model": "other/reranker"})
    assert search_config_version(other, other_reranker, "fake-embedding@4") != (
        plain.config_version
    )

    # Without a runner the process-wide one is used (the model is loaded once per process).
    shared = RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1)
    monkeypatch.setattr(pipeline_module, "get_rerank_runner", lambda s: shared)
    assert build(cfg).postprocessors[1].runner is shared


def test_pipeline_passes_the_rerank_title_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec P2-03 AC4: `retrieval.rerank_with_title` reaches the reranker, defaults to true and
    is part of the config version (another setting is another version)."""
    assert RetrievalConfig().rerank_with_title is True
    hf = load_kb_config(repo_path("kb/public/hf_docs.yaml"), kb_root=repo_path("kb"))
    assert hf.retrieval.rerank_with_title is True
    es = _expand_es()
    es.mappings[INDEX] = {"_meta": {"embed_model": "fake-embedding@4"}, "properties": {}}
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]
    runner = RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1)

    def build(cfg: KnowledgeBaseConfig) -> SearchPipeline:
        return make_search_pipeline(
            cfg,
            settings,
            acl_groups=["public"],
            es=es,
            embed_model=FakeEmbedding(dim=4),
            count_tokens=count_tokens,
            runner=runner,
        )

    on = build(KnowledgeBaseConfig(kb=KB))
    off_cfg = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(rerank_with_title=False))
    off = build(off_cfg)
    assert on.postprocessors[1].include_title is True
    assert off.postprocessors[1].include_title is False
    assert on.config_version != off.config_version
    assert off.config_version == search_config_version(off_cfg, settings, "fake-embedding@4")


def test_retrieval_config_rerank_and_expand_defaults() -> None:
    cfg = RetrievalConfig()
    assert (cfg.rerank_n, cfg.final_k, cfg.threshold) == (30, 8, 0.3)
    assert (cfg.expand, cfg.expand_window, cfg.expand_max_tokens) == ("none", 1, 4000)
    for bad in ({"threshold": 1.5}, {"threshold": -0.1}, {"expand": "page"}, {"expand_window": 0}):
        with pytest.raises(ValidationError):
            RetrievalConfig(**bad)
    hf = load_kb_config(repo_path("kb/public/hf_docs.yaml"), kb_root=repo_path("kb"))
    assert hf.retrieval.expand == "none"
    assert hf.retrieval.expand_max_tokens == 4000


def _acl_terms(search: dict[str, Any]) -> list[str]:
    clauses = search["knn"]["filter"] if "knn" in search else search["query"]["bool"]["filter"]
    return next(c["terms"]["acl_groups"] for c in clauses if "terms" in c)


def test_per_request_acl_groups_override_the_pipeline() -> None:
    # Spec P0-09 AC5: the service passes each caller's groups per request (design s16-4).
    msearch = _msearch()
    pipeline = _pipeline(msearch, FakeCrossEncoder(SCORES))
    pipeline.retrieve("q", kb=KB, top_k=2, acl_groups=["team-a", "team-b"])
    bodies = msearch.calls[-1]["searches"][1::2]
    assert [_acl_terms(body) for body in bodies] == [["team-a", "team-b"]] * 2
    pipeline.retrieve("q", kb=KB, top_k=2)  # without: the pipeline's own groups
    assert [_acl_terms(body) for body in msearch.calls[-1]["searches"][1::2]] == [["public"]] * 2
    with pytest.raises(TypeError, match="list"):
        pipeline.retrieve("q", kb=KB, top_k=2, acl_groups="team-a")

    # get_context reads with the caller's groups: the pages are only open to "public".
    assert pipeline.get_context(f"{_page(3)}#0", "chunk").found is True
    hidden = pipeline.get_context(f"{_page(3)}#0", "chunk", acl_groups=["team-a"])
    assert hidden.found is False and hidden.chunks == []


def test_config_version_ignores_description() -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    plain = KnowledgeBaseConfig(kb=KB)
    described = KnowledgeBaseConfig(kb=KB, description="Hugging Face documentation")
    assert search_config_version(plain, settings, "e@1") == search_config_version(
        described, settings, "e@1"
    )
