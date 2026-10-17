"""Spec P0-08 AC15-AC16: the real bge-reranker-base and the pipeline on the 50-doc index.

Needs the reranker in the local Hugging Face cache (read with `local_files_only`; skipped when it
is missing) and, for the index tests, the compose Elasticsearch with the 50-document indices
that the ingest job builds (`kb_hf_docs` with real vectors, read only, and `fake_kb_hf_docs`
with fake vectors, so no query embedding costs money). Loading the model takes about a minute
(imports included); it is loaded once for the module, as a service would at start.

`test_real_model_latency` prints the load time and the per-candidate latency for 8, 20 and 30
candidates (run with `-s`); its assertions are loose, the numbers go into the spec handoff.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from elasticsearch import Elasticsearch

from rag.config import ServiceSettings, load_kb_configs, repo_path
from rag.eval.questions import load_question_set
from rag.index.manage import aliased_indices, index_embed_model
from rag.index.names import KbIndexNames
from rag.retrieve import rerank as rerank_module
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline
from rag.retrieve.rerank import (
    CrossEncoderScorer,
    RerankRunner,
    get_rerank_runner,
    load_cross_encoder,
    rerank_text,
)

pytestmark = pytest.mark.integration

QUESTION_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"
KNOWN_QID = "huggingface_doc_qa_eval:025"
KNOWN_PAGE = "hf_docs:huggingface/hf-endpoints-documentation/blob/main/docs/source/guides/logs.mdx"
LONG_PAGE = "hf_docs:huggingface/transformers/blob/main/docs/source/en/perf_train_tpu_tf.md"
UNRELATED_QUESTION = "What is the boiling point of water at sea level in degrees Fahrenheit?"


@pytest.fixture(scope="module")
def settings() -> ServiceSettings:
    return ServiceSettings(embed_backend="fake")


@pytest.fixture(scope="module")
def scorer(settings: ServiceSettings) -> Iterator[CrossEncoderScorer]:
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(settings.reranker_model, "config.json"), str):
        pytest.skip(f"{settings.reranker_model} is not in the local Hugging Face cache")
    rerank_module.reset_rerank_runners()
    scorer = load_cross_encoder(settings)
    assert isinstance(scorer, CrossEncoderScorer)
    yield scorer
    rerank_module.reset_rerank_runners()


def _question(qid: str) -> str:
    questions = load_question_set(repo_path(QUESTION_SET), kb="hf_docs").questions
    return next(q.question for q in questions if q.qid == qid)


def _page_chunks(es: Elasticsearch, alias: str, doc_id: str) -> list[dict[str, Any]]:
    resp = es.search(
        index=alias,
        query={"term": {"doc_id": doc_id}},
        size=200,
        sort=[{"chunk_index": "asc"}],
        source={"includes": ["chunk_id", "chunk_index", "title_path", "content"]},
    )
    return [h["_source"] for h in resp["hits"]["hits"]]


def _require_alias(es: Elasticsearch, prefix: str, embed_model: str | None = None) -> str:
    names = KbIndexNames("hf_docs", prefix=prefix)
    indices = aliased_indices(es, names)
    if not indices:
        pytest.skip(f"{names.alias} not built; run the ingest job with --limit 50")
    if embed_model is not None and index_embed_model(es, indices[0]) != embed_model:
        pytest.skip(f"{names.alias} holds vectors of another model")
    return names.alias


def test_real_scores_are_sigmoid_in_unit_interval(scorer: CrossEncoderScorer) -> None:
    import torch

    pairs = [
        ("How do I install transformers?", "Installation\nRun pip install transformers."),
        ("How do I install transformers?", "Weather\nIt rains in Hamburg."),
        (UNRELATED_QUESTION, "Datasets\nLoad a dataset with load_dataset."),
        ("如何安装 transformers", "安装\n运行 pip install transformers"),
    ]
    scores = scorer.predict(pairs)
    raw = scorer.model.predict(
        [list(p) for p in pairs],
        batch_size=len(pairs),
        show_progress_bar=False,
        convert_to_numpy=True,
        activation_fn=torch.nn.Identity(),
    )
    assert all(0.0 <= s <= 1.0 for s in scores)
    for score, logit in zip(scores, raw, strict=True):
        assert score == pytest.approx(1.0 / (1.0 + math.exp(-float(logit))), abs=1e-6)
    assert scores[0] > 0.5 > scores[1]


def test_known_relevant_pair_high_unrelated_low(
    scorer: CrossEncoderScorer, es: Elasticsearch
) -> None:
    alias = _require_alias(es, "")
    chunks = _page_chunks(es, alias, KNOWN_PAGE)
    if not chunks:
        pytest.skip(f"{KNOWN_PAGE} is not in {alias}")
    text = rerank_text(chunks[0]["title_path"], chunks[0]["content"])
    relevant, unrelated = scorer.predict([(_question(KNOWN_QID), text), (UNRELATED_QUESTION, text)])
    print(f"\nrelevant pair {relevant:.6f}, unrelated pair {unrelated:.6f}")
    assert relevant > 0.9
    assert unrelated < 0.01


def _texts(es: Elasticsearch, alias: str, n: int) -> list[str]:
    """The first `n` chunks of the index by chunk id, as reranker passages."""
    resp = es.search(
        index=alias,
        query={"match_all": {}},
        size=n,
        sort=[{"chunk_id": "asc"}],
        source={"includes": ["title_path", "content"]},
    )
    texts = [
        rerank_text(h["_source"]["title_path"], h["_source"]["content"])
        for h in resp["hits"]["hits"]
    ]
    if len(texts) < n:
        pytest.skip(f"{alias} has fewer than {n} chunks")
    return texts


def test_real_model_latency(
    scorer: CrossEncoderScorer, es: Elasticsearch, settings: ServiceSettings
) -> None:
    import torch

    texts = _texts(es, _require_alias(es, ""), 30)
    question = _question(KNOWN_QID)
    runner = RerankRunner(scorer, batch_size=settings.rerank_batch_size, max_concurrency=1)
    rows = []
    try:
        runner.warm_up()
        for n in (8, 20, 30):
            times = []
            for _ in range(3):
                began = time.perf_counter()
                run = runner.rerank(question, texts[:n])
                times.append((time.perf_counter() - began) * 1000.0)
                assert len(run.scores) == n and not run.timed_out
            median = statistics.median(times)
            rows.append((n, median, median / n))
    finally:
        runner.close()
    print(
        f"\nreranker load {scorer.load_s:.1f} s, warm-up batch {runner.warmup_s:.2f} s, "
        f"torch threads {torch.get_num_threads()} (setting {settings.reranker_torch_threads}), "
        f"slots {settings.rerank_max_concurrency}, batch {settings.rerank_batch_size}"
    )
    for n, median, per in rows:
        print(f"  {n:2d} candidates: median {median:7.1f} ms, {per:6.1f} ms per candidate")
    assert all(per < 2000 for _, _, per in rows)
    assert torch.get_num_threads() == settings.reranker_torch_threads


def test_two_threads_score_like_a_sequential_run(
    scorer: CrossEncoderScorer, es: Elasticsearch, settings: ServiceSettings
) -> None:
    """Two reranks at once (two slots, each on the configured torch threads) give the same
    scores as one after the other; prints both wall times."""
    texts = _texts(es, _require_alias(es, ""), 16)
    questions = [_question(KNOWN_QID), "How do I train a TensorFlow model on a TPU?"]
    runner = RerankRunner(scorer, batch_size=settings.rerank_batch_size, max_concurrency=2)
    try:
        runner.warm_up()
        began = time.perf_counter()
        sequential = [runner.rerank(q, texts).scores for q in questions]
        sequential_s = time.perf_counter() - began

        results: dict[int, list[float]] = {}

        def work(i: int) -> None:
            results[i] = runner.rerank(questions[i], texts).scores

        threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
        began = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        concurrent_s = time.perf_counter() - began
    finally:
        runner.close()
    for i in range(2):
        assert len(results[i]) == len(texts)
        assert results[i] == pytest.approx(sequential[i], abs=1e-5)
    pairs = 2 * len(texts)
    print(
        f"\n2 x {len(texts)} candidates: sequential {sequential_s:.1f} s "
        f"({sequential_s * 1000 / pairs:.0f} ms per candidate), two threads {concurrent_s:.1f} s "
        f"({concurrent_s * 1000 / pairs:.0f} ms per candidate of throughput)"
    )


def test_pipeline_ranks_known_page_first(
    scorer: CrossEncoderScorer, es: Elasticsearch, settings: ServiceSettings
) -> None:
    _require_alias(es, "fake_", f"fake-embedding@{settings.embed_dim}")
    cfg = load_kb_configs(repo_path(settings.kb_dir))["hf_docs"]
    pipeline = make_search_pipeline(
        cfg, settings, acl_groups=["public"], index_prefix="fake_", es=es
    )
    assert pipeline.postprocessors[1].runner is get_rerank_runner(settings)
    try:
        result = pipeline.retrieve(_question(KNOWN_QID), kb="hf_docs", top_k=8, deadline_ms=60_000)
    finally:
        pipeline.close()
    if KNOWN_PAGE not in {h.doc_id for hits in result.routes.values() for h in hits}:
        pytest.skip(f"{KNOWN_PAGE} is not in fake_kb_hf_docs")

    assert result.rerank_timeout is False
    assert result.chunks and result.chunks[0].doc_id == KNOWN_PAGE
    assert result.chunks[0].rerank_score is not None and result.chunks[0].rerank_score > 0.9
    assert all(0.0 <= h.score <= 1.0 for h in result.rerank if h.score is not None)
    assert 0 < len(result.rerank) <= cfg.retrieval.rerank_n
    assert result.threshold is not None and result.threshold.passed is True
    assert all(c.rerank_score >= cfg.retrieval.threshold for c in result.chunks if c.rerank_score)
    assert result.config_version
    print(f"\npipeline timings (ms): { {k: round(v, 1) for k, v in result.timings_ms.items()} }")


def test_context_on_real_index(
    scorer: CrossEncoderScorer, es: Elasticsearch, settings: ServiceSettings
) -> None:
    alias = _require_alias(es, "fake_", f"fake-embedding@{settings.embed_dim}")
    page = _page_chunks(es, alias, LONG_PAGE)
    if len(page) < 6:
        pytest.skip(f"{LONG_PAGE} is not in {alias}")
    cfg = load_kb_configs(repo_path(settings.kb_dir))["hf_docs"]
    expanding = cfg.model_copy(
        update={"retrieval": cfg.retrieval.model_copy(update={"expand": "neighbors"})}
    )
    pipeline: SearchPipeline = make_search_pipeline(
        expanding, settings, acl_groups=["public"], index_prefix="fake_", es=es
    )
    try:
        # get_context: neighbours of chunk 5, then a chunk id that no longer exists.
        context = pipeline.get_context(f"{LONG_PAGE}#5")
        assert [c.chunk_id for c in context.chunks] == [f"{LONG_PAGE}#{i}" for i in (4, 5, 6)]
        assert context.found and not context.updated
        gone = pipeline.get_context(f"{LONG_PAGE}#9999")
        assert gone.updated is True and gone.found is True
        assert [c.chunk_id for c in gone.chunks] == [c["chunk_id"] for c in page][
            : len(gone.chunks)
        ]
        assert gone.chunks[0].chunk_id == f"{LONG_PAGE}#0"

        # Expansion in the pipeline: neighbours of the same page, never a returned chunk.
        result = pipeline.retrieve(
            "How do I train a TensorFlow model on a TPU?", kb="hf_docs", top_k=5, deadline_ms=60_000
        )
    finally:
        pipeline.close()
    returned = {c.chunk_id for c in result.chunks}
    pieces = [(c, p) for c in result.chunks for p in c.context]
    assert pieces, "no context added"
    for chunk, piece in pieces:
        assert piece.chunk_id.rsplit("#", 1)[0] == chunk.doc_id
        assert abs(piece.chunk_index - chunk.metadata["chunk_index"]) == 1
        assert piece.chunk_id not in returned
    assert len({p.chunk_id for _, p in pieces}) == len(pieces)
    assert "expand" in result.timings_ms
