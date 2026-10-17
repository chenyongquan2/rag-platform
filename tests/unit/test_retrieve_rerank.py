"""Spec P0-08 AC1-AC7: BatchedTimeoutRerank, RerankRunner, model loading (design s10-1, s10-2)."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.config import ServiceSettings
from rag.config import settings as settings_module
from rag.eval.questions import LabeledQuestion, QuestionSet
from rag.eval.retrieval import Reranker, rerank_latency_curve
from rag.retrieve import rerank as rerank_module
from rag.retrieve.deadline import Deadline
from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import (
    BatchedTimeoutRerank,
    ChunkReranker,
    RerankRunner,
    RerankScaleError,
    cross_encoder_kwargs,
    get_rerank_runner,
    load_cross_encoder,
    rerank_text,
)
from rag.retrieve.search import node_to_chunk
from rag.testing import FakeRetriever
from rag.testing.rerank import FakeCrossEncoder, ManualClock
from rag.types import Chunk, RetrievalResult, RouteHit

KB = "hf_docs"


def _node(i: int, *, score: float | None = None, **meta: Any) -> NodeWithScore:
    doc = f"hf_docs:page{i}.md"
    metadata = {
        "doc_id": doc,
        "kb": KB,
        "title_path": f"Page {i} > Part",
        "source_rel": f"page{i}.md",
        "source_url": f"https://example.org/page{i}.md",
        "doc_type": "markdown",
        "language": "en",
        "chunk_index": 0,
        "route_ranks": {"bm25": i + 1},
        **meta,
    }
    node = TextNode(id_=f"{doc}#0", text=f"body {i}", metadata=metadata)
    return NodeWithScore(node=node, score=score if score is not None else 1.0 / (61 + i))


def _query(question: str = "q", deadline: Deadline | None = None) -> HybridQuery:
    return HybridQuery(
        question,
        acl_groups=("public",),
        deadline=deadline or Deadline.after_ms(60_000),
    )


def _runner(scorer: FakeCrossEncoder, **kw: Any) -> RerankRunner:
    kw.setdefault("batch_size", 8)
    kw.setdefault("max_concurrency", 2)
    return RerankRunner(scorer, **kw)


def test_rerank_text_is_question_title_path_and_body() -> None:
    scorer = FakeCrossEncoder()
    post = BatchedTimeoutRerank(runner=_runner(scorer), top_n=30)
    post.postprocess_nodes([_node(0), _node(1)], query_bundle=_query("how to install?"))

    pairs = [pair for batch in scorer.batches for pair in batch]
    assert pairs == [
        ("how to install?", "Page 0 > Part\nbody 0"),
        ("how to install?", "Page 1 > Part\nbody 1"),
    ]
    # Only the question, the title path and the body: no page path, URL, doc type or language.
    for _, text in pairs:
        for leaked in ("page0.md", "https://", "markdown", "en\n"):
            assert leaked not in text
    assert rerank_text("", "only body") == "only body"
    assert rerank_text(["A", "B"], "body") == "A > B\nbody"


def test_rerank_batches_of_eight_and_sorts_by_rerank_score() -> None:
    # 25 candidates, rerank_n 20: batches of 8, 8, 4; candidates 20-24 are not reranked.
    scores = {f"Page {i} > Part\nbody {i}": (i % 5) / 10 for i in range(25)}
    scorer = FakeCrossEncoder(scores)
    post = BatchedTimeoutRerank(runner=_runner(scorer), top_n=20)
    nodes = [_node(i) for i in range(25)]
    fusion = {n.node.node_id: n.score for n in nodes}
    query = _query()

    out = post.postprocess_nodes(nodes, query_bundle=query)

    assert [len(b) for b in scorer.batches] == [8, 8, 4]
    assert len(out) == 20
    got = [n.node.metadata["rerank_score"] for n in out]
    assert got == sorted(got, reverse=True)
    # Ties keep the fusion order: score 0.4 belongs to candidates 4, 9, 14, 19 in that order.
    assert [n.node.node_id for n in out[:4]] == [f"hf_docs:page{i}.md#0" for i in (4, 9, 14, 19)]
    # The fusion score stays in NodeWithScore.score; the rerank score is separate.
    assert all(n.score == fusion[n.node.node_id] for n in out)
    chunk = node_to_chunk(out[0], KB)
    assert chunk.rerank_score == pytest.approx(0.4)
    assert chunk.score == pytest.approx(1 / 65)
    assert "rerank_score" not in chunk.metadata

    trace = query.trace
    assert trace.rerank_timeout is False
    assert trace.rerank_candidates == 20
    assert [h.chunk_id for h in trace.rerank] == [n.node.node_id for n in out]
    assert trace.rerank[0] == RouteHit(
        chunk_id="hf_docs:page4.md#0", doc_id="hf_docs:page4.md", rank=1, score=0.4
    )
    assert "rerank_queue" in trace.timings_ms


def test_rerank_stops_at_deadline_and_appends_unscored_in_fusion_order() -> None:
    clock = ManualClock()
    # Each pair takes 0.1 s, a batch of 8 takes 0.8 s. With 2.0 s: batch 1 ends at 0.8 s
    # (1.2 s left, a batch fits), batch 2 ends at 1.6 s (0.4 s left: the next one would not fit).
    scores = {f"Page {i} > Part\nbody {i}": i / 100 for i in range(30)}
    scorer = FakeCrossEncoder(scores, clock=clock, per_pair_s=0.1)
    post = BatchedTimeoutRerank(runner=_runner(scorer), top_n=30)
    nodes = [_node(i) for i in range(30)]
    query = _query(deadline=Deadline.after_ms(2000, clock=clock))

    out = post.postprocess_nodes(nodes, query_bundle=query)

    assert [len(b) for b in scorer.batches] == [8, 8]
    assert query.trace.rerank_timeout is True
    assert len(out) == 30
    scored, unscored = out[:16], out[16:]
    assert [n.node.metadata["rerank_score"] for n in scored] == [i / 100 for i in range(15, -1, -1)]
    assert [n.node.node_id for n in unscored] == [f"hf_docs:page{i}.md#0" for i in range(16, 30)]
    assert all("rerank_score" not in n.node.metadata for n in unscored)
    assert node_to_chunk(unscored[0], KB).rerank_score is None
    assert len(query.trace.rerank) == 16

    # A deadline that has already passed scores nothing: fusion order, flagged.
    expired = Deadline(at=clock() - 1, clock=clock)
    scorer.batches.clear()
    query = _query(deadline=expired)
    out = post.postprocess_nodes([_node(i) for i in range(5)], query_bundle=query)
    assert scorer.batches == []
    assert [n.node.node_id for n in out] == [f"hf_docs:page{i}.md#0" for i in range(5)]
    assert query.trace.rerank_timeout is True
    assert query.trace.rerank == []


def test_rerank_semaphore_limits_concurrency() -> None:
    scorer = FakeCrossEncoder(sleep_s=0.05)
    runner = _runner(scorer, max_concurrency=2)
    texts = [f"t{i}" for i in range(16)]
    errors: list[BaseException] = []

    def work() -> None:
        try:
            run = runner.rerank("q", texts, Deadline.after_ms(30_000))
            assert len(run.scores) == 16 and not run.timed_out
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert scorer.max_active == 2

    # A request that gets no slot before its deadline scores nothing and is flagged.
    gate = threading.Event()
    blocking = FakeCrossEncoder(gate=gate)
    runner = _runner(blocking, max_concurrency=1)
    holder = threading.Thread(target=runner.rerank, args=("q", ["a"], Deadline.after_ms(30_000)))
    holder.start()
    assert blocking.started.wait(5)
    began = time.perf_counter()
    run = runner.rerank("q", ["b", "c"], Deadline.after_ms(150))
    waited = time.perf_counter() - began
    gate.set()
    holder.join(timeout=10)
    assert run.timed_out is True
    assert run.scores == []
    assert run.batches == 0
    assert run.queue_ms >= 100
    assert waited < 1.0
    runner.close()


def test_rerank_queue_time_counts_against_deadline() -> None:
    # The holder keeps the only slot for 3 batches of ~0.12 s (~0.36 s). The second request has
    # 0.7 s in total and needs 5 batches: after the wait ~0.34 s are left, so it stops after 1
    # or 2 batches. Without counting the wait all 5 (0.6 s) would have fitted.
    scorer = FakeCrossEncoder(sleep_s=0.12)
    runner = _runner(scorer, max_concurrency=1)
    holder = threading.Thread(
        target=runner.rerank, args=("q", [f"h{i}" for i in range(24)], Deadline.after_ms(30_000))
    )
    holder.start()
    assert scorer.started.wait(5)

    run = runner.rerank("q", [f"t{i}" for i in range(40)], Deadline.after_ms(700))
    holder.join(timeout=10)

    assert run.queue_ms >= 250
    assert run.timed_out is True
    assert 8 <= len(run.scores) <= 24
    assert run.batches == len(run.scores) // 8
    runner.close()


def test_batch_that_would_not_finish_is_not_started() -> None:
    clock = ManualClock()
    scorer = FakeCrossEncoder(clock=clock, per_pair_s=0.1)
    # Seeded estimate (as after the warm-up batch at load): 0.1 s per pair, 0.8 s per batch.
    runner = _runner(scorer, max_concurrency=1, per_pair_s=0.1)
    run = runner.rerank("q", [f"t{i}" for i in range(16)], Deadline.after_ms(500, clock=clock))
    assert scorer.batches == []  # not even the first batch: it would end 0.3 s past the deadline
    assert run.timed_out is True and run.scores == [] and run.batches == 0
    # The slot was not held: the next request gets it at once and scores everything.
    run = runner.rerank("q", [f"t{i}" for i in range(16)], Deadline.after_ms(5000, clock=clock))
    assert len(run.scores) == 16 and not run.timed_out

    # The estimate is a moving average over requests: slower batches raise it.
    scorer.per_pair_s = 0.3
    runner.rerank("q", [f"t{i}" for i in range(8)], Deadline.after_ms(60_000, clock=clock))
    assert 0.1 < runner.per_pair_s < 0.3


def test_warm_up_seeds_the_estimate() -> None:
    scorer = FakeCrossEncoder(sleep_s=0.04)
    runner = _runner(scorer)
    assert runner.per_pair_s is None
    runner.warm_up()
    assert [len(b) for b in scorer.batches] == [8]  # one full batch of realistic pairs
    assert len(scorer.batches[0][0][1].split()) >= 150
    assert runner.per_pair_s is not None and runner.per_pair_s >= 0.04 / 8 * 0.5
    assert runner.warmup_s >= 0.03
    runner.close()


def test_scores_outside_unit_interval_are_refused() -> None:
    runner = _runner(FakeCrossEncoder({"Page 0 > Part\nbody 0": 3.2}))
    post = BatchedTimeoutRerank(runner=runner, top_n=30)
    with pytest.raises(RerankScaleError, match=r"\[0, 1\]"):
        post.postprocess_nodes([_node(0)], query_bundle=_query())
    with pytest.raises(RerankScaleError):
        _runner(FakeCrossEncoder(default=float("nan"))).rerank("q", ["x"])


def test_cross_encoder_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert cross_encoder_kwargs(settings) == {
        "device": "cpu",
        "max_length": 512,
        "local_files_only": True,
        "backend": "torch",
        "trust_remote_code": False,
    }
    loads: list[tuple[str, dict[str, Any]]] = []

    scorers: list[FakeCrossEncoder] = []

    def fake_load(model: str, **kwargs: Any) -> FakeCrossEncoder:
        loads.append((model, kwargs))
        scorers.append(FakeCrossEncoder())
        return scorers[-1]

    rerank_module.reset_rerank_runners()
    monkeypatch.setattr(rerank_module.CrossEncoderScorer, "load", staticmethod(fake_load))
    try:
        first = load_cross_encoder(settings)
        assert load_cross_encoder(settings) is first
        runner = get_rerank_runner(settings)
        assert get_rerank_runner(settings) is runner
        assert runner.scorer is first
        assert runner.batch_size == 8
        assert runner.max_concurrency == settings.rerank_max_concurrency
        threads = settings.reranker_torch_threads
        expected = {**cross_encoder_kwargs(settings), "torch_threads": threads}
        assert loads == [("BAAI/bge-reranker-base", expected)]
        # The process runner is warmed up once with a full batch, which seeds its estimate.
        assert [len(b) for b in scorers[0].batches] == [8]
        assert runner.per_pair_s is not None
    finally:
        rerank_module.reset_rerank_runners()


def test_rerank_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Torch threads default to half the physical cores; slots x threads = physical cores.
    monkeypatch.setattr(settings_module, "physical_cores", lambda: 8)
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_max_length == 512
    assert settings.rerank_batch_size == 8
    assert settings.reranker_torch_threads == 4
    assert settings.rerank_max_concurrency == 2
    monkeypatch.setenv("RERANKER_TORCH_THREADS", "2")
    assert ServiceSettings(_env_file=None).rerank_max_concurrency == 4  # type: ignore[call-arg]
    monkeypatch.setenv("RERANK_MAX_CONCURRENCY", "3")
    assert ServiceSettings(_env_file=None).rerank_max_concurrency == 3  # type: ignore[call-arg]
    monkeypatch.delenv("RERANKER_TORCH_THREADS")
    monkeypatch.delenv("RERANK_MAX_CONCURRENCY")
    monkeypatch.setattr(settings_module, "physical_cores", lambda: 1)
    one = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert (one.reranker_torch_threads, one.rerank_max_concurrency) == (1, 1)
    with pytest.raises(ValueError):
        RerankRunner(FakeCrossEncoder(), batch_size=0, max_concurrency=1)
    with pytest.raises(ValueError):
        RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=0)


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"hf_docs:p{i}.md#0",
        doc_id=f"hf_docs:p{i}.md",
        kb=KB,
        text=f"body {i}",
        title_path=["P", str(i)],
        score=1 / (61 + i),
    )


def test_chunk_reranker_follows_eval_protocol() -> None:
    scorer = FakeCrossEncoder({"P > 2\nbody 2": 0.9, "P > 0\nbody 0": 0.1}, default=0.5)
    reranker = ChunkReranker(_runner(scorer))
    assert isinstance(reranker, Reranker)

    out = list(reranker("q", [_chunk(0), _chunk(1), _chunk(2)]))
    assert [c.chunk_id for c in out] == ["hf_docs:p2.md#0", "hf_docs:p1.md#0", "hf_docs:p0.md#0"]
    assert [c.rerank_score for c in out] == [0.9, 0.5, 0.1]
    assert out[0].score == pytest.approx(1 / 63)  # fusion score kept

    question = LabeledQuestion(qid="q1", question="q", answer="a", relevant_pages=["hf_docs:p2.md"])
    qs = QuestionSet(path="mem.jsonl", sha256="0" * 64, questions=[question])
    retriever = FakeRetriever(
        {"q": RetrievalResult(request_id="r", kb=KB, chunks=[_chunk(i) for i in range(30)])}
    )
    curve = rerank_latency_curve(
        retriever, reranker, qs, kb=KB, rerank_ns=(10, 20), final_k=8, ks=(1, 3, 8)
    )
    by_n = {p.rerank_n: p for p in curve.points}
    assert by_n[None].means["hit@1"] == 0.0
    assert by_n[10].means["hit@1"] == 1.0
    assert by_n[20].mean_candidates == 20


def test_rerank_without_hybrid_query_has_no_deadline() -> None:
    scorer = FakeCrossEncoder({"Page 1 > Part\nbody 1": 0.8}, default=0.2)
    post = BatchedTimeoutRerank(runner=_runner(scorer), top_n=30)
    out = post.postprocess_nodes([_node(0), _node(1)], query_bundle=QueryBundle("q"))
    assert [n.node.node_id for n in out] == ["hf_docs:page1.md#0", "hf_docs:page0.md#0"]
    assert post.postprocess_nodes([], query_bundle=_query()) == []
