"""Spec P0-10 AC15: rerank latency and nDCG for rerank_n 10, 20, 30 (design s10-1, s14-6)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.eval.questions import LabeledQuestion, load_question_set, save_question_set
from rag.eval.retrieval import Provenance, Reranker
from rag.eval.retrieval.rerank_curve import (
    render_rerank_curve,
    rerank_latency_curve,
    write_rerank_curve,
)
from rag.testing import FakeRetriever
from rag.types import Chunk, RetrievalResult

KB = "hf_docs"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeReranker:
    """Moves the relevant pages to the front; costs 2 ms of fake time per candidate."""

    def __init__(self, clock: FakeClock, relevant: set[str]) -> None:
        self.clock, self.relevant = clock, relevant
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, question: str, candidates: Sequence[Chunk]) -> Sequence[Chunk]:
        self.calls.append((question, [c.chunk_id for c in candidates]))
        self.clock.now += 0.002 * len(candidates)
        return sorted(candidates, key=lambda c: c.doc_id not in self.relevant)


def _candidates(qid: str, relevant_at: int) -> RetrievalResult:
    """30 fused candidates; the relevant page `<qid>.md` sits at fusion rank `relevant_at`."""
    docs = [f"{KB}:{qid}-noise{i}.md" for i in range(29)]
    docs.insert(relevant_at - 1, f"{KB}:{qid}.md")
    chunks = [Chunk(chunk_id=f"{d}#0", doc_id=d, kb=KB, text="t") for d in docs]
    return RetrievalResult(request_id="r", kb=KB, chunks=chunks)


def test_rerank_curve_measures_each_n(tmp_path: Path) -> None:
    save_question_set(
        [
            LabeledQuestion(qid="q1", question="one?", relevant_pages=[f"{KB}:q1.md"]),
            LabeledQuestion(qid="q2", question="two?", relevant_pages=[f"{KB}:q2.md"]),
        ],
        tmp_path / "set.jsonl",
    )
    qs = load_question_set(tmp_path / "set.jsonl", kb=KB)
    retriever = FakeRetriever({"one?": _candidates("q1", 25), "two?": _candidates("q2", 5)})
    clock = FakeClock()
    reranker = FakeReranker(clock, {f"{KB}:q1.md", f"{KB}:q2.md"})
    assert isinstance(reranker, Reranker)

    with pytest.warns(UserWarning, match="final list"):  # k = 10 is reported as @8
        curve = rerank_latency_curve(
            retriever,
            reranker,
            qs,
            kb=KB,
            rerank_ns=(10, 20, 30),
            final_k=8,
            ks=(1, 5, 10),
            warmup=1,
            clock=clock,
            provenance=Provenance(models={"reranker": "fake"}),
        )
    assert curve.ks == [1, 5, 8]
    # Candidates are fetched once per question, as many as the largest rerank_n.
    assert retriever.calls == [("one?", KB, 30), ("two?", KB, 30)]
    # One unmeasured warm-up call, then each n in turn per question, always the first n
    # candidates in fusion order.
    assert [(q, len(ids)) for q, ids in reranker.calls] == [
        ("one?", 30),
        ("one?", 10),
        ("one?", 20),
        ("one?", 30),
        ("two?", 10),
        ("two?", 20),
        ("two?", 30),
    ]
    assert reranker.calls[1][1] == [c.chunk_id for c in _candidates("q1", 25).chunks[:10]]

    fusion, n10, n20, n30 = curve.points
    assert fusion.rerank_n is None and fusion.p50_ms is None
    assert fusion.means["mrr@8"] == pytest.approx((0 + 1 / 5) / 2)  # top 8 in fusion order
    assert [p.rerank_n for p in (n10, n20, n30)] == [10, 20, 30]
    assert [p.p50_ms for p in (n10, n20, n30)] == pytest.approx([20.0, 40.0, 60.0])
    assert n30.p95_ms == pytest.approx(60.0)
    assert n10.ms_per_candidate == pytest.approx(2.0)
    assert n10.mean_candidates == 10
    # q1's page is at fusion rank 25: only rerank_n = 30 lets the reranker see it.
    assert n10.means["mrr@8"] == pytest.approx(0.5)
    assert n20.means["mrr@8"] == pytest.approx(0.5)
    assert n30.means["ndcg@8"] == pytest.approx(1.0)
    assert n30.questions == 2

    table = render_rerank_curve(curve)
    assert "| rerank_n | candidates | P50 ms | P95 ms | mean ms | ms per candidate |" in table
    assert "| hit@1 | hit@5 | hit@8 | mrr@8 | ndcg@8 |" in table
    assert "| fusion (no rerank) | 30.0 | - | - | - | - |" in table
    assert (
        "| 30 | 30.0 | 60.0 | 60.0 | 60.0 | 2.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |" in table
    )

    out = write_rerank_curve(curve, tmp_path / "run")
    assert sorted(p.name for p in out.iterdir()) == ["rerank_curve.json", "rerank_curve.md"]
    data = json.loads((out / "rerank_curve.json").read_text(encoding="utf-8"))
    assert data["question_set"]["sha256"] == qs.sha256
    assert data["provenance"]["models"] == {"reranker": "fake"}
    assert [p["rerank_n"] for p in data["points"]] == [None, 10, 20, 30]
