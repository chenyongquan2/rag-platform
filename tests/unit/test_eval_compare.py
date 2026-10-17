"""Spec P0-10 AC14: comparing two eval runs question by question (design s14-4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.config import KnowledgeBaseConfig
from rag.eval.questions import LabeledQuestion, load_question_set, save_question_set
from rag.eval.retrieval import Provenance, RunRecord, cli, run_retrieval_eval, write_run
from rag.eval.retrieval.compare import compare_runs, render_comparison
from rag.eval.stats import mcnemar_exact
from rag.testing import FakeRetriever
from rag.types import Chunk, RetrievalResult, RouteHit

KB = "hf_docs"
QIDS = [f"q{i}" for i in range(1, 9)]


def _page(i: int) -> str:
    return f"{KB}:p{i}.md"


def _result(*pages: str) -> RetrievalResult:
    chunks = [Chunk(chunk_id=f"{p}#0", doc_id=p, kb=KB, text="t") for p in pages]
    route = [
        RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r) for r, c in enumerate(chunks, 1)
    ]
    return RetrievalResult(request_id="r", kb=KB, chunks=chunks, routes={"bm25": route})


def _question_set(tmp_path: Path):
    questions = [
        LabeledQuestion(qid=qid, question=f"question {qid}?", relevant_pages=[_page(i)])
        for i, qid in enumerate(QIDS, start=1)
    ]
    save_question_set(questions, tmp_path / "set.jsonl")
    return load_question_set(tmp_path / "set.jsonl", kb=KB)


def _run(tmp_path: Path, ranks: dict[int, int | None], *, label: str, rerank_n: int, minute: int):
    """`ranks[i]`: where question i's relevant page sits in the final list (None: missing)."""
    results = {}
    for i in range(1, 9):
        pages = [f"{KB}:noise{i}-{j}.md" for j in range(3)]
        if ranks.get(i) is not None:
            pages.insert(ranks[i] - 1, _page(i))
        results[f"question q{i}?"] = _result(*pages)
    cfg = KnowledgeBaseConfig.model_validate(
        {"kb": KB, "retrieval": {"rerank_n": rerank_n, "final_k": 8}}
    )
    return run_retrieval_eval(
        FakeRetriever(results),
        _question_set(tmp_path),
        kb_config=cfg,
        ks=(1, 3),
        label=label,
        provenance=Provenance(git_commit="a" * 40 if minute == 0 else "b" * 40),
        now=datetime(2026, 10, 17, 10, minute, tzinfo=UTC),
    )


@pytest.fixture
def runs(tmp_path: Path) -> tuple[RunRecord, RunRecord]:
    # A: q1, q2 at rank 1, the rest missed. B: q2 drops to rank 2, q3-q7 found at rank 1.
    a = _run(tmp_path, {1: 1, 2: 1}, label="baseline", rerank_n=30, minute=0)
    b = _run(
        tmp_path,
        {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        label="rerank20",
        rerank_n=20,
        minute=1,
    )
    return a, b


def test_compare_runs_statistics(runs: tuple[RunRecord, RunRecord]) -> None:
    a, b = runs
    c = compare_runs(a, b, metric="mrr@3")
    assert c.n == 8
    assert c.same_question_set is True
    by_metric = {m.metric: m for m in c.metrics}
    assert list(by_metric) == [
        "hit@1",
        "hit@3",
        "mrr@3",
        "ndcg@1",
        "ndcg@3",
        "recall@1",
        "recall@3",
    ]

    hit1 = by_metric["hit@1"]
    assert hit1.test == "mcnemar"
    assert (hit1.changes.improved, hit1.changes.worsened, hit1.changes.unchanged) == (5, 1, 2)
    assert hit1.p_value == pytest.approx(mcnemar_exact(5, 1))
    assert hit1.mean_a == pytest.approx(2 / 8) and hit1.mean_b == pytest.approx(6 / 8)
    assert by_metric["hit@3"].p_value == pytest.approx(0.0625)

    mrr = by_metric["mrr@3"]
    assert mrr.test == "permutation"
    assert mrr.exact is True and mrr.iterations == 64
    assert mrr.p_value == pytest.approx(4 / 64)
    assert mrr.reliable(0.05) is False

    assert [q.qid for q in c.improved] == ["q3", "q4", "q5", "q6", "q7"]
    assert [q.qid for q in c.worsened] == ["q2"]
    q2 = c.worsened[0]
    assert (q2.rank_a, q2.rank_b, q2.value_a, q2.value_b) == (1, 2, 1.0, 0.5)
    assert q2.top_b[:2] == [f"{KB}:noise2-0.md", _page(2)]
    assert "kb_config.retrieval.rerank_n: 30 -> 20" in c.differences
    assert any(d.startswith("git_commit: aaaaaaaaaaaa") for d in c.differences)

    # Per-route comparison, and pairing by qid when the runs differ.
    assert compare_runs(a, b, list_name="bm25").n == 8
    with pytest.raises(ValueError, match="vector"):
        compare_runs(a, b, list_name="vector")
    with pytest.raises(ValueError, match="mrr@10"):
        compare_runs(a, b, metric="mrr@10")
    smaller = b.model_copy(
        update={
            "questions": b.questions[:-1],
            "run": b.run.model_copy(
                update={"question_set": b.run.question_set.model_copy(update={"sha256": "x"})}
            ),
        }
    )
    partial = compare_runs(a, smaller)
    assert partial.n == 7
    assert partial.only_in_a == ["q8"]
    assert partial.same_question_set is False
    assert "different question sets" in render_comparison(partial)


def test_compare_cli_prints_tests_and_changed_questions(
    tmp_path: Path, runs: tuple[RunRecord, RunRecord], capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = runs
    reports = tmp_path / "reports"
    write_run(a, public=True, reports_root=reports)
    out_b = write_run(b, public=False, reports_root=reports)
    md = tmp_path / "compare.md"
    argv = [
        "compare",
        a.run.run_id,  # found by run id under public/
        str(out_b),  # or by directory
        "--reports-dir",
        str(reports),
        "--metric",
        "mrr@3",
        "--out",
        str(md),
    ]
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert printed == md.read_text(encoding="utf-8")
    assert f"{a.run.run_id} -> {b.run.run_id}" in printed
    assert "| hit@1 | 0.250 | 0.750 | +0.500 | 5 | 1 | 2 | McNemar exact | 0.219 | no |" in printed
    assert "| mrr@3 |" in printed and "permutation (exact, 64 sign patterns)" in printed
    assert "Improved on mrr@3: 5 questions" in printed
    assert "Worsened on mrr@3: 1 question" in printed
    assert "`q2` question q2? mrr@3 1.000 -> 0.500, first relevant rank 1 -> 2" in printed
    assert "kb_config.retrieval.rerank_n: 30 -> 20" in printed
    assert "Resolution: with 8 questions" in printed
