"""Spec P2-03 AC5-AC7: the ablation table (variants against one baseline run, design s14-4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.config import KnowledgeBaseConfig
from rag.eval.questions import LabeledQuestion, load_question_set, save_question_set
from rag.eval.retrieval import Provenance, RunRecord, cli, run_retrieval_eval, write_run
from rag.eval.retrieval.ablation import ablation_table, render_ablation
from rag.eval.retrieval.compare import compare_runs
from rag.testing import FakeRetriever
from rag.types import Chunk, RetrievalResult, RouteHit

KB = "hf_docs"
N = 12
QIDS = [f"q{i}" for i in range(1, N + 1)]


def _page(i: int) -> str:
    return f"{KB}:p{i}.md"


def _result(*pages: str) -> RetrievalResult:
    chunks = [Chunk(chunk_id=f"{p}#0", doc_id=p, kb=KB, text="t") for p in pages]
    route = [
        RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r) for r, c in enumerate(chunks, 1)
    ]
    return RetrievalResult(request_id="r", kb=KB, chunks=chunks, routes={"bm25": route})


def _run(tmp_path: Path, ranks: dict[int, int | None], *, label: str, minute: int) -> RunRecord:
    """`ranks[i]`: where question i's page sits in the final list (None or absent: missing)."""
    questions = [
        LabeledQuestion(qid=qid, question=f"question {qid}?", relevant_pages=[_page(i)])
        for i, qid in enumerate(QIDS, start=1)
    ]
    save_question_set(questions, tmp_path / "set.jsonl")
    results = {}
    for i in range(1, N + 1):
        pages = [f"{KB}:noise{i}-{j}.md" for j in range(3)]
        if ranks.get(i) is not None:
            pages.insert(ranks[i] - 1, _page(i))
        results[f"question q{i}?"] = _result(*pages)
    cfg = KnowledgeBaseConfig.model_validate({"kb": KB, "retrieval": {"rerank_n": 10}})
    return run_retrieval_eval(
        FakeRetriever(results),
        load_question_set(tmp_path / "set.jsonl", kb=KB),
        kb_config=cfg,
        ks=(1, 3, 5),
        label=label,
        provenance=Provenance(git_commit="a" * 40),
        now=datetime(2026, 10, 19, 10, minute, tzinfo=UTC),
    )


@pytest.fixture
def runs(tmp_path: Path) -> tuple[RunRecord, dict[str, RunRecord]]:
    # Baseline: 6 questions at rank 1, q7 at rank 2, the rest missed.
    base = _run(tmp_path, {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2}, label="base", minute=0)
    # "better": every question at rank 1: q7 and q8-q12 are better, none worse (6 / 0 / 6).
    better = _run(tmp_path, {i: 1 for i in range(1, 13)}, label="better", minute=1)
    # "mixed": q1 lost, q8 found: 1 better, 1 worse, 10 unchanged.
    mixed = _run(tmp_path, {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 1}, label="mixed", minute=2)
    return base, {"better": better, "mixed": mixed}


def test_rows_say_better_worse_unchanged_with_the_repo_tests(
    runs: tuple[RunRecord, dict[str, RunRecord]],
) -> None:
    base, variants = runs
    table = ablation_table(base, variants, metrics=["hit@1", "mrr@5"])
    assert table.n == N
    rows = {(r.variant, r.metric): r for r in table.rows}
    assert list(rows) == [
        ("better", "hit@1"),
        ("better", "mrr@5"),
        ("mixed", "hit@1"),
        ("mixed", "mrr@5"),
    ]
    better = rows["better", "hit@1"]
    assert (better.better, better.worse, better.unchanged) == (6, 0, 6)
    assert better.test == "McNemar exact"
    # Same numbers as `compare` gives for the same two runs.
    c = {m.metric: m for m in compare_runs(base, variants["better"]).metrics}["hit@1"]
    assert better.p_value == pytest.approx(c.p_value)
    assert (better.mean_baseline, better.mean_variant) == (
        pytest.approx(c.mean_a),
        pytest.approx(c.mean_b),
    )
    assert better.verdict == "better, reliable"
    mixed = rows["mixed", "hit@1"]
    assert (mixed.better, mixed.worse, mixed.unchanged) == (1, 1, 10)
    assert mixed.verdict == "no reliable difference"
    assert rows["better", "mrr@5"].test.startswith("permutation")


def test_default_metrics_and_bad_metric(runs: tuple[RunRecord, dict[str, RunRecord]]) -> None:
    base, variants = runs
    table = ablation_table(base, variants)
    assert [r.metric for r in table.rows if r.variant == "mixed"] == [
        "hit@1",
        "hit@5",
        "mrr@5",
        "ndcg@5",
    ]
    with pytest.raises(ValueError, match="hit@10"):
        ablation_table(base, variants, metrics=["hit@10"])
    with pytest.raises(ValueError, match="at least one variant"):
        ablation_table(base, {})


def test_render_has_intervals_latency_resolution_and_small_difference_warning(
    runs: tuple[RunRecord, dict[str, RunRecord]],
) -> None:
    base, variants = runs
    text = render_ablation(ablation_table(base, variants), baseline_name="both routes")
    assert "both routes" in text and "better" in text and "mixed" in text
    assert "Resolution: with 12 questions" in text  # what 12 questions can tell apart
    assert "| variant | metric |" in text
    assert (
        "| better | hit@1 | 0.500 | 1.000 | 6 | 0 | 6 | McNemar exact | 0.0312 | better, reliable |"
        in text
    )
    # Wilson intervals of Hit@k and the latency of every run.
    assert "Wilson" in text and "P50" in text and "P95" in text
    # A change smaller than what the question count can show is called out as such.
    assert "below the smallest difference 12 questions can show" in text


def test_ablate_cli_prints_and_writes(
    tmp_path: Path,
    runs: tuple[RunRecord, dict[str, RunRecord]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    base, variants = runs
    reports = tmp_path / "reports"
    write_run(base, public=True, reports_root=reports)
    for run in variants.values():
        write_run(run, public=True, reports_root=reports)
    out = tmp_path / "ablation.md"
    argv = [
        "ablate",
        base.run.run_id,
        "--variant",
        f"better={variants['better'].run.run_id}",
        "--variant",
        f"mixed={variants['mixed'].run.run_id}",
        "--baseline-name",
        "both routes",
        "--reports-dir",
        str(reports),
        "--metrics",
        "hit@1,mrr@5",
        "--out",
        str(out),
    ]
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert printed == out.read_text(encoding="utf-8")
    assert "| mixed | hit@1 |" in printed and "| mixed | ndcg@5 |" not in printed
    # A variant given without NAME=RUN is a usage error: exit 2, one line on stderr.
    assert (
        cli.main(
            ["ablate", base.run.run_id, "--variant", "nonsense", "--reports-dir", str(reports)]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "--variant" in err
