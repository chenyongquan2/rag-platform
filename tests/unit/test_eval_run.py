"""Spec P0-10 AC12-AC13: one retrieval eval run and its record (design s14-1, s14-4)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings, repo_path
from rag.eval.questions import LabeledQuestion, content_sha256, load_question_set, save_question_set
from rag.eval.retrieval import (
    Provenance,
    Retriever,
    collect_provenance,
    load_run,
    render_run_report,
    run_retrieval_eval,
    write_run,
)
from rag.eval.retrieval.metrics import metric_names
from rag.eval.retrieval.runner import git_env
from rag.testing import FakeRetriever
from rag.types import Chunk, RetrievalResult, RouteHit

KB = "hf_docs"
A, B, C, X, Y, Z = (f"{KB}:{name}.md" for name in "abcxyz")
NOW = datetime(2026, 10, 17, 9, 30, 0, tzinfo=UTC)


def _chunk(doc_id: str, n: int, score: float = 0.5) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, kb=KB, text="t", score=score)


def _route(*hits: tuple[str, int]) -> list[RouteHit]:
    return [
        RouteHit(chunk_id=f"{doc}#{n}", doc_id=doc, rank=rank)
        for rank, (doc, n) in enumerate(hits, start=1)
    ]


def _result(
    chunks: list[Chunk], routes: dict[str, list[RouteHit]], **kw: object
) -> RetrievalResult:
    return RetrievalResult(
        request_id="r",
        kb=KB,
        index_version="kb_hf_docs_v1",
        data_generation=7,
        chunks=chunks,
        routes=routes,
        **kw,
    )


@pytest.fixture
def question_set(tmp_path: Path):
    questions = [
        LabeledQuestion(qid="q1", question="one?", relevant_pages=[A], relevant_chunks=[f"{A}#1"]),
        LabeledQuestion(qid="q2", question="two?", relevant_pages=[B]),
        LabeledQuestion(qid="q3", question="three?", relevant_pages=[C]),
    ]
    path = tmp_path / "set.jsonl"
    save_question_set(questions, path)
    return load_question_set(path, kb=KB)


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever(
        {
            # Final list has 4 chunks; top_k=3 keeps 3. Relevant page A first at rank 2.
            "one?": _result(
                [_chunk(X, 0), _chunk(A, 1), _chunk(A, 2), _chunk(Y, 0)],
                {"bm25": _route((A, 1), (X, 0)), "vector": _route((X, 0), (Z, 0), (A, 2))},
            ),
            # The BM25 route failed: only the vector route, flagged degraded.
            "two?": _result([_chunk(B, 0)], {"vector": _route((B, 0))}, degraded=True),
            "three?": RuntimeError("es down"),
        }
    )


def _clock():
    # The harness reads the clock right before and right after each retrieve() call.
    return iter([0.0, 0.010, 1.0, 1.020, 2.0, 2.030]).__next__


def _run(retriever: Retriever, question_set, **kw):
    return run_retrieval_eval(
        retriever,
        question_set,
        kb_config=KnowledgeBaseConfig(kb=KB),
        top_k=3,
        ks=(1, 3),
        label="baseline",
        clock=_clock(),
        now=NOW,
        **kw,
    )


def test_run_scores_final_list_and_routes(retriever: FakeRetriever, question_set) -> None:
    assert isinstance(retriever, Retriever)
    record = _run(retriever, question_set)
    assert retriever.calls == [("one?", KB, 3), ("two?", KB, 3), ("three?", KB, 3)]
    assert record.run.run_id == "20261017T093000Z_hf_docs_baseline"
    assert record.run.lists == ["final", "bm25", "vector"]

    q1, q2, _ = record.questions
    assert [h.chunk_id for h in q1.hits] == [f"{X}#0", f"{A}#1", f"{A}#2"]
    assert [h.route_ranks for h in q1.hits] == [
        {"bm25": 2, "vector": 1},
        {"bm25": 1, "vector": None},
        {"bm25": None, "vector": 3},
    ]
    assert q1.metrics["final"]["page"]["hit@1"] == 0.0
    assert q1.metrics["final"]["page"]["mrr@3"] == 0.5
    assert q1.metrics["bm25"]["page"]["mrr@3"] == 1.0
    assert q1.metrics["vector"]["page"]["mrr@3"] == pytest.approx(1 / 3)
    assert q1.metrics["final"]["chunk"]["mrr@3"] == 0.5
    assert q1.metrics["vector"]["chunk"]["hit@3"] == 0.0  # A#2 is not the labelled chunk
    assert q1.first_relevant_rank == {
        "final": {"page": 2, "chunk": 2},
        "bm25": {"page": 1, "chunk": 1},
        "vector": {"page": 3, "chunk": None},
    }
    assert q1.latency_ms == pytest.approx(10.0)
    # q2 has no chunk labels and a missing route: chunk metrics absent, BM25 scored 0.
    assert "chunk" not in q2.metrics["final"]
    assert q2.metrics["bm25"]["page"]["hit@3"] == 0.0
    assert q2.metrics["vector"]["page"]["mrr@3"] == 1.0
    assert q2.degraded is True

    page = record.summary["final"]["page"]
    assert page.n == 3
    assert page.means["mrr@3"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert page.means["hit@1"] == pytest.approx(1 / 3)
    assert page.hit_wilson["hit@1"] == pytest.approx((0.0615, 0.7923), abs=1e-4)
    assert record.summary["bm25"]["page"].means["mrr@3"] == pytest.approx(1 / 3)
    assert record.summary["vector"]["page"].means["mrr@3"] == pytest.approx((1 / 3 + 1) / 3)
    assert record.summary["final"]["chunk"].n == 1
    assert record.latency.p50_ms == pytest.approx(20.0)
    assert record.latency.p95_ms == pytest.approx(29.0)
    assert record.counts == {"questions": 3, "errors": 1, "degraded": 1, "rerank_timeout": 0}
    assert record.run.index_versions == ["kb_hf_docs_v1"]
    assert record.run.data_generations == [7]


def test_run_records_retriever_errors(retriever: FakeRetriever, question_set) -> None:
    record = _run(retriever, question_set)
    q3 = record.questions[2]
    assert q3.error == "RuntimeError: es down"
    assert q3.hits == []
    assert q3.latency_ms == pytest.approx(30.0)
    assert set(q3.metrics["final"]["page"].values()) == {0.0}
    assert record.missed_by_all_lists == ["q3"]
    with pytest.raises(ValueError, match="outside kb"):
        run_retrieval_eval(
            retriever, question_set, kb_config=KnowledgeBaseConfig(kb="other"), now=NOW
        )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path(".")), *args], capture_output=True, text=True, check=False
    )


def test_write_run_record_and_provenance(
    tmp_path: Path, retriever: FakeRetriever, question_set
) -> None:
    settings = ServiceSettings()
    cfg = KnowledgeBaseConfig(kb=KB)
    record = run_retrieval_eval(
        retriever,
        question_set,
        kb_config=cfg,
        top_k=3,
        ks=(1, 3),
        provenance=collect_provenance(settings=settings),
        clock=_clock(),
        now=NOW,
    )
    out = write_run(record, public=True, reports_root=tmp_path)
    assert out == tmp_path / "public" / "20261017T093000Z_hf_docs"
    assert sorted(p.name for p in out.iterdir()) == ["report.md", "results.json"]

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    run = data["run"]
    assert run["public"] is True
    assert run["git_commit"] == _git("rev-parse", "HEAD").stdout.strip()
    assert isinstance(run["git_dirty"], bool)
    assert run["kb_config"] == cfg.model_dump(mode="json")
    assert len(run["kb_config_sha256"]) == 64
    assert run["index_versions"] == ["kb_hf_docs_v1"]
    assert run["data_generations"] == [7]
    assert run["question_set"] == {
        "path": question_set.path,
        "sha256": question_set.sha256,
        "count": 3,
    }
    assert run["models"] == {
        "embed": settings.embed_model,
        "embed_dim": str(settings.embed_dim),
        "reranker": settings.reranker_model,
    }
    assert run["lock_sha256"] == content_sha256(repo_path("uv.lock").read_bytes())
    assert run["top_k"] == 3
    assert data["questions"][0]["hits"][0]["route_ranks"] == {"bm25": 2, "vector": 1}
    assert load_run(out) == load_run(out.name, reports_root=tmp_path)
    assert load_run(out).summary == record.summary

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "20261017T093000Z_hf_docs" in report
    assert question_set.sha256 in report
    assert "| hit@1 |" in report
    assert "no reliable difference" in report  # 3 questions cannot show any binary difference
    # The resolution line of each level counts the questions labelled at that level.
    page_part, chunk_part = report.split("## Chunk-level metrics")
    assert "Resolution: 3 questions cannot" in page_part.split("## Page-level metrics")[1]
    assert "Resolution: 1 question cannot" in chunk_part
    assert "three?" in report  # missed by every list: check the label first
    assert "RuntimeError: es down" in report

    with pytest.raises(FileExistsError):
        write_run(record, public=True, reports_root=tmp_path)


def test_private_runs_go_to_ignored_dir(
    tmp_path: Path, retriever: FakeRetriever, question_set
) -> None:
    record = _run(retriever, question_set)
    out = write_run(record, public=False, reports_root=tmp_path)
    assert out == tmp_path / "private" / record.run.run_id
    assert load_run(out).run.public is False
    # In the repo, private reports are git-ignored and public ones are not.
    assert _git("check-ignore", "-q", "evals/reports/private/x/results.json").returncode == 0
    assert _git("check-ignore", "-q", "evals/reports/public/x/results.json").returncode == 1


def test_ks_above_top_k_are_capped_with_a_warning(retriever: FakeRetriever, question_set) -> None:
    """A k larger than the final list would only repeat @top_k under another name."""
    with pytest.warns(UserWarning, match="final list"):
        record = run_retrieval_eval(
            retriever,
            question_set,
            kb_config=KnowledgeBaseConfig(kb=KB),
            top_k=3,
            ks=(1, 3, 5, 10),
            provenance=Provenance(),
            clock=_clock(),
            now=NOW,
        )
    assert record.run.ks == [1, 3]
    assert list(record.summary["final"]["page"].means) == metric_names((1, 3))
    assert list(record.summary["bm25"]["page"].means) == metric_names((1, 3))
    assert "MRR within the top 3" in render_run_report(record)


def test_git_dirty_counts_untracked_files(tmp_path: Path) -> None:
    # git_env(): inside a git hook GIT_DIR points at this repo, not at tmp_path.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=git_env())
    assert collect_provenance(repo_root=tmp_path, models={}).git_dirty is False
    (tmp_path / "evals" / "reports" / "public" / "r1").mkdir(parents=True)
    (tmp_path / "evals" / "reports" / "public" / "r1" / "results.json").write_text("{}")
    # Eval reports are outputs, not inputs: they never make a run dirty.
    assert collect_provenance(repo_root=tmp_path, models={}).git_dirty is False
    (tmp_path / "new_step.py").write_text("x = 1\n")
    provenance = collect_provenance(repo_root=tmp_path, models={})
    assert provenance.git_dirty is True
    assert provenance.git_commit is None  # no commit yet
    assert provenance.lock_sha256 is None
