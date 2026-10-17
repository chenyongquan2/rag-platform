"""Spec P0-12 AC1-AC6: the `run` and `rerank-curve` commands of the retrieval eval CLI.

The real chain (Elasticsearch, gateway, cross-encoder) is replaced by `FakeRetriever` and a fake
reranker through the `build_retriever` / `build_reranker` seams of the CLI module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.eval.questions import LabeledQuestion, save_question_set
from rag.eval.retrieval import cli
from rag.testing import FakeRetriever
from rag.types import Chunk, RetrievalResult, RouteHit

KB = "hf_docs"
QUESTIONS = [
    LabeledQuestion(qid="q1", question="how to log in?", answer="a", relevant_pages=["hf_docs:a"]),
    LabeledQuestion(qid="q2", question="where are logs?", answer="b", relevant_pages=["hf_docs:b"]),
]
KB_YAML = """\
kb: hf_docs
source: {connector: hf_dataset, limit: null}
language: en
retrieval: {rerank_n: 30, final_k: 8}
"""


def _chunk(doc: str, n: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"hf_docs:{doc}#{n}",
        doc_id=f"hf_docs:{doc}",
        kb=KB,
        text=f"text {doc} {n}",
        title_path=["T"],
        score=1.0 / (n + 1),
    )


def _result(*docs: str) -> RetrievalResult:
    chunks = [_chunk(d, i) for i, d in enumerate(docs)]
    return RetrievalResult(
        request_id="r",
        kb=KB,
        index_version="kb_hf_docs_v1",
        data_generation=3,
        chunks=chunks,
        routes={
            "bm25": [
                RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=i + 1)
                for i, c in enumerate(chunks)
            ],
        },
    )


class _Built:
    """Records what `build_retriever` was called with and returns a FakeRetriever."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.retriever = FakeRetriever(
            {"how to log in?": _result("a", "c"), "where are logs?": _result("c", "b")}
        )

    def __call__(
        self, cfg: KnowledgeBaseConfig, settings: ServiceSettings, **kw: Any
    ) -> FakeRetriever:
        self.calls.append({"cfg": cfg, "settings": settings, **kw})
        return self.retriever


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    kb_dir = tmp_path / "kb"
    (kb_dir / "public").mkdir(parents=True)
    (kb_dir / "public" / "hf_docs.yaml").write_text(KB_YAML, encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    save_question_set(QUESTIONS, questions)
    built = _Built()
    monkeypatch.setattr(cli, "build_retriever", built)
    monkeypatch.setattr(cli, "reranker_revision", lambda model: "abc123")
    settings = ServiceSettings(_env_file=None, kb_dir=str(kb_dir))
    reports = tmp_path / "reports"
    base = ["--kb", KB, "--questions", str(questions), "--reports-dir", str(reports)]
    return {"settings": settings, "built": built, "reports": reports, "base": base}


def _only_run(reports: Path, visibility: str = "public") -> dict[str, Any]:
    (run,) = (reports / visibility).iterdir()
    assert (run / "report.md").is_file()
    return json.loads((run / "results.json").read_text(encoding="utf-8"))


def test_run_writes_public_record(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", *env["base"]], settings=env["settings"]) == 0
    record = _only_run(env["reports"])
    assert record["run"]["public"] is True
    assert record["counts"]["questions"] == 2
    assert [q["qid"] for q in record["questions"]] == ["q1", "q2"]
    assert record["run"]["lists"] == ["final", "bm25"]
    # q1: page a at rank 1; q2: page b at rank 2.
    assert record["summary"]["final"]["page"]["means"]["hit@1"] == pytest.approx(0.5)
    assert record["summary"]["final"]["page"]["means"]["mrr@8"] == pytest.approx(0.75)
    assert [c[0] for c in env["built"].retriever.calls] == ["how to log in?", "where are logs?"]
    assert all(c[2] == 8 for c in env["built"].retriever.calls)  # top_k = final_k
    out = capsys.readouterr().out
    assert "hit@1" in out and str(env["reports"]) in out


def test_run_rerank_options(env: dict[str, Any]) -> None:
    assert cli.main(["run", *env["base"], "--rerank-n", "10"], settings=env["settings"]) == 0
    call = env["built"].calls[-1]
    assert call["rerank"] is True
    assert call["cfg"].retrieval.rerank_n == 10
    record = _only_run(env["reports"])
    assert record["run"]["kb_config"]["retrieval"]["rerank_n"] == 10
    assert record["run"]["label"] == "rerank10"

    reports = env["reports"].parent / "reports2"
    base = [*env["base"][:-1], str(reports)]
    assert cli.main(["run", *base, "--no-rerank"], settings=env["settings"]) == 0
    assert env["built"].calls[-1]["rerank"] is False
    assert _only_run(reports)["run"]["label"] == "norerank"

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *base, "--no-rerank", "--rerank-n", "5"], settings=env["settings"])
    assert exc.value.code == 2


def test_run_routes_option(env: dict[str, Any]) -> None:
    assert cli.main(["run", *env["base"], "--routes", "bm25"], settings=env["settings"]) == 0
    call = env["built"].calls[-1]
    assert call["routes"] == ("bm25",)
    assert _only_run(env["reports"])["run"]["label"] == "bm25_rerank30"

    argv = ["run", *env["base"], "--routes", "vector,sparse"]
    assert cli.main(argv, settings=env["settings"]) == 2


def test_run_deadline_and_unknown_kb(env: dict[str, Any]) -> None:
    assert cli.main(["run", *env["base"], "--deadline-ms", "60000"], settings=env["settings"]) == 0
    assert env["built"].calls[-1]["settings"].search_deadline_ms == 60000
    argv = ["run", "--kb", "nope", *env["base"][2:]]
    assert cli.main(argv, settings=env["settings"]) == 2


def test_run_provenance_models(env: dict[str, Any]) -> None:
    assert cli.main(["run", *env["base"]], settings=env["settings"]) == 0
    models = _only_run(env["reports"])["run"]["models"]
    settings: ServiceSettings = env["settings"]
    assert models == {
        "embed": settings.embed_model,
        "embed_dim": str(settings.embed_dim),
        "reranker": settings.reranker_model,
        "reranker_revision": "abc123",
    }


def test_rerank_curve_command(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []

    def reverse(question: str, candidates: Sequence[Chunk]) -> list[Chunk]:
        calls.append(len(candidates))
        return list(reversed(candidates))

    monkeypatch.setattr(cli, "build_reranker", lambda settings: reverse)
    out = tmp_path / "curve"
    argv = ["rerank-curve", *env["base"][:4], "--rerank-ns", "1,2", "--out", str(out)]
    assert cli.main(argv, settings=env["settings"]) == 0
    assert env["built"].calls[-1]["rerank"] is False
    data = json.loads((out / "rerank_curve.json").read_text(encoding="utf-8"))
    assert [p["rerank_n"] for p in data["points"]] == [None, 1, 2]
    assert data["provenance"]["models"]["reranker_revision"] == "abc123"
    assert (out / "rerank_curve.md").is_file()
    assert calls == [2, 1, 2, 1, 2]  # one warm-up call, then n = 1, 2 per question


def test_output_survives_a_non_utf8_console(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A question with an emoji must not crash `compare` on a GBK console (Windows); the
    `--out` file keeps the exact text."""
    import io
    import sys

    emoji = "What does 🤗 Datasets do?"
    questions = tmp_path / "emoji.jsonl"
    save_question_set(
        [LabeledQuestion(qid="e1", question=emoji, answer="a", relevant_pages=["hf_docs:a"])],
        questions,
    )
    env["built"].retriever.results[emoji] = _result("a")
    base = ["--kb", KB, "--questions", str(questions), "--reports-dir", str(env["reports"])]
    assert cli.main(["run", *base, "--label", "a"], settings=env["settings"]) == 0
    env["built"].retriever.results[emoji] = _result("c", "a")
    assert cli.main(["run", *base, "--label", "b"], settings=env["settings"]) == 0
    run_a, run_b = sorted(p.name for p in (env["reports"] / "public").iterdir())

    console = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
    monkeypatch.setattr(sys, "stdout", console)
    out = tmp_path / "cmp.md"
    argv = ["compare", run_a, run_b, "--reports-dir", str(env["reports"]), "--out", str(out)]
    assert cli.main(argv) == 0
    assert emoji in out.read_text(encoding="utf-8")
