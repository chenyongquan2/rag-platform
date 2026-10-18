"""Spec P1-04 AC11: the answer eval CLI with fake retriever and fake chat models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from rag.config import ServiceSettings
from rag.eval.answer import cli
from rag.eval.answer.runner import load_answer_run
from rag.eval.questions import LabeledQuestion, save_question_set
from rag.testing import FakeChatModel, FakeRetriever, StreamingFakeChatModel
from rag.types import Chunk, RetrievalResult

QUESTIONS = [
    LabeledQuestion(
        qid="q1", question="How to log in?", answer="Use login.", relevant_pages=["hf_docs:a.md"]
    ),
    LabeledQuestion(
        qid="q2", question="Where are logs?", answer="In logs/.", relevant_pages=["hf_docs:b.md"]
    ),
]


def _result(doc: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"hf_docs:{doc}#0",
        doc_id=f"hf_docs:{doc}",
        kb="hf_docs",
        text=f"text of {doc}",
        score=1.0,
        rerank_score=0.9,
    )
    return RetrievalResult(request_id="r", kb="hf_docs", chunks=[chunk])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    qfile = tmp_path / "questions.jsonl"
    save_question_set(QUESTIONS, qfile)
    retriever = FakeRetriever(
        {"How to log in?": _result("a.md"), "Where are logs?": _result("b.md")}
    )
    monkeypatch.setattr(cli, "build_retriever", lambda cfg, settings, **kw: retriever)
    monkeypatch.setattr(
        cli,
        "build_answer_model",
        lambda settings: StreamingFakeChatModel(
            responses=["Log in with login [1].", "In logs [1]."]
        ),
    )
    judge = FakeChatModel(default_response="Feedback: right. [RESULT] 3")
    monkeypatch.setattr(cli, "build_judge_model", lambda settings, config: judge)
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    return {"qfile": qfile, "out": tmp_path / "out", "settings": settings, "tmp": tmp_path}


def _main(env: dict[str, Any], *argv: str) -> int:
    return cli.main(list(argv), settings=env["settings"])


def test_cli_with_fakes(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    out = env["out"]
    assert (
        _main(
            env,
            "run",
            "--kb",
            "hf_docs",
            "--questions",
            str(env["qfile"]),
            "--label",
            "pilot",
            "--limit",
            "1",
            "--out-dir",
            str(out),
        )
        == 0
    )
    runs = sorted((out / "answers").iterdir())
    assert len(runs) == 1
    pilot = load_answer_run(runs[0])
    assert [q.qid for q in pilot.questions] == ["q1"]
    assert (runs[0] / "summary.md").is_file()

    assert (
        _main(
            env,
            "run",
            "--kb",
            "hf_docs",
            "--questions",
            str(env["qfile"]),
            "--label",
            "full",
            "--out-dir",
            str(out),
        )
        == 0
    )
    full_dir = next(p for p in (out / "answers").iterdir() if p.name.endswith("_full"))
    full = load_answer_run(full_dir)
    assert full.compliance.fully_cited.count == 2

    assert (
        _main(
            env,
            "judge",
            "--answers",
            str(full_dir),
            "--scales",
            "1-3",
            "--repeat",
            "2",
            "--label",
            "t",
            "--out-dir",
            str(out),
        )
        == 0
    )
    judges = sorted((out / "judge").iterdir())
    assert len(judges) == 2
    data = json.loads((judges[0] / "results.json").read_text("utf-8"))
    assert data["summary"]["1-3"]["percent"] == pytest.approx(100.0)
    assert data["answer_run_id"] == full.run_id

    assert _main(env, "stability", str(judges[0]), str(judges[1]), "--scale", "1-3") == 0
    assert "flips: 0/2" in capsys.readouterr().out

    assert (
        _main(
            env,
            "human-sheet",
            "--answers",
            str(full_dir),
            "--n",
            "2",
            "--out-dir",
            str(out / "human"),
        )
        == 0
    )
    with (out / "human" / "human_sheet.csv").open(encoding="utf-8-sig", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2
    assert (out / "human" / "human_key.csv").is_file()

    assert (
        _main(
            env,
            "project",
            "--answers",
            str(runs[0]),
            "--judge",
            str(judges[0]),
            "--questions",
            "65",
        )
        == 0
    )
    assert "total" in capsys.readouterr().out
