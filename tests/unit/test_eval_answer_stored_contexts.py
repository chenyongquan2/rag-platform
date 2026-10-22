"""Spec P7-01 AC1, AC2: replay the chunks of an earlier answer run (same contexts, other model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.config import ServiceSettings
from rag.eval.answer import cli
from rag.eval.answer.record import AnswerItem, AnswerQuestionRecord
from rag.eval.answer.runner import StoredContexts, load_answer_run
from rag.eval.questions import LabeledQuestion, save_question_set
from rag.testing import FakeRetriever, StreamingFakeChatModel
from rag.types import Answer, Chunk, RetrievalResult


def _chunk(doc: str, i: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"hf_docs:{doc}#{i}", doc_id=f"hf_docs:{doc}", kb="hf_docs", text=f"{doc} {i}"
    )


class _Run:
    """The two fields of an `AnswerRunRecord` that `StoredContexts` reads."""

    def __init__(self, questions: list[AnswerQuestionRecord]) -> None:
        self.run_id = "r1"
        self.questions = questions


def test_stored_contexts_replay_chunks_and_not_found() -> None:
    chunks = [_chunk("b.md", 2), _chunk("a.md", 0)]
    answered = AnswerQuestionRecord(
        qid="q1",
        question="Q1?",
        answer=Answer(request_id="x", kb="hf_docs", chunks=chunks),
        degraded=True,
    )
    refused = AnswerQuestionRecord(
        qid="q2",
        question="Q2?",
        answer=Answer(request_id="y", kb="hf_docs"),
        retrieval_not_found=True,
    )
    contexts = StoredContexts(_Run([answered, refused]))  # type: ignore[arg-type]

    got = contexts(AnswerItem(qid="q1", question="Q1?"))
    assert [c.chunk_id for c in got.chunks] == ["hf_docs:b.md#2", "hf_docs:a.md#0"]
    assert got.not_found is False and got.degraded is True
    got2 = contexts(AnswerItem(qid="q2", question="Q2?"))
    assert got2.chunks == [] and got2.not_found is True


def test_unknown_qid_raises() -> None:
    contexts = StoredContexts(_Run([]))  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="q9"):
        contexts(AnswerItem(qid="q9", question="?"))


def test_cli_contexts_from_skips_the_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qfile = tmp_path / "questions.jsonl"
    save_question_set(
        [LabeledQuestion(qid="q1", question="How to log in?", answer="Use login.")], qfile
    )
    result = RetrievalResult(request_id="r", kb="hf_docs", chunks=[_chunk("a.md")])
    built: list[int] = []

    def build_retriever(cfg: Any, settings: Any, **kw: Any) -> FakeRetriever:
        built.append(1)
        return FakeRetriever({"How to log in?": result})

    monkeypatch.setattr(cli, "build_retriever", build_retriever)
    monkeypatch.setattr(
        cli,
        "build_answer_model",
        lambda settings: StreamingFakeChatModel(responses=["Log in [1].", "Use login [1]."]),
    )
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    out = tmp_path / "out"
    common = ["run", "--kb", "hf_docs", "--questions", str(qfile), "--out-dir", str(out)]
    assert cli.main([*common, "--label", "first"], settings=settings) == 0
    first = next((out / "answers").iterdir())
    assert built == [1]

    assert (
        cli.main([*common, "--label", "second", "--contexts-from", str(first)], settings=settings)
        == 0
    )
    assert built == [1]  # no second retriever
    second_dir = next(p for p in (out / "answers").iterdir() if p.name.endswith("_second"))
    second = load_answer_run(second_dir)
    assert second.questions[0].context_chunk_ids == ["hf_docs:a.md#0"]
    assert second.questions[0].found and second.questions[0].cited_chunk_ids == ["hf_docs:a.md#0"]
