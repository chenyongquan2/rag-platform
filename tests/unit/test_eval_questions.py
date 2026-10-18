"""Spec P0-10 AC2-AC3: labelled question sets (page-level labels, optional chunk labels)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag.config import repo_path
from rag.data.hf import EvalQuestion
from rag.eval.questions import (
    LabeledQuestion,
    labels_from_hf_eval,
    load_question_set,
    save_question_set,
)
from rag.eval.retrieval import cli

COMMITTED_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"


def _questions() -> list[LabeledQuestion]:
    return [
        LabeledQuestion(
            qid="q1",
            question="How do I push a model?",
            answer="Call push_to_hub.",
            relevant_pages=["hf_docs:huggingface/hub-docs/blob/main/docs/hub/models.md"],
        ),
        LabeledQuestion(
            qid="q2",
            question="Wie lädt man ein Modell?",  # non-ASCII stays readable in the file
            relevant_pages=["hf_docs:a.md", "hf_docs:b.md"],
            relevant_chunks=["hf_docs:a.md#3"],
        ),
    ]


def test_question_set_roundtrip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    sha = save_question_set(_questions(), path)
    raw = path.read_bytes()
    assert sha == hashlib.sha256(raw).hexdigest()
    assert raw.count(b"\n") == 2 and b"\r" not in raw
    assert "Wie lädt" in raw.decode("utf-8")

    qs = load_question_set(path, kb="hf_docs")
    assert qs.questions == _questions()
    assert qs.sha256 == sha
    assert qs.path == path.resolve().as_posix()
    assert qs.has_chunk_labels is True
    assert qs.questions[0].relevant_chunks == []

    # The same content saved with Windows line endings has the same hash.
    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert load_question_set(crlf).sha256 == sha


def test_question_set_rejects_bad_input(tmp_path: Path) -> None:
    def load(lines: list[str], **kw: str) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        load_question_set(path, **kw)

    good = json.dumps({"qid": "q1", "question": "x", "relevant_pages": ["hf_docs:a.md"]})
    with pytest.raises(ValueError, match="line 2"):
        load([good, "{not json"])
    with pytest.raises(ValueError, match="duplicate qid 'q1'"):
        load([good, good])
    with pytest.raises(ValueError, match="line 1"):
        load([json.dumps({"qid": "q1", "question": "x", "pages": ["hf_docs:a.md"]})])
    with pytest.raises(ValueError, match="line 1"):
        load([json.dumps({"qid": "q1", "question": "", "relevant_pages": ["hf_docs:a.md"]})])
    with pytest.raises(ValueError, match="line 1"):
        load([json.dumps({"qid": "q1", "question": "x", "relevant_pages": ["a.md", "a.md"]})])
    with pytest.raises(ValueError, match="kb 'mt5'"):
        load([good], kb="mt5")
    with pytest.raises(ValueError, match="empty"):
        load([])
    load([good, ""], kb="hf_docs")  # a blank line (trailing newline) is fine


def test_labels_from_hf_eval_questions() -> None:
    eq = EvalQuestion(
        qid="huggingface_doc_qa_eval:007",
        question="What is alpha?\n",
        answer=" alpha body",
        source_doc="huggingface/transformers/blob/main/docs/alpha.md",
        doc_id="hf_docs:huggingface/transformers/blob/main/docs/alpha.md",
        context="context",
    )
    (label,) = labels_from_hf_eval([eq])
    assert label == LabeledQuestion(
        qid="huggingface_doc_qa_eval:007",
        question="What is alpha?",
        answer="alpha body",
        relevant_pages=["hf_docs:huggingface/transformers/blob/main/docs/alpha.md"],
    )


def test_export_hf_cli_writes_question_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    eq = EvalQuestion(
        qid="huggingface_doc_qa_eval:000",
        question="What is alpha?",
        answer="alpha body",
        source_doc="a.md",
        doc_id="hf_docs:a.md",
        context="",
    )
    calls = []

    def fake_load(manifest: object, *, download: bool) -> list[EvalQuestion]:
        calls.append(download)
        return [eq]

    monkeypatch.setattr(cli, "load_eval_questions", fake_load)
    out = tmp_path / "out.jsonl"
    assert cli.main(["export-hf", "--out", str(out), "--no-download"]) == 0
    assert calls == [False]
    qs = load_question_set(out, kb="hf_docs")
    assert [q.qid for q in qs.questions] == ["huggingface_doc_qa_eval:000"]
    printed = capsys.readouterr().out
    assert "1 questions" in printed
    assert qs.sha256 in printed


def test_committed_hf_question_set() -> None:
    qs = load_question_set(COMMITTED_SET, kb="hf_docs")
    assert qs.path == COMMITTED_SET
    assert len(qs.questions) == 65
    assert [q.qid for q in qs.questions] == [f"huggingface_doc_qa_eval:{i:03d}" for i in range(65)]
    assert all(len(q.relevant_pages) == 1 and q.answer for q in qs.questions)
    assert qs.has_chunk_labels is False
    assert repo_path(COMMITTED_SET).read_bytes().count(b"\n") == 65


def test_no_answer_set_and_label_overrides() -> None:
    """Spec P0-13 AC4: 10 hand-written no-answer questions without labels; every label override
    names a question of the public set, gives a finding, an action and a reason."""
    root = repo_path("evals/datasets/hf_docs")
    no_answer = load_question_set(root / "no_answer.jsonl", kb="hf_docs")
    assert len(no_answer.questions) == 10
    assert all(not q.relevant_pages and not q.relevant_chunks for q in no_answer.questions)
    public = {q.qid for q in load_question_set(root / "huggingface_doc_qa_eval.jsonl").questions}
    assert not public & {q.qid for q in no_answer.questions}
    lines = (root / "label_overrides.jsonl").read_text(encoding="utf-8").splitlines()
    overrides = [json.loads(line) for line in lines if line.strip()]
    assert overrides
    for entry in overrides:
        assert entry["qid"] in public
        assert entry["finding"] in {"label_error", "retrieval_failure"}
        assert entry["action"] in {"exclude", "keep"}
        assert len(entry["reason"]) > 40
