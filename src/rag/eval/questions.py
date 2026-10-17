"""Labelled question sets (design s14-2): one JSON object per line, labels at page level.

A page is a doc_id (s5-4: knowledge base + relative path + unit), so labels survive a change of
chunking. Chunk labels (`relevant_chunks`) are an optional extra; questions without them are
left out of chunk-level metrics. The set's sha256 goes into every run record; it is taken with
Windows line endings normalized to `\\n`, so a checkout with CRLF has the same hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag.config import REPO_ROOT, repo_path
from rag.data.hf import EvalQuestion


class LabeledQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = ""
    """Reference answer (not used by retrieval metrics)."""
    relevant_pages: list[str] = Field(default_factory=list)
    """doc_ids of the pages that answer the question."""
    relevant_chunks: list[str] = Field(default_factory=list)
    """Optional chunk_ids; depend on the chunking, so they are a supplement only."""

    @field_validator("relevant_pages", "relevant_chunks")
    @classmethod
    def _unique_non_empty(cls, ids: list[str]) -> list[str]:
        if any(not i for i in ids):
            raise ValueError("ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in {ids}")
        return ids


class QuestionSet(BaseModel):
    path: str
    """Repo-relative POSIX path when the file is inside the repo, else the absolute path."""
    sha256: str
    questions: list[LabeledQuestion]

    @property
    def has_chunk_labels(self) -> bool:
        return any(q.relevant_chunks for q in self.questions)

    def check_kb(self, kb: str) -> None:
        """Every page label must be a doc_id of knowledge base `kb`."""
        for q in self.questions:
            wrong = [p for p in q.relevant_pages if not p.startswith(f"{kb}:")]
            if wrong:
                raise ValueError(
                    f"{self.path}: question {q.qid!r} labels pages outside kb {kb!r}: {wrong}"
                )


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def load_question_set(path: str | Path, *, kb: str | None = None) -> QuestionSet:
    """Read a JSONL question set; a relative path is taken from the repo root.

    Rejects malformed lines (with the line number), unknown keys, duplicate qids and an empty
    file; with `kb`, also page labels of another knowledge base.
    """
    file = repo_path(path)
    data = file.read_bytes()
    questions: list[LabeledQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = LabeledQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{file}: line {lineno}: {exc}") from exc
        if question.qid in seen:
            raise ValueError(f"{file}: line {lineno}: duplicate qid {question.qid!r}")
        seen.add(question.qid)
        questions.append(question)
    if not questions:
        raise ValueError(f"{file}: question set is empty")
    qs = QuestionSet(path=display_path(file), sha256=content_sha256(data), questions=questions)
    if kb is not None:
        qs.check_kb(kb)
    return qs


def save_question_set(questions: Sequence[LabeledQuestion], path: str | Path) -> str:
    """Write one JSON object per line (UTF-8, `\\n` line ends); returns the file's sha256."""
    qids = [q.qid for q in questions]
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate qids")
    lines = [json.dumps(q.model_dump(mode="json"), ensure_ascii=False) for q in questions]
    data = "".join(f"{line}\n" for line in lines).encode("utf-8")
    file = repo_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return content_sha256(data)


def labels_from_hf_eval(questions: Iterable[EvalQuestion]) -> list[LabeledQuestion]:
    """The HF eval set records the source document of each question: one page label each.

    Leading and trailing whitespace is stripped (most HF questions end with a newline).
    """
    return [
        LabeledQuestion(
            qid=q.qid,
            question=q.question.strip(),
            answer=q.answer.strip(),
            relevant_pages=[q.doc_id],
        )
        for q in questions
    ]
