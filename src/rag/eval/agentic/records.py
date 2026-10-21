"""Per-question records of the Agentic RAG comparison and their JSONL files (spec P3-05).

One line per question and system, appended as soon as the question is done, so a crash loses at
most the question in flight and a second run skips what is already there (`done_qids`). The
judge's scores are lines of their own (`JudgeLine`) in a second file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from rag.eval.answer.record import Usage
from rag.types import AnswerSentence, Chunk

SystemName = Literal["standard", "agentic"]
SYSTEMS: tuple[SystemName, ...] = ("standard", "agentic")
RECORDS_NAME = "records.jsonl"
JUDGE_NAME = "judge.jsonl"


def usage_by_profile(rows: list[dict[str, Any]]) -> dict[str, Usage]:
    """Model calls of one question by the chat profile that made them (`online`: the answer,
    `cheap`: route, rewrite and judge); embedding calls the cache missed are `embedding`."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = "embedding" if row.get("kind") == "embedding" else str(row.get("profile") or "chat")
        groups.setdefault(name, []).append(row)
    return {name: Usage.from_rows(group) for name, group in sorted(groups.items())}


class RoundInfo(BaseModel):
    """One search of one round: the text searched and the pages it returned."""

    round: int
    sub_question: str
    query: str = ""
    """The text searched, when it is not the sub-question (a phrasing, a HyDE passage)."""
    kind: str = "original"
    pages: list[str] = Field(default_factory=list)
    """Distinct pages of the chunks that passed the rerank threshold, best first."""
    n_chunks: int = 0
    top_score: float | None = None
    not_found: bool = False


class HookEvent(BaseModel):
    """What one hook of the graph decided: `route_model`, `rewrite` (round, queries, technique,
    fallback) or `judge` (round, verdict, missing, fallback)."""

    step: str
    round: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class SystemRecord(BaseModel):
    """One question answered by one system."""

    system: SystemName
    qid: str
    qtype: str
    set: str = ""
    question: str
    reference: str = ""
    relevant_pages: list[str] = Field(default_factory=list)
    requires: str = "any"
    answerable: bool = True
    # What came out.
    status: str = ""
    """`answered`, `not_found`, `need_clarification` or `error` (the question was not answered)."""
    found: bool = False
    clarification: bool = False
    """The graph stopped to ask the caller back; the harness does not answer it (a refusal)."""
    answered: bool = False
    """An answer text exists: found, not a clarification, not an error."""
    text: str = ""
    """The answer without citation markers (what the judge reads); empty when not answered."""
    error: str | None = None
    sentences: int = 0
    unsupported_sentences: int = 0
    answer_parse_failed: bool = False
    generation_error: str | None = None
    retrieved_pages: list[str] = Field(default_factory=list)
    """Pages of every chunk retrieved above the threshold (standard: the one search; agentic:
    all rounds)."""
    context_pages: list[str] = Field(default_factory=list)
    """Pages of the numbered fragments the answer was written from."""
    cited_pages: list[str] = Field(default_factory=list)
    answer_sentences: list[AnswerSentence] = Field(default_factory=list)
    """The answer's sentences with their citations (spec P1-06: citation check level 2 needs
    which sentence cites which chunk). Empty in records written before P1-06."""
    cited_chunks: list[Chunk] = Field(default_factory=list)
    """The chunks the sentences cite, once each, in first-cited order (not the whole context)."""
    retrieval_not_found: bool = False
    # Time and money.
    total_ms: float = 0.0
    steps_ms: dict[str, float] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    usage_by_kind: dict[str, Usage] = Field(default_factory=dict)
    usage_by_profile: dict[str, Usage] = Field(default_factory=dict)
    # The graph only.
    rounds: int = 0
    stop_reason: str | None = None
    coverage_unconfirmed: bool = False
    degraded: bool = False
    rerank_timeout: bool = False
    fallbacks: list[str] = Field(default_factory=list)
    error_categories: list[str] = Field(default_factory=list)
    sufficiency: str | None = None
    missing: list[str] = Field(default_factory=list)
    standalone_question: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    clarify_question: str | None = None
    two_hop: bool = False
    tokens_used: int = 0
    round_info: list[RoundInfo] = Field(default_factory=list)
    events: list[HookEvent] = Field(default_factory=list)


class JudgeLine(BaseModel):
    """The judge's score of one answer (1-3 scale) in one pass."""

    system: SystemName
    qid: str
    run: int
    score: int | None = None
    by_rule: str | None = None
    """`no_answer`: the lowest score without a call (nothing was answered)."""
    error: str | None = None
    feedback: str = ""
    strict_format: bool = True
    usage: Usage = Field(default_factory=Usage)
    judge_model: str = ""
    prompt_version: str = ""


def system_file(out_dir: str | Path, system: str) -> Path:
    return Path(out_dir) / system / RECORDS_NAME


def judge_file(out_dir: str | Path) -> Path:
    return Path(out_dir) / JUDGE_NAME


def repair_tail(path: Path) -> None:
    """Drop a last line cut off by a crash (no final newline), so the next append starts clean."""
    if not path.is_file():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with path.open("r+b") as fh:
            fh.truncate(cut)


def append_line(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        fh.write(model.model_dump_json() + "\n")
        fh.flush()


def _read_lines[T: BaseModel](path: str | Path, cls: type[T]) -> list[T]:
    file = Path(path)
    if not file.is_file():
        return []
    text = file.read_text(encoding="utf-8")
    lines = text.split("\n")
    complete = text.endswith("\n")
    out: list[T] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            out.append(cls.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as exc:
            if not complete and number == len(lines):
                break  # the line a crash cut off
            raise ValueError(f"{file}: line {number}: {str(exc).splitlines()[0]}") from exc
    return out


def read_records(path: str | Path) -> list[SystemRecord]:
    """The records of a file, in order; a cut-off last line is ignored, a bad line elsewhere
    raises `ValueError`."""
    return _read_lines(path, SystemRecord)


def read_judge_lines(path: str | Path) -> list[JudgeLine]:
    return _read_lines(path, JudgeLine)


def done_qids(path: str | Path) -> set[str]:
    return {r.qid for r in read_records(path)}


__all__ = [
    "JUDGE_NAME",
    "RECORDS_NAME",
    "SYSTEMS",
    "HookEvent",
    "JudgeLine",
    "RoundInfo",
    "SystemName",
    "SystemRecord",
    "append_line",
    "done_qids",
    "judge_file",
    "read_judge_lines",
    "read_records",
    "repair_tail",
    "system_file",
    "usage_by_profile",
]
