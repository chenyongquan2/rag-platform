"""The shape of a hard or ambiguous question (spec EVAL-2): what `meta` must say."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag.eval.questions import LabeledQuestion, load_question_set

HARD_FILE = "evals/datasets/hf_docs/hard.jsonl"
HARD_PREFIX = "hf_docs_hard:"
AMBIGUOUS_PREFIX = "hf_docs_ambiguous:"
CLARIFICATION_PREFIX = "clarification expected"

Shape = Literal["bridge", "comparison", "list_all", "chain3"]
FailureType = Literal["bridge", "comparison", "vocabulary_gap"]


class Fact(BaseModel):
    """One statement a hard question needs, on one page."""

    model_config = ConfigDict(extra="forbid")

    page: str = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=200)
    """Verbatim text of the page (whitespace-insensitive)."""
    evidence_also: list[str] = Field(default_factory=list)
    key: str = Field(min_length=1)
    """Regular expression (case-insensitive) of the statement; it must match this page and no
    other page of the corpus except the declared copies."""
    also_on: list[str] = Field(default_factory=list)
    """Other pages that repeat the statement word for word."""
    subjects: list[str] = Field(default_factory=list)
    """Extra words that name what the fact is about (a model, a library); used only by the
    co-mention search."""
    loose: list[list[str | int]] = Field(min_length=1)
    """Anchor and fact phrases (optional window in characters) searched with the inflection-aware
    rules of `rag.eval.absence`: another way of writing the statement."""

    @field_validator("loose")
    @classmethod
    def _pairs(cls, pairs: list[list[str | int]]) -> list[list[str | int]]:
        for pair in pairs:
            ok = len(pair) in (2, 3) and all(isinstance(x, str) and x for x in pair[:2])
            if not ok or (len(pair) == 3 and not isinstance(pair[2], int)):
                raise ValueError(
                    f"loose pair must be [anchor, fact] or [anchor, fact, window]: {pair}"
                )
        return pairs


class HardMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hard"]
    requires: Literal["all"]
    shape: Shape
    failure_type: FailureType
    modelled_on: str = Field(min_length=1)
    """The round-1 failure the question is modelled on (a question id; not copied)."""
    style: list[str] = Field(default_factory=list)
    avoided_terms: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(min_length=2, max_length=3)
    reasoning: str = Field(min_length=1)
    read_and_cleared: dict[str, str] = Field(default_factory=dict)
    """Pages other than the gold pages that mention every term of every fact (found by the
    co-mention search of `rag.eval.hard.corpus`), each read by the author, with the reason it does
    not answer the question."""
    checked_single_page: str = Field(min_length=1)


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reading: str = Field(min_length=1)
    page: str = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1)


class AmbiguousMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ambiguous"]
    ambiguity: Literal["scope", "missing_referent"]
    readings: list[Reading] = Field(min_length=2)
    check: Literal["rule:asked_back"]
    why_ambiguous: str = Field(min_length=1)


def load_hard(path: str = HARD_FILE) -> list[LabeledQuestion]:
    return load_question_set(path, kb="hf_docs").questions


def validate_questions(questions: Sequence[LabeledQuestion]) -> list[str]:
    """Shape problems of the questions, one line each (empty: fine). No corpus needed."""
    problems: list[str] = []
    for qid, n in Counter(q.qid for q in questions).items():
        if n > 1:
            problems.append(f"{qid}: duplicate question id")
    for q in questions:
        kind = q.meta.get("type")
        try:
            if kind == "hard":
                _check_hard(q, problems)
            elif kind == "ambiguous":
                _check_ambiguous(q, problems)
            else:
                problems.append(f"{q.qid}: meta.type must be hard or ambiguous, got {kind!r}")
        except ValidationError as exc:
            for err in exc.errors():
                where = ".".join(str(p) for p in err["loc"])
                problems.append(f"{q.qid}: meta.{where}: {err['msg']}")
    return problems


def _check_hard(q: LabeledQuestion, problems: list[str]) -> None:
    meta = HardMeta.model_validate(q.meta)
    if not q.qid.startswith(HARD_PREFIX):
        problems.append(f"{q.qid}: id must start with {HARD_PREFIX}")
    if not q.answer.strip():
        problems.append(f"{q.qid}: no reference answer")
    pages = [f.page for f in meta.facts]
    if q.relevant_pages != pages:
        problems.append(f"{q.qid}: relevant_pages must be the fact pages in order")
    if len(set(pages)) != len(pages):
        problems.append(f"{q.qid}: the facts must be on different pages")
    if meta.shape == "chain3" and len(pages) != 3:
        problems.append(f"{q.qid}: a chain3 question needs three facts")
    for fact in meta.facts:
        _compile(q.qid, fact.key, problems)
        if not fact.subjects:
            problems.append(f"{q.qid}: the fact on {fact.page} needs at least one subject phrase")


def _compile(qid: str, pattern: str, problems: list[str]) -> None:
    import re

    try:
        re.compile(pattern)
    except re.error as exc:
        problems.append(f"{qid}: key pattern {pattern!r} is not a regular expression: {exc}")


def _check_ambiguous(q: LabeledQuestion, problems: list[str]) -> None:
    meta = AmbiguousMeta.model_validate(q.meta)
    if not q.qid.startswith(AMBIGUOUS_PREFIX):
        problems.append(f"{q.qid}: id must start with {AMBIGUOUS_PREFIX}")
    if q.relevant_pages or q.relevant_chunks:
        problems.append(f"{q.qid}: an ambiguous question has no gold page")
    if not q.answer.lower().startswith(CLARIFICATION_PREFIX):
        problems.append(f"{q.qid}: the reference must start with 'Clarification expected'")
    answers = [r.answer for r in meta.readings]
    if len(set(answers)) != len(answers):
        problems.append(f"{q.qid}: the readings must have different answers")


def meta_of(q: LabeledQuestion) -> Any:
    """The validated meta of a question (HardMeta or AmbiguousMeta)."""
    return (HardMeta if q.meta.get("type") == "hard" else AmbiguousMeta).model_validate(q.meta)
