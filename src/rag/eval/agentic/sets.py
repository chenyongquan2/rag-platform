"""The question sets of the Agentic RAG comparison (spec P3-05, design s13-5).

    items = load_items("hf_docs")                      # 150 questions, four types
    pilot = load_items("hf_docs", take={"single_hop": 4, "no_answer": 2})

Four types, each from fixed files of the KB's eval directory:

    single_hop   the 65 public questions (answer in one page), the P1-04 set
    two_hop      20 hand-written questions that need two pages (P3-03)
    colloquial   15 hand-written casual rewrites of public questions (P3-05)
    no_answer    10 + 20 + 20 questions not answered by the documentation (P1-04, P2-02, P3-03)

Two more types are selected by name and are not part of the default 150 (spec EVAL-2); both come
from `hard.jsonl` and are told apart by `meta.type`:

    hard         25 hand-written answerable questions that need two or three pages
    ambiguous    10 hand-written questions whose right reply is a clarifying question (rule check,
                 no judge: `rag.eval.hard.asked_back`)

Every item carries what both systems and the report need: the reference answer and gold pages of
an answerable question, and `requires` ("all" for a two-hop question: both pages are needed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rag.config import repo_path
from rag.eval.questions import load_question_set

QType = Literal["single_hop", "two_hop", "colloquial", "no_answer", "hard", "ambiguous"]

SET_FILES: dict[str, list[str]] = {
    "single_hop": ["huggingface_doc_qa_eval.jsonl"],
    "two_hop": ["two_hop.jsonl"],
    "colloquial": ["colloquial.jsonl"],
    "no_answer": ["no_answer.jsonl", "no_answer_extra.jsonl", "no_answer_hard.jsonl"],
    "hard": ["hard.jsonl"],
    "ambiguous": ["hard.jsonl"],
}
DEFAULT_TYPES: tuple[str, ...] = ("single_hop", "two_hop", "colloquial", "no_answer")
"""What `load_items` loads without `types`: the 150 questions of the P3-05 comparison."""
HARD_TYPES: tuple[str, ...] = ("hard", "ambiguous")
"""Selected by name only (spec EVAL-2); the two types share one file."""
QTYPES: tuple[str, ...] = (*DEFAULT_TYPES, *HARD_TYPES)
NOT_ANSWERABLE: frozenset[str] = frozenset({"no_answer", "ambiguous"})
"""Types with no reference answer to score: refusals and clarifications are judged by rule."""


class EvalItem(BaseModel):
    """One question of the comparison."""

    qid: str = Field(min_length=1)
    qtype: QType
    set: str
    """The question file's stem (`no_answer_hard`)."""
    question: str = Field(min_length=1)
    reference: str = ""
    relevant_pages: list[str] = Field(default_factory=list)
    requires: Literal["any", "all"] = "any"
    answerable: bool = True
    source_qid: str | None = None
    """Colloquial rewrites: the public question they were derived from."""


def load_items(
    kb: str = "hf_docs",
    *,
    types: Sequence[str] | None = None,
    take: Mapping[str, int] | None = None,
    eval_dir: str | None = None,
) -> list[EvalItem]:
    """The questions of `types` (default the four of `DEFAULT_TYPES`), in the order of `QTYPES`
    and of the files.

    `take` keeps only the first N questions of a type (a pilot); without `types` only the types
    named in `take` are loaded. Unknown type names raise
    `ValueError`; so does a question file whose gold pages belong to another knowledge base.
    """
    chosen = list(types) if types is not None else list(take) if take else list(DEFAULT_TYPES)
    for name in [*chosen, *(take or {})]:
        if name not in SET_FILES:
            raise ValueError(f"unknown question type {name!r}; choose from {list(QTYPES)}")
    directory = repo_path(eval_dir or f"evals/datasets/{kb}")
    items: list[EvalItem] = []
    for qtype in QTYPES:
        if qtype not in chosen:
            continue
        of_type: list[EvalItem] = []
        for filename in SET_FILES[qtype]:
            questions = load_question_set(directory / filename, kb=kb).questions
            for q in questions:
                if filename == "hard.jsonl" and q.meta.get("type") != qtype:
                    continue  # one file, two types
                answerable = qtype not in NOT_ANSWERABLE
                if answerable and not q.relevant_pages:
                    raise ValueError(f"{filename}: {q.qid} is answerable but has no gold page")
                of_type.append(
                    EvalItem(
                        qid=q.qid,
                        qtype=qtype,  # type: ignore[arg-type]
                        set=filename.removesuffix(".jsonl"),
                        question=q.question,
                        reference=q.answer,
                        relevant_pages=list(q.relevant_pages),
                        requires="all" if q.meta.get("requires") == "all" else "any",
                        answerable=answerable,
                        source_qid=q.meta.get("source_qid"),
                    )
                )
        if take and qtype in take:
            of_type = of_type[: take[qtype]]
        items.extend(of_type)
    if len({i.qid for i in items}) != len(items):
        raise ValueError("duplicate qids across the question sets")
    return items
