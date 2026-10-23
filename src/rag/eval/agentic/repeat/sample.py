"""The question list of the repeated runs (spec EVAL-3, plan step 3, option B).

    items = load_items("hf_docs", types=[...five types...])
    ql = build_question_list(items, seed=20261023, date="2026-10-23")

40 old answerable questions (single-hop 20, two-hop 12, colloquial 8) drawn from the 150-question
set, plus all the hard and ambiguous questions of EVAL-2. No-answer questions are not repeated
(both systems refuse all 50 every time, so repeating them adds nothing).

The draw ranks the questions of a type by `sha256("{seed}:{qid}")` and keeps the first N, so it
depends on neither the order of the question files nor the Python version. The list is committed
(`evals/datasets/hf_docs/repeat_step3.json`) with the seed and the date, and is never changed after
the first repeated pass has run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from rag.config import repo_path
from rag.eval.agentic.sets import EvalItem

LIST_PATH = "evals/datasets/hf_docs/repeat_step3.json"
OLD_COUNTS: dict[str, int] = {"single_hop": 20, "two_hop": 12, "colloquial": 8}
NEW_TYPES: tuple[str, ...] = ("hard", "ambiguous")
LIST_TYPES: tuple[str, ...] = (*OLD_COUNTS, *NEW_TYPES)
DEFAULT_SEED = 20261023
DEFAULT_DATE = "2026-10-23"
METHOD = (
    "old answerable questions: the first N of each type by sha256('seed:qid'); "
    "hard and ambiguous questions: all; no-answer questions: not repeated"
)


class QuestionList(BaseModel):
    version: int = 1
    kb: str = "hf_docs"
    seed: int
    date: str
    method: str = METHOD
    counts: dict[str, int] = Field(default_factory=dict)
    qids: dict[str, list[str]] = Field(default_factory=dict)
    """Question ids per type, in the order of the question files."""

    def all_qids(self) -> list[str]:
        return [q for t in LIST_TYPES for q in self.qids.get(t, [])]

    def old_qids(self) -> list[str]:
        return [q for t in OLD_COUNTS for q in self.qids.get(t, [])]

    def new_qids(self) -> list[str]:
        return [q for t in NEW_TYPES for q in self.qids.get(t, [])]

    def qtype_of(self) -> dict[str, str]:
        return {q: t for t, ids in self.qids.items() for q in ids}


def _rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def draw_sample(
    items: Sequence[EvalItem], *, seed: int, counts: Mapping[str, int]
) -> dict[str, list[str]]:
    """Per type of `counts`, the ids of the `N` questions with the smallest rank, in file order."""
    drawn: dict[str, list[str]] = {}
    for qtype, n in counts.items():
        of_type = [i.qid for i in items if i.qtype == qtype]
        chosen = set(sorted(of_type, key=lambda q: (_rank(seed, q), q))[:n])
        drawn[qtype] = [q for q in of_type if q in chosen]
    return drawn


def build_question_list(
    items: Sequence[EvalItem], *, seed: int, date: str, old_counts: Mapping[str, int] = OLD_COUNTS
) -> QuestionList:
    qids = draw_sample(items, seed=seed, counts=old_counts)
    for qtype in NEW_TYPES:
        qids[qtype] = [i.qid for i in items if i.qtype == qtype]
    qids = {t: qids[t] for t in LIST_TYPES if qids.get(t)}
    return QuestionList(
        seed=seed, date=date, counts={t: len(v) for t, v in qids.items()}, qids=qids
    )


def read_question_list(path: str | Path | None = None) -> QuestionList:
    file = Path(path) if path else repo_path(LIST_PATH)
    return QuestionList.model_validate_json(file.read_text(encoding="utf-8"))


def write_question_list(path: str | Path, ql: QuestionList) -> None:
    text = json.dumps(ql.model_dump(), indent=2, ensure_ascii=False) + "\n"
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8", newline="\n")
