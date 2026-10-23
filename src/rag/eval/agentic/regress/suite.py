"""The regression question set and its baseline (spec EVAL-4, plan step 4).

    rs, baseline = build_suite(passes, qlist=ql, items=items, seed=20261023, date="2026-10-23",
                               max_per_type={"single_hop": 12}, sources={...})

A question is *stable for a system* when it is correct in every pass (judge score 3; an ambiguous
question: asked back), read through the same functions as `repeat-analyze`. The set holds questions
stable for the graph (`ask_docs`, the system it guards); `stable_for` also names standard RAG when
the question is stable for it too. A type with more stable questions than its cap is drawn down by
`sha256("{seed}:{qid}")` rank (so the draw depends on neither file order nor the Python version); a
scarcer type is taken whole.

Two committed files, never edited once used (a new version is a new file):

- the set: version, seed, date, sources, counts, and per question its type, `stable_for` and a
  fingerprint of question, reference and gold pages, so a changed dataset is noticed;
- the baseline: every pass of every question for both systems (score, correct, status, stop reason,
  time, CNY), which is also where the cost estimate of a later run comes from.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.config import repo_path
from rag.eval.agentic.records import SYSTEMS
from rag.eval.agentic.repeat.analysis import is_correct
from rag.eval.agentic.repeat.passes import Cell, PassData
from rag.eval.agentic.repeat.sample import LIST_TYPES, QuestionList, draw_sample
from rag.eval.agentic.sets import EvalItem

SET_NAME = "regression_v1"
SET_PATH = f"evals/datasets/hf_docs/{SET_NAME}.json"
BASELINE_PATH = f"evals/datasets/hf_docs/{SET_NAME}.baseline.json"
DEFAULT_SEED = 20261023
DEFAULT_DATE = "2026-10-23"
DEFAULT_MAX_PER_TYPE: dict[str, int] = {"single_hop": 12}
"""Single-hop questions never moved between passes, so they are capped to leave room for the types
that do move; every other type is taken whole."""
GUARDED = "agentic"
METHOD = (
    "questions correct in every pass for the graph (judge score 3; ambiguous: asked back); a type "
    "with more of them than its cap is drawn by sha256('seed:qid') rank, the others are taken whole"
)


class SuiteQuestion(BaseModel):
    qid: str
    qtype: str
    stable_for: list[str]
    """`agentic` always; `standard` too when standard RAG was correct in every pass."""
    fingerprint: str


class RegressionSet(BaseModel):
    version: int = 1
    name: str = SET_NAME
    kb: str = "hf_docs"
    system: str = GUARDED
    seed: int
    date: str
    method: str = METHOD
    passes: int
    sources: dict[str, Any] = Field(default_factory=dict)
    """Where the passes it was chosen from are (paths on the owner's machine, the question list)."""
    stable_available: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Per system and type: how many questions were stable (before the cap)."""
    counts: dict[str, int] = Field(default_factory=dict)
    questions: list[SuiteQuestion] = Field(default_factory=list)

    def qids(self) -> list[str]:
        return [q.qid for q in self.questions]

    def qids_for(self, system: str) -> list[str]:
        return [q.qid for q in self.questions if system in q.stable_for]

    def qtype_of(self) -> dict[str, str]:
        return {q.qid: q.qtype for q in self.questions}


class BaselinePass(BaseModel):
    pass_no: int
    correct: bool
    score: int | None = None
    status: str = ""
    stop_reason: str | None = None
    total_ms: float = 0.0
    cny: float = 0.0
    """The answer's cost plus the judge's call for it."""


class Baseline(BaseModel):
    version: int = 1
    set_name: str = SET_NAME
    passes: int
    entries: dict[str, dict[str, list[BaselinePass]]] = Field(default_factory=dict)
    """question id -> system -> its passes, in order."""

    def mean_cny(self, system: str, qid: str) -> float | None:
        rows = self.entries.get(qid, {}).get(system, [])
        return sum(r.cny for r in rows) / len(rows) if rows else None


def fingerprint(item: EvalItem) -> str:
    text = "\0".join([item.question, item.reference, ",".join(item.relevant_pages), item.requires])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _baseline_pass(number: int, qtype: str, cell: Cell) -> BaselinePass:
    return BaselinePass(
        pass_no=number,
        correct=bool(is_correct(qtype, cell)),
        score=cell.score,
        status=cell.record.status,
        stop_reason=cell.record.stop_reason,
        total_ms=cell.record.total_ms,
        cny=round(cell.record.usage.cny + cell.judge_cny, 6),
    )


def _stable(passes: Sequence[PassData], system: str, qid: str, qtype: str) -> bool:
    cells = [data.get(system, {}).get(qid) for data in passes]
    return all(c is not None and is_correct(qtype, c) is True for c in cells)


def build_suite(
    passes: Sequence[PassData],
    *,
    qlist: QuestionList,
    items: Sequence[EvalItem],
    seed: int,
    date: str,
    max_per_type: Mapping[str, int] = DEFAULT_MAX_PER_TYPE,
    sources: Mapping[str, Any] | None = None,
    name: str = SET_NAME,
) -> tuple[RegressionSet, Baseline]:
    """The set and its baseline from the passes of the repeated runs (pass 1 first)."""
    by_qid = {i.qid: i for i in items}
    qtypes = qlist.qtype_of()
    stable = {
        system: {
            t: [q for q in qlist.qids.get(t, []) if _stable(passes, system, q, t)]
            for t in LIST_TYPES
        }
        for system in SYSTEMS
    }
    chosen: list[str] = []
    for qtype in LIST_TYPES:
        pool = stable["agentic"][qtype]
        cap = max_per_type.get(qtype)
        if cap is not None and len(pool) > cap:
            drawn = draw_sample([by_qid[q] for q in pool], seed=seed, counts={qtype: cap})
            pool = drawn[qtype]
        chosen += pool
    standard_stable = {q for ids in stable["standard"].values() for q in ids}
    questions = [
        SuiteQuestion(
            qid=q,
            qtype=qtypes[q],
            stable_for=["agentic", "standard"] if q in standard_stable else ["agentic"],
            fingerprint=fingerprint(by_qid[q]),
        )
        for q in chosen
    ]
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.qtype] = counts.get(q.qtype, 0) + 1
    rs = RegressionSet(
        name=name,
        seed=seed,
        date=date,
        passes=len(passes),
        sources=dict(sources or {}),
        stable_available={s: {t: len(v) for t, v in by.items() if v} for s, by in stable.items()},
        counts=counts,
        questions=questions,
    )
    entries: dict[str, dict[str, list[BaselinePass]]] = {}
    for q in questions:
        entries[q.qid] = {
            system: [
                _baseline_pass(n, q.qtype, data[system][q.qid])
                for n, data in enumerate(passes, start=1)
                if q.qid in data.get(system, {})
            ]
            for system in SYSTEMS
        }
    return rs, Baseline(set_name=name, passes=len(passes), entries=entries)


def check_items(rs: RegressionSet, items: Sequence[EvalItem]) -> None:
    """Raise `ValueError` when a question of the set is gone from the question sets or its text,
    reference or gold pages changed since the set was built."""
    by_qid = {i.qid: i for i in items}
    missing = [q.qid for q in rs.questions if q.qid not in by_qid]
    if missing:
        raise ValueError(f"questions of the set are not in the question sets: {missing[:5]}")
    changed = [q.qid for q in rs.questions if fingerprint(by_qid[q.qid]) != q.fingerprint]
    if changed:
        raise ValueError(
            f"questions changed since the set {rs.name} was built (a new set version is needed): "
            f"{changed[:5]}"
        )


def _path(path: str | Path) -> Path:
    return repo_path(path)


def _write(path: str | Path, model: BaseModel) -> None:
    file = _path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model.model_dump(), indent=2, ensure_ascii=False) + "\n"
    file.write_text(text, encoding="utf-8", newline="\n")


def write_suite(path: str | Path, rs: RegressionSet) -> None:
    _write(path, rs)


def write_baseline(path: str | Path, baseline: Baseline) -> None:
    _write(path, baseline)


def read_suite(path: str | Path = SET_PATH) -> RegressionSet:
    return RegressionSet.model_validate_json(_path(path).read_text(encoding="utf-8"))


def read_baseline(path: str | Path = BASELINE_PATH) -> Baseline:
    return Baseline.model_validate_json(_path(path).read_text(encoding="utf-8"))
