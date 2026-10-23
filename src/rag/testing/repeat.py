"""Fake run records for the repeated-run tests (spec EVAL-3): no model, no files unless asked.

One character per pass describes what a system did with a question:

    c  correct (judge score 3; an ambiguous question: the system asked back)
    w  answered, not correct (score 2; ambiguous: it answered instead of asking)
    x  answer cut off by the total deadline (agentic only; no answer step ran)
    t  answer model timed out (`generation_error = llm_timeout`)
    e  the run failed (`status = error`)
    r  clean refusal (not found, nothing broke)
    u  answered but not judged (no judge line)
    n  no record at all

    spec = {"standard": {"single_hop:00": "ccw"}, "agentic": {"single_hop:00": "ccc"}}
    passes = passes_from(spec)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rag.eval.agentic.records import JudgeLine, SystemRecord, append_line
from rag.eval.agentic.repeat.passes import Cell, PassData
from rag.eval.agentic.repeat.sample import QuestionList
from rag.eval.agentic.sets import NOT_ANSWERABLE
from rag.eval.answer.record import Usage

SCORES = {"c": 3, "w": 2, "x": 1, "t": 1, "e": 1, "r": 1, "u": None}


def qtype_of(qid: str) -> str:
    return qid.split(":")[0]


def record(system: str, qid: str, outcome: str, *, cny: float = 0.01) -> SystemRecord:
    qtype = qtype_of(qid)
    fields: dict[str, object] = {
        "system": system,
        "qid": qid,
        "qtype": qtype,
        "question": f"Question {qid}?",
        "reference": "Reference.",
        "answerable": qtype not in NOT_ANSWERABLE,
        "usage": Usage(calls=1, cny=cny),
        "total_ms": 1000.0,
        "steps_ms": {"search": 10.0, "answer": 20.0},
    }
    if qtype == "ambiguous" and outcome == "c":
        fields.update(status="need_clarification", clarification=True)
    elif outcome in ("c", "w", "u"):
        fields.update(status="answered", found=True, answered=True, text="Use the option.")
    elif outcome == "x":
        fields.update(status="not_found", stop_reason="deadline", steps_ms={"search": 10.0})
    elif outcome == "t":
        fields.update(status="not_found", generation_error="llm_timeout")
    elif outcome == "e":
        fields.update(status="error", error="RuntimeError: boom")
    elif outcome == "r":
        fields.update(status="not_found")
    else:
        raise ValueError(f"unknown outcome {outcome!r}")
    return SystemRecord(**fields)  # type: ignore[arg-type]


def cell(system: str, qid: str, outcome: str, *, cny: float = 0.01) -> Cell:
    return Cell(record=record(system, qid, outcome, cny=cny), score=SCORES.get(outcome))


def passes_from(spec: Mapping[str, Mapping[str, str]]) -> list[PassData]:
    """`spec[system][qid]` is one outcome character per pass; every row has the same length."""
    length = len(next(iter(next(iter(spec.values())).values())))
    out: list[PassData] = [{"standard": {}, "agentic": {}} for _ in range(length)]
    for system, rows in spec.items():
        for qid, outcomes in rows.items():
            assert len(outcomes) == length, (system, qid)
            for index, outcome in enumerate(outcomes):
                if outcome != "n":
                    out[index][system][qid] = cell(system, qid, outcome)
    return out


def qlist_for(qids: list[str], *, seed: int = 1, date: str = "2026-10-23") -> QuestionList:
    grouped: dict[str, list[str]] = {}
    for qid in qids:
        grouped.setdefault(qtype_of(qid), []).append(qid)
    return QuestionList(
        seed=seed,
        date=date,
        method="test",
        counts={t: len(v) for t, v in grouped.items()},
        qids=grouped,
    )


def write_pass(directory: Path, data: PassData, *, only: set[str] | None = None) -> None:
    """Write one pass as a records directory (`standard/`, `agentic/`, `judge.jsonl`)."""
    for system, cells in data.items():
        for qid, c in cells.items():
            if only is not None and qid not in only:
                continue
            append_line(directory / system / "records.jsonl", c.record)
            if c.record.answerable and c.score is not None:
                append_line(
                    directory / "judge.jsonl",
                    JudgeLine(
                        system=system,  # type: ignore[arg-type]
                        qid=qid,
                        run=1,
                        score=c.score,
                        by_rule=None if c.record.answered else "no_answer",
                    ),
                )
