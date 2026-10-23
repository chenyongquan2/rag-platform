"""What one attempt at a regression question is, and when a question is a regression (spec EVAL-4).

An *attempt* is one run of one question by one system, read from its record and judge score:

- `correct`: judge score 3 (an ambiguous question: the system asked back);
- `wrong`: a clean answer or refusal that is not correct (the system decided it);
- `problem`: the run itself failed, so the attempt says nothing about the code: a record the run
  broke (`status` error, `generation_error`, an answer cut off by the total deadline, EVAL-0), a
  rerank timeout, a gateway or helper-model timeout or error in a record that is not correct, a
  retrieval route that failed, no record at all, an answered record without a judge score. A
  correct record is correct whatever else happened on the way.

`decide` applies the plan's lenient rule (section 10) to the attempts of one question, problems left
out: the first clean attempt correct passes; otherwise two of the first three clean attempts
correct is `recovered`, two wrong is a `regression`. Too many problems before it is decided is a
`run_problem`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag.eval.agentic.records import SystemRecord, retry_reason
from rag.eval.agentic.repeat.analysis import is_correct
from rag.eval.agentic.repeat.passes import Cell

Outcome = Literal["correct", "wrong", "problem"]
Status = Literal["pass", "recovered", "regression", "run_problem", "pending"]

GATEWAY_REASONS = frozenset(
    {
        "timeout",
        "llm_timeout",
        "llm_error",
        "retrieval_failed",
        "retrieval_timeout",
        "rerank_timeout",
        "not_ready",
        "internal_error",
    }
)
"""Reasons (a fallback `step:reason`, an error category) that come from the environment."""
FIRST_CLEAN = 3
"""Clean attempts the rule looks at: 2 of these 3 must be correct."""


@dataclass(frozen=True)
class Attempt:
    outcome: Outcome
    reason: str = ""


def _record_problem(record: SystemRecord) -> str | None:
    if (reason := retry_reason(record)) is not None:
        return reason
    if record.rerank_timeout:
        return "rerank_timeout"
    for category in record.error_categories:
        if category in GATEWAY_REASONS:
            return f"error category {category}"
    for fallback in record.fallbacks:
        if fallback.rpartition(":")[2] in GATEWAY_REASONS:
            return f"fallback {fallback}"
    if record.degraded and not record.fallbacks:
        return "a retrieval route failed (degraded)"
    return None


def classify_attempt(qtype: str, found: Cell | None) -> Attempt:
    """The attempt a record (with its judge score) is; `None` is a question without a record."""
    if found is None:
        return Attempt("problem", "no record")
    correct = is_correct(qtype, found)
    if correct:
        return Attempt("correct")
    if (reason := _record_problem(found.record)) is not None:
        return Attempt("problem", reason)
    if correct is None:
        return Attempt("problem", "answer not judged")
    return Attempt("wrong")


def decide(attempts: Sequence[Attempt], *, max_attempts: int = 5) -> Status:
    """The status of one question after `attempts` (in order); `pending` means run it again."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    clean = [a.outcome == "correct" for a in attempts if a.outcome != "problem"][:FIRST_CLEAN]
    if clean and clean[0]:
        return "pass"
    right, wrong = sum(clean), len(clean) - sum(clean)
    if right >= 2:
        return "recovered"
    if wrong >= 2:
        return "regression"
    return "run_problem" if len(attempts) >= max_attempts else "pending"
