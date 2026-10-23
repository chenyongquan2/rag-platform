"""Spec EVAL-4 AC4, AC5, AC6: what one attempt is, and when a question is a regression."""

from __future__ import annotations

import pytest

from rag.eval.agentic.regress.verdict import Attempt, classify_attempt, decide
from rag.eval.agentic.repeat.passes import Cell
from rag.testing.repeat import cell

C, W, P = Attempt("correct"), Attempt("wrong"), Attempt("problem", "run problem")


def _with(system: str, qid: str, outcome: str, **fields: object) -> Cell:
    base = cell(system, qid, outcome)
    return Cell(record=base.record.model_copy(update=fields), score=base.score)


def test_decide_pass_recovered_regression() -> None:
    assert decide([C]) == "pass"
    assert decide([W]) == "pending"
    assert decide([W, C]) == "pending"
    assert decide([W, C, C]) == "recovered"
    assert decide([W, C, W]) == "regression"
    assert decide([W, W]) == "regression"  # decided at once: two wrong cannot be undone
    assert decide([W, W, C]) == "regression"
    assert decide([]) == "pending"


def test_run_problems_are_not_wrong() -> None:
    cases = {
        "deadline cut": cell("agentic", "hard:00", "x"),
        "model timeout": cell("agentic", "hard:00", "t"),
        "run error": cell("agentic", "hard:00", "e"),
        "not judged": cell("agentic", "hard:00", "u"),
        "no record": None,
        "rerank timeout": _with("agentic", "hard:00", "w", rerank_timeout=True),
        "helper timeout": _with("agentic", "hard:00", "w", fallbacks=["judge:timeout"]),
        "gateway error": _with("agentic", "hard:00", "w", fallbacks=["rewrite:llm_error"]),
        "error category": _with("agentic", "hard:00", "w", error_categories=["llm_timeout"]),
        "route failed": _with("standard", "hard:00", "w", degraded=True),
        "retrieval failed": _with(
            "agentic", "hard:00", "w", fallbacks=["retrieve:retrieval_failed"]
        ),
    }
    for name, found in cases.items():
        attempt = classify_attempt("hard", found)
        assert attempt.outcome == "problem", name
        assert attempt.reason, name
    # A refusal that nothing broke is a wrong answer: the system decided it.
    assert classify_attempt("hard", cell("agentic", "hard:00", "r")).outcome == "wrong"
    assert classify_attempt("hard", cell("agentic", "hard:00", "w")).outcome == "wrong"
    assert classify_attempt("hard", cell("agentic", "hard:00", "c")).outcome == "correct"
    # A degraded record whose helper fell back for a non-gateway reason is not a run problem.
    quiet = _with("agentic", "hard:00", "w", degraded=True, fallbacks=["rewrite:empty"])
    assert classify_attempt("hard", quiet).outcome == "wrong"
    # Ambiguous questions are judged by rule: answering instead of asking back is wrong.
    assert classify_attempt("ambiguous", cell("agentic", "ambiguous:00", "c")).outcome == "correct"
    assert classify_attempt("ambiguous", cell("agentic", "ambiguous:00", "w")).outcome == "wrong"


def test_correct_with_timeouts_is_correct() -> None:
    noisy = _with(
        "agentic",
        "hard:00",
        "c",
        degraded=True,
        rerank_timeout=True,
        fallbacks=["judge:timeout"],
        error_categories=["degraded", "rerank_timeout"],
    )
    assert classify_attempt("hard", noisy) == Attempt("correct")


def test_problem_attempts_are_replaced() -> None:
    assert decide([P]) == "pending"
    assert decide([P, C]) == "pass"  # a problem carries no information
    assert decide([P, P, C]) == "pass"
    assert decide([W, P, C, C]) == "recovered"
    assert decide([W, P, W]) == "regression"
    assert decide([C, P]) == "pass"


def test_undecided_is_a_run_problem() -> None:
    attempts = [W, P, P, P]
    assert decide(attempts, max_attempts=5) == "pending"
    assert decide([*attempts, P], max_attempts=5) == "run_problem"
    assert decide([P] * 5, max_attempts=5) == "run_problem"
    # Not a regression: one wrong answer is not enough, however many attempts failed after it.
    assert decide([W, C, P, P, P], max_attempts=5) == "run_problem"
    with pytest.raises(ValueError):
        decide([C], max_attempts=0)
