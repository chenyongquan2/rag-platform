"""Spec EVAL-3 AC7, AC8, AC9: pass sources, run order, warm-up, cost estimate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.eval.agentic.repeat.passes import (
    check_budget,
    estimate_pass,
    load_pass,
    load_stored_pass,
    mean_costs,
    system_order,
    warm_up,
)
from rag.eval.agentic.sets import EvalItem
from rag.eval.retrieval.cli import UsageError
from rag.testing.repeat import cell, passes_from, qlist_for, record, write_pass


def test_system_order_alternates() -> None:
    assert system_order(2) == ("standard", "agentic")
    assert system_order(3) == ("agentic", "standard")
    assert system_order(4) == ("standard", "agentic")


def test_load_pass_overlay_and_restriction(tmp_path: Path) -> None:
    first, retry = tmp_path / "p3-07", tmp_path / "retry"
    data = passes_from(
        {
            "standard": {"single_hop:00": "t", "single_hop:01": "c", "no_answer:00": "r"},
            "agentic": {"single_hop:00": "c", "single_hop:01": "c", "no_answer:00": "r"},
        }
    )[0]
    write_pass(first, data)
    retried = {"standard": {"single_hop:00": cell("standard", "single_hop:00", "c")}, "agentic": {}}
    write_pass(retry, retried)

    pass1 = load_pass([first, retry], {"single_hop:00", "single_hop:01"})
    assert set(pass1["standard"]) == {"single_hop:00", "single_hop:01"}  # no_answer left out
    assert pass1["standard"]["single_hop:00"].score == 3  # the later directory wins
    assert pass1["standard"]["single_hop:00"].record.generation_error is None
    assert pass1["agentic"]["single_hop:00"].score == 3
    alone = load_pass([first], {"single_hop:00"})
    assert alone["standard"]["single_hop:00"].record.generation_error == "llm_timeout"
    assert load_pass([tmp_path / "missing"], None) == {"standard": {}, "agentic": {}}


def test_load_stored_pass_takes_old_and_new_questions_from_their_own_runs(tmp_path: Path) -> None:
    ql = qlist_for(["single_hop:00", "hard:00", "ambiguous:00"])
    both = passes_from(
        {
            "standard": {"single_hop:00": "c", "hard:00": "w", "ambiguous:00": "w"},
            "agentic": {"single_hop:00": "c", "hard:00": "c", "ambiguous:00": "c"},
        }
    )[0]
    old, new = tmp_path / "old", tmp_path / "new"
    write_pass(old, both)  # an old run that happens to hold a hard question: not used
    write_pass(new, both)
    pass1 = load_stored_pass([old], [new], ql)
    assert set(pass1["agentic"]) == {"single_hop:00", "hard:00", "ambiguous:00"}
    # Without the step-2 run the new questions are absent (the analysis lists them as incomplete).
    assert set(load_stored_pass([old], [], ql)["agentic"]) == {"single_hop:00"}


def test_mean_costs_by_system_and_type() -> None:
    records = [
        record("standard", "single_hop:00", "c", cny=0.004),
        record("standard", "single_hop:01", "c", cny=0.006),
        record("agentic", "two_hop:00", "c", cny=0.02),
    ]
    means = mean_costs(records)
    assert means["standard"]["single_hop"] == pytest.approx(0.005)
    assert means["agentic"]["two_hop"] == pytest.approx(0.02)


def _item(qid: str) -> EvalItem:
    qtype = qid.split(":")[0]
    return EvalItem(
        qid=qid,
        qtype=qtype,  # type: ignore[arg-type]
        set="x",
        question="q",
        answerable=qtype != "ambiguous",
        relevant_pages=["p"] if qtype != "ambiguous" else [],
    )


def test_estimate_uses_means_fallback_and_margin() -> None:
    means = {
        "standard": {"single_hop": 0.004, "two_hop": 0.008},
        "agentic": {"single_hop": 0.005, "two_hop": 0.016},
    }
    items = [_item("single_hop:00"), _item("hard:00"), _item("ambiguous:00")]
    todo = {"standard": items, "agentic": items[:2]}
    est = estimate_pass(
        todo,
        means,
        judge_calls=3,
        judge_cny_per_call=0.0014,
        margin=0.2,
    )
    # hard and ambiguous have no earlier records: the two-hop mean is used (plan section 14).
    assert est.by_system["standard"] == pytest.approx(0.004 + 0.008 + 0.008)
    assert est.by_system["agentic"] == pytest.approx(0.005 + 0.016)
    assert est.judge == pytest.approx(3 * 0.0014)
    base = 0.02 + 0.021 + 0.0042
    assert est.subtotal == pytest.approx(base)
    assert est.total == pytest.approx(base * 1.2)
    assert est.counts["standard"] == {"single_hop": 1, "hard": 1, "ambiguous": 1}
    assert "estimated cost" in "\n".join(est.lines()).lower()

    with pytest.raises(UsageError, match="earlier records"):
        estimate_pass(todo, {"standard": {}, "agentic": {}}, judge_calls=0)
    # A system with a mean over other types only falls back to its overall mean.
    odd = estimate_pass(
        {"standard": [_item("hard:00")], "agentic": []},
        {"standard": {"single_hop": 0.01}, "agentic": {}},
        judge_calls=0,
        margin=0.0,
    )
    assert odd.by_system["standard"] == pytest.approx(0.01)


def test_estimate_of_nothing_left_is_zero() -> None:
    est = estimate_pass(
        {"standard": [], "agentic": []}, {"standard": {}, "agentic": {}}, judge_calls=0
    )
    assert est.total == 0.0


def test_check_budget() -> None:
    est = estimate_pass(
        {"standard": [_item("single_hop:00")], "agentic": []},
        {"standard": {"single_hop": 1.0}, "agentic": {}},
        judge_calls=0,
        margin=0.0,
    )
    check_budget(est, spent=35.9, stop=37.0)  # 35.9 + 1.0 < 37
    check_budget(est, spent=36.0, stop=37.0)  # exactly at the line is allowed (not past it)
    with pytest.raises(UsageError, match="stop line"):
        check_budget(est, spent=36.01, stop=37.0)


class _Contexts:
    """Stands in for `RetrieverContexts`: calls are logged, rerank times out for named questions."""

    def __init__(self, timeouts: dict[str, int] | None = None) -> None:
        self.calls: list[str] = []
        self.timeouts = dict(timeouts or {})

    def __call__(self, item: Any) -> Any:
        self.calls.append(item.qid)
        left = self.timeouts.get(item.qid, 0)
        if left:
            self.timeouts[item.qid] = left - 1
        return SimpleNamespace(rerank_timeout=bool(left))


def test_warm_up_runs_retrieval_only() -> None:
    contexts = _Contexts({"b": 1})  # times out once, fine the second time
    items = [_item("single_hop:00"), _item("two_hop:00")]
    items[1] = items[1].model_copy(update={"qid": "b"})
    result = warm_up(contexts, items, say=lambda text: None)
    assert result.failed == 0 and result.rounds == 2
    assert contexts.calls[:2] == ["single_hop:00", "b"]
    assert contexts.calls.count("b") == 2 and contexts.calls.count("single_hop:00") == 1

    clean = _Contexts()
    done = warm_up(clean, items, say=lambda text: None)
    assert done.rounds == 1 and clean.calls == ["single_hop:00", "b"]


def test_warm_up_refuses_persistent_timeouts() -> None:
    contexts = _Contexts({"a": 9})
    with pytest.raises(UsageError, match="rerank"):
        warm_up(
            contexts, [_item("single_hop:00").model_copy(update={"qid": "a"})], say=lambda t: None
        )
    tolerant = _Contexts({"a": 9})
    result = warm_up(
        tolerant,
        [_item("single_hop:00").model_copy(update={"qid": "a"})],
        max_failed=1,
        say=lambda t: None,
    )
    assert result.failed == 1


def test_warm_up_counts_a_failed_search() -> None:
    def broken(item: Any) -> Any:
        raise RuntimeError("elasticsearch is down")

    with pytest.raises(UsageError, match=r"rerank|search"):
        warm_up(broken, [_item("single_hop:00")], say=lambda t: None)


def test_check_budget_refuses_a_spend_limit_that_is_already_used_up() -> None:
    est = estimate_pass(
        {"standard": [_item("single_hop:00")], "agentic": []},
        {"standard": {"single_hop": 0.01}, "agentic": {}},
        judge_calls=0,
    )
    with pytest.raises(UsageError, match="used up"):
        check_budget(est, spent=1.0, stop=37.0, run_cap=0.0)
    check_budget(est, spent=1.0, stop=37.0, run_cap=0.05)
