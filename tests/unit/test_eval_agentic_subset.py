"""Spec P3-06: the before-and-after tables of the re-measurement (pure functions, no model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.eval.agentic.cli import main
from rag.eval.agentic.records import (
    HookEvent,
    JudgeLine,
    RoundInfo,
    SystemRecord,
    append_line,
    judge_file,
    system_file,
)
from rag.eval.agentic.subset import (
    compare_subset,
    group_of,
    groups,
    reached_answer_model,
    row_of,
)
from rag.eval.answer.record import Usage


def rec(system: str, qid: str, qtype: str = "single_hop", **kw: Any) -> SystemRecord:
    answerable = qtype != "no_answer"
    fields: dict[str, Any] = {
        "system": system,
        "qid": qid,
        "qtype": qtype,
        "question": f"Question {qid}?",
        "relevant_pages": [f"g{qid}"] if answerable else [],
        "answerable": answerable,
        "status": "answered" if answerable else "not_found",
        "found": answerable,
        "answered": answerable,
        "text": "an answer" if answerable else "",
        "rounds": 1,
        "total_ms": 8000.0,
        "usage": Usage(calls=1, cny=0.002),
        "usage_by_profile": {"online": Usage(calls=1, cny=0.002)},
    }
    fields.update(kw)
    return SystemRecord(**fields)


def verdict(value: str, round_no: int = 1) -> HookEvent:
    return HookEvent(step="judge", round=round_no, data={"verdict": value, "skipped": 0})


def test_group_of_follows_what_happened_in_p3_05() -> None:
    crashed = rec("agentic", "n1", "no_answer", status="error", answered=False)
    needed = rec("agentic", "q1", events=[verdict("partly_enough")])
    fine = rec("agentic", "q2", events=[verdict("enough")])
    no_answer = rec("agentic", "n2", "no_answer")
    two_hop = rec("agentic", "t1", "two_hop", events=[verdict("enough")])
    assert [group_of(r) for r in (crashed, needed, fine, no_answer, two_hop)] == [
        "crashed",
        "another_round",
        "single_hop",
        "no_answer",
        "other",
    ]
    # A no-answer question whose first verdict was "not enough" is not an "answerable" one.
    assert (
        group_of(rec("agentic", "n3", "no_answer", events=[verdict("not_enough")])) == "no_answer"
    )
    before = {r.qid: r for r in (crashed, needed, fine, no_answer)}
    assert groups(before, ["q2", "n1", "q1", "n2"]) == {
        "crashed": ["n1"],
        "another_round": ["q1"],
        "no_answer": ["n2"],
        "single_hop": ["q2"],
    }


def test_reached_answer_model_for_both_kinds_of_record() -> None:
    assert reached_answer_model(rec("agentic", "a"))
    assert not reached_answer_model(rec("agentic", "a", usage_by_profile={}))
    assert not reached_answer_model(
        rec("agentic", "a", usage_by_profile={"cheap": Usage(calls=2, cny=0.007)})
    )
    # Standard RAG makes no other call than the answer.
    assert reached_answer_model(rec("standard", "a", usage_by_profile={}, usage=Usage(calls=1)))
    assert not reached_answer_model(rec("standard", "a", usage_by_profile={}, usage=Usage()))


def test_row_counts_stages_time_and_cost() -> None:
    records = [
        rec("agentic", "q1", total_ms=6000.0, rounds=1),
        rec(
            "agentic",
            "q2",
            total_ms=20_000.0,
            rounds=2,
            stop_reason="deadline",
            rerank_timeout=True,
            error_categories=["rerank_timeout", "retrieval_timeout", "agent_deadline"],
            usage=Usage(calls=3, cny=0.012),
            usage_by_profile={"cheap": Usage(calls=2, cny=0.01), "online": Usage(calls=1)},
        ),
        rec(
            "agentic",
            "q3",
            status="error",
            answered=False,
            found=False,
            total_ms=18_000.0,
            usage_by_profile={},
        ),
    ]
    row = row_of("g", "B", records, {"q1": 3, "q2": 2})
    assert (row.n, row.errors, row.answered, row.refused) == (3, 1, 2, 1)
    assert (row.two_or_more_rounds, row.deadline_stops, row.retrieval_errors) == (1, 1, 1)
    assert (row.rerank_timeouts, row.over_15s, row.over_30s) == (1, 2, 0)
    assert (row.correct, row.scored) == (1, 2)
    assert row.cny_total == pytest.approx(0.016)
    assert row.reached_answer_model == 2 and row.online_calls_mean == pytest.approx(2 / 3)
    assert row.p50_s == pytest.approx(18.0)


def _before_and_after() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    standard = {
        "n1": rec("standard", "n1", "no_answer", usage_by_profile={}, usage=Usage()),
        "q1": rec("standard", "q1", total_ms=5000.0),
    }
    before = {
        "n1": rec("agentic", "n1", "no_answer", status="error", answered=False, total_ms=18_000.0),
        "q1": rec("agentic", "q1", events=[verdict("enough")], total_ms=13_000.0),
        "x": rec("agentic", "x"),  # not in the new run: left out
    }
    after = {
        "n1": rec(
            "agentic",
            "n1",
            "no_answer",
            total_ms=7000.0,
            usage_by_profile={},
            usage=Usage(),
            events=[verdict("not_enough")],
        ),
        "q1": rec("agentic", "q1", events=[verdict("enough")], total_ms=6000.0),
    }
    return standard, before, after


def test_compare_subset_report_lists_groups_and_every_question() -> None:
    standard, before, after = _before_and_after()
    text = compare_subset(
        standard,
        before,
        after,
        {"A standard": {"q1": 3}, "B before": {"q1": 3}, "B after": {"q1": 2}},
    )
    assert text.startswith("Questions in both runs: 2; groups: crashed 1, single_hop 1")
    # One row per group and system, then the pooled rows.
    rows = [ln for ln in text.splitlines() if ln.startswith(("| crashed |", "| single_hop |"))]
    assert len(rows) == 6 and sum(ln.startswith("| all |") for ln in text.splitlines()) == 3
    # The crashed question: an error before, a refusal without the answer model after.
    assert "error, r1, ans-model, 18.0s" in text
    assert "not_found, r1, no ans-model, 7.0s" in text
    assert "judge 2" in text and "judge 3" in text
    assert "x |" not in text
    assert "First-round verdicts of B after: {'not_enough': 1, 'enough': 1}" in text


def test_compare_command_reads_two_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    standard, before, after = _before_and_after()
    old_dir, new_dir = tmp_path / "p3-05", tmp_path / "p3-06"
    for system, records, out in (
        ("standard", standard, old_dir),
        ("agentic", before, old_dir),
        ("agentic", after, new_dir),
    ):
        for record in records.values():
            append_line(system_file(out, system), record)
    judge = judge_file(old_dir)
    judge.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        JudgeLine(system="standard", qid="q1", run=1, score=3),
        JudgeLine(system="agentic", qid="q1", run=1, score=3),
    ]
    judge.write_text(
        "".join(x.model_dump_json() + "\n" for x in lines), encoding="utf-8", newline="\n"
    )
    md = tmp_path / "out.md"
    assert (
        main(["compare", "--before", str(old_dir), "--after", str(new_dir), "--md", str(md)]) == 0
    )
    assert "Questions in both runs: 2" in md.read_text("utf-8")
    assert "crashed" in capsys.readouterr().out
    # No records in the new directory: a usage error, not a crash.
    assert main(["compare", "--before", str(old_dir), "--after", str(tmp_path / "none")]) == 2


def test_round_info_is_used_for_the_gold_page() -> None:
    standard, before, after = _before_and_after()
    after["q1"] = after["q1"].model_copy(
        update={"round_info": [RoundInfo(round=1, sub_question="s", pages=["gq1"])]}
    )
    text = compare_subset(standard, before, after, {})
    assert "gold yes" in text
