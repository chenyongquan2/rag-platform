"""Spec EVAL-0: answers cut by the total deadline are told apart from clean refusals, and
`--retry-errors` re-runs generation errors and deadline-cut records (evaluation tool only).

Pure functions of records plus scripted retrieval and fake chat models; nothing calls a network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime
from rag.eval.agentic import cli
from rag.eval.agentic.analysis import judge_scores, per_type_summary
from rag.eval.agentic.judging import judge_records
from rag.eval.agentic.records import (
    JudgeLine,
    SystemRecord,
    append_line,
    is_deadline_truncated,
    is_retryable,
    read_judge_lines,
    read_records,
    retry_reason,
    system_file,
)
from rag.eval.agentic.report import render_report
from rag.eval.agentic.runner import Tracer, run_agentic, run_standard
from rag.eval.agentic.runs import compare_runs, render_compare_runs
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.judge import JudgeConfig
from rag.eval.answer.runner import RetrieverContexts
from rag.observe import shutdown_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings, kb_config, pages_result
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat

KB = "hf_docs"
PRE_ANSWER = {"route": 10.0, "rewrite": 800.0, "retrieve": 2000.0, "judge": 900.0}
WITH_ANSWER = {**PRE_ANSWER, "answer": 4000.0, "verify": 5.0}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def rec(system: str, qid: str, qtype: str = "single_hop", **kw: Any) -> SystemRecord:
    answerable = qtype != "no_answer"
    fields: dict[str, Any] = {
        "system": system,
        "qid": qid,
        "qtype": qtype,
        "question": f"Question {qid}?",
        "reference": "ref" if answerable else "",
        "relevant_pages": [f"{KB}:p1.md"] if answerable else [],
        "answerable": answerable,
        "status": "answered",
        "found": True,
        "answered": True,
        "text": "an answer",
    }
    fields.update(kw)
    return SystemRecord(**fields)


def refusal(system: str, qid: str, qtype: str = "no_answer", **kw: Any) -> SystemRecord:
    """A record with no answer, status not_found, no error: what a clean refusal looks like."""
    fields: dict[str, Any] = {"status": "not_found", "found": False, "answered": False, "text": ""}
    fields.update(kw)
    return rec(system, qid, qtype, **fields)


def cut(qid: str, qtype: str = "two_hop", **kw: Any) -> SystemRecord:
    """An agentic record as P3-07 stored two_hop:04: not_found, deadline, no answer step."""
    fields: dict[str, Any] = {
        "stop_reason": "deadline",
        "steps_ms": dict(PRE_ANSWER),
        "total_ms": 30_015.0,
        "rounds": 1,
    }
    fields.update(kw)
    return refusal("agentic", qid, qtype, **fields)


# ---- AC1, AC2: the rule ---------------------------------------------------------------------


def test_is_deadline_truncated_rule() -> None:
    assert is_deadline_truncated(cut("q1"))
    # The same stop reason with an answer step: the answer model ran and said it found nothing.
    assert not is_deadline_truncated(cut("q2", steps_ms=dict(WITH_ANSWER)))
    # Other statuses and stop reasons never count, nor does the standard system.
    assert not is_deadline_truncated(cut("q3", status="answered", found=True, answered=True))
    assert not is_deadline_truncated(cut("q4", status="need_clarification", clarification=True))
    assert not is_deadline_truncated(cut("q5", status="error", error="X: y"))
    assert not is_deadline_truncated(cut("q6", stop_reason="max_rounds"))
    assert not is_deadline_truncated(cut("q7", stop_reason=None))
    standard = refusal("standard", "q8", stop_reason="deadline", steps_ms={"retrieval": 1.0})
    assert not is_deadline_truncated(standard)


def test_retry_reason_and_is_retryable() -> None:
    cases = {
        "error": rec("standard", "a", status="error", found=False, answered=False, error="X: y"),
        "generation_error": refusal("standard", "b", generation_error="llm_timeout"),
        "deadline_truncated": cut("c"),
    }
    for reason, record in cases.items():
        assert retry_reason(record) == reason and is_retryable(record)
    # A model timeout on a record that was also cut short: the recorded failure wins.
    assert retry_reason(cut("d", generation_error="llm_timeout")) == "generation_error"
    # A clean refusal (even with a deadline stop) and an answer are final.
    clean = refusal("agentic", "e", stop_reason="deadline", steps_ms=dict(WITH_ANSWER))
    for record in (clean, refusal("standard", "h"), rec("agentic", "f"), rec("standard", "g")):
        assert retry_reason(record) is None and not is_retryable(record)


# ---- AC3, AC4: the numbers ------------------------------------------------------------------


def test_truncated_is_not_a_clean_refusal() -> None:
    standard = [refusal("standard", f"n{i}") for i in range(10)]
    agentic = [
        cut(f"n{i}", "no_answer")
        if i < 2
        else refusal("agentic", f"n{i}", stop_reason="deadline", steps_ms=dict(WITH_ANSWER))
        for i in range(10)
    ]
    summary = per_type_summary([*standard, *agentic], [])["no_answer"]
    assert summary.refusal is not None and summary.clean_refusal is not None
    assert (summary.refusal.a.count, summary.refusal.b.count) == (10, 10)  # no answer was given
    assert (summary.clean_refusal.a.count, summary.clean_refusal.b.count) == (10, 8)
    assert summary.truncated == {"standard": 0, "agentic": 2}
    assert summary.errors == {"standard": 0, "agentic": 0}  # a separate category, not an error


def test_truncated_answerable_is_counted_separately() -> None:
    records = [
        rec("standard", "t1", "two_hop"),
        rec("standard", "t2", "two_hop"),
        cut("t1"),
        rec("agentic", "t2", "two_hop"),
    ]
    lines = [
        JudgeLine(system="standard", qid="t1", run=1, score=3),
        JudgeLine(system="standard", qid="t2", run=1, score=3),
        JudgeLine(system="agentic", qid="t1", run=1, score=1, by_rule="no_answer"),
        JudgeLine(system="agentic", qid="t2", run=1, score=3),
    ]
    summary = per_type_summary(records, lines)
    assert summary["two_hop"].truncated == {"standard": 0, "agentic": 1}
    assert summary["answerable"].truncated == {"standard": 0, "agentic": 1}
    # It stays a miss for the correct rate (no answer was given); the report lists it apart.
    correct = summary["two_hop"].correct
    assert correct is not None and (correct.a.count, correct.b.count) == (2, 1)


# ---- AC5, AC6: the reports ------------------------------------------------------------------


def test_report_lists_deadline_truncated_separately() -> None:
    records = [
        rec("standard", "two_hop:04", "two_hop"),
        cut("two_hop:04"),
        refusal("standard", "n1"),
        cut("n1", "no_answer"),
    ]
    text = render_report(records, [JudgeLine(system="standard", qid="two_hop:04", run=1, score=3)])
    assert "## Answers cut by the deadline" in text
    section = text.split("## Answers cut by the deadline", 1)[1].split("\n## ", 1)[0]
    assert "two_hop:04" in section and "n1" in section and "30.0 s" in section
    assert "refused cleanly" in text  # the clean-refusal row is still there


def test_compare_runs_truncated_not_clean() -> None:
    before = [refusal("agentic", "n1"), refusal("agentic", "n2")]
    after = [cut("n1", "no_answer"), before[1]]
    report = compare_runs(before, after, [], [], system="agentic")
    assert report.refusal is not None and report.clean_refusal is not None
    assert (report.refusal.a.count, report.refusal.b.count) == (2, 2)
    assert (report.clean_refusal.a.count, report.clean_refusal.b.count) == (2, 1)
    rows = {(r.label, r.qtype): r for r in report.cost}
    assert (rows["before", "all"].truncated, rows["after", "all"].truncated) == (0, 1)
    assert "cut by the deadline" in render_compare_runs(report)


# ---- AC7: --retry-errors --------------------------------------------------------------------


def _item(qid: str) -> EvalItem:
    return EvalItem(
        qid=qid,
        qtype="single_hop",
        set="x",
        question=f"Question {qid}?",
        reference="ref",
        relevant_pages=[f"{KB}:p1.md"],
        answerable=True,
    )


def _answer_model(tmp_path: Path) -> LedgerStreamingChat:
    return LedgerStreamingChat(
        default_response="Use it [1].", ledger_path=str(tmp_path / "ledger.jsonl")
    )


def test_retry_errors_reruns_generation_errors(tmp_path: Path) -> None:
    out = tmp_path / "standard.jsonl"
    items = [_item(q) for q in ("timeout", "clean", "done")]
    for record in (
        refusal("standard", "timeout", "single_hop", generation_error="llm_timeout"),
        refusal("standard", "clean", "single_hop"),
        rec("standard", "done"),
    ):
        append_line(out, record)
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1]))

    def run(retry: bool) -> Any:
        return asyncio.run(
            run_standard(
                items,
                records_path=out,
                contexts=RetrieverContexts(backend, kb=KB, top_k=8),
                kb_config=kb_config(KB),
                model=_answer_model(tmp_path),
                retry_errors=retry,
            )
        )

    assert run(False).ran == 0  # without the flag nothing is re-run
    stats = run(True)
    assert (stats.ran, stats.skipped) == (1, 2)
    records = {r.qid: r for r in read_records(out)}
    assert records["timeout"].answered and records["timeout"].generation_error is None
    assert not records["clean"].answered  # a real refusal is final
    assert [r.qid for r in read_records(out)] == ["clean", "done", "timeout"]


def test_retry_errors_reruns_deadline_truncated(tmp_path: Path) -> None:
    out = tmp_path / "agentic.jsonl"
    items = [_item(q) for q in ("cut", "refused", "done")]
    for record in (
        cut("cut", "single_hop"),
        cut("refused", "single_hop", steps_ms=dict(WITH_ANSWER)),  # the answer step ran
        rec("agentic", "done"),
    ):
        append_line(out, record)
    ledger = str(tmp_path / "ledger.jsonl")
    tracer = Tracer()

    def run(retry: bool) -> Any:
        return asyncio.run(
            run_agentic(
                items,
                records_path=out,
                runtime=AgentRuntime.from_settings(agent_settings(), hooks=tracer.hooks()),
                retriever=ScriptedSearchBackend(KB, default=pages_result(KB, [1])),
                kb_config=kb_config(KB, judge=False),
                tracer=tracer,
                model=_answer_model(tmp_path),
                helper_model=LedgerChat(default_response="x", ledger_path=ledger, profile="cheap"),
                judge_model=LedgerChat(default_response="x", ledger_path=ledger, profile="cheap"),
                retry_errors=retry,
            )
        )

    assert run(False).ran == 0
    stats = run(True)
    assert (stats.ran, stats.skipped) == (1, 2)
    records = {r.qid: r for r in read_records(out)}
    assert records["cut"].answered and not is_deadline_truncated(records["cut"])
    assert records["refused"].stop_reason == "deadline" and not records["refused"].answered


# ---- AC8: the judge ------------------------------------------------------------------------


def test_judge_rejudges_stale_rule_line(tmp_path: Path) -> None:
    out = tmp_path / "judge.jsonl"
    # q1 was cut by the deadline (scored 1 by rule), then re-run and answered.
    append_line(out, JudgeLine(system="agentic", qid="q1", run=1, score=1, by_rule="no_answer"))
    append_line(out, JudgeLine(system="agentic", qid="q2", run=1, score=1, by_rule="no_answer"))
    append_line(out, JudgeLine(system="agentic", qid="q3", run=1, score=2))
    records = [
        rec("agentic", "q1"),  # answered now: the old rule line is stale
        cut("q2", "single_hop"),  # still not answered: the rule line stands
        rec("agentic", "q3"),  # judged by the model before: stays
    ]
    model = LedgerChat(
        default_response="Feedback: right. [RESULT] 3",
        ledger_path=str(tmp_path / "ledger.jsonl"),
        profile="offline",
    )
    new = asyncio.run(
        judge_records(records, run=1, model=model, config=JudgeConfig(), out_path=out)
    )
    assert [x.qid for x in new] == ["q1"] and len(model.calls) == 1
    scores = judge_scores(read_judge_lines(out), "agentic")
    assert scores == {"q1": 3, "q2": 1, "q3": 2}  # the newer line of q1 wins


# ---- AC9: the classify command --------------------------------------------------------------


def test_classify_lists_retryable_by_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`classify` re-reads stored records with the read-time rule; no model, nothing re-run."""
    for record in (
        refusal("standard", "two_hop:01", "two_hop", generation_error="llm_timeout"),
        rec("standard", "single_hop:01"),
    ):
        append_line(system_file(tmp_path, "standard"), record)
    for record in (
        cut("two_hop:04"),
        cut("two_hop:08", steps_ms=dict(WITH_ANSWER)),  # a real refusal: not listed
        rec("agentic", "colloquial:05", "colloquial", status="error", found=False, answered=False),
    ):
        append_line(system_file(tmp_path, "agentic"), record)
    assert cli.main(["classify", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "standard: 2 records, retryable 1" in text and "generation_error: two_hop:01" in text
    assert "agentic: 3 records, retryable 2" in text
    assert "deadline_truncated: two_hop:04" in text and "error: colloquial:05" in text
    assert "two_hop:08" not in text
