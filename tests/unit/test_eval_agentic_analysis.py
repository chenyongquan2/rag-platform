"""Spec P3-05 AC8-AC12: the paired statistics and tables of the comparison (pure functions)."""

from __future__ import annotations

from typing import Any

import pytest

from rag.eval.agentic.analysis import (
    cost_latency,
    judge_stability,
    limits_report,
    loop_report,
    min_detectable_table,
    paired_binary,
    paired_scores,
    per_type_summary,
    rewrite_effect,
)
from rag.eval.agentic.records import HookEvent, JudgeLine, RoundInfo, SystemRecord
from rag.eval.answer.record import Usage


def rec(system: str, qid: str, qtype: str = "single_hop", **kw: Any) -> SystemRecord:
    answerable = qtype != "no_answer"
    fields: dict[str, Any] = {
        "system": system,
        "qid": qid,
        "qtype": qtype,
        "question": f"Question {qid}?",
        "reference": "ref" if answerable else "",
        "relevant_pages": [f"g{qid}"] if answerable else [],
        "answerable": answerable,
        "status": "answered",
        "found": True,
        "answered": True,
        "text": "an answer",
    }
    fields.update(kw)
    return SystemRecord(**fields)


def line(system: str, qid: str, score: int, run: int = 1, **kw: Any) -> JudgeLine:
    return JudgeLine(system=system, qid=qid, run=run, score=score, **kw)  # type: ignore[arg-type]


def scores(system: str, values: dict[str, int], run: int = 1) -> list[JudgeLine]:
    return [line(system, q, s, run) for q, s in values.items()]


# ---- paired tests -------------------------------------------------------------------------------


def test_paired_binary_counts_and_p() -> None:
    a = {f"q{i}": i < 5 for i in range(20)}  # q0-q4 right
    b = {f"q{i}": 3 <= i < 10 for i in range(20)}  # q3-q9 right: +5 (q5-q9), -3 (q0-q2)
    result = paired_binary("correct", a, b)
    assert (result.n, result.improved, result.worsened, result.both, result.neither) == (
        20,
        5,
        3,
        2,
        10,
    )
    assert result.p_value == pytest.approx(0.7265625)  # exact McNemar on 5 against 3
    assert (result.a.count, result.b.count) == (5, 7)
    assert result.difference == pytest.approx(0.10)
    assert result.a.low < 0.25 < result.a.high  # Wilson interval of 5/20
    assert result.verdict == "no reliable difference (direction: B higher)"
    # The smallest net change 20 questions could show at all is far above the observed 10 points.
    assert result.min_detectable is not None and result.min_detectable > 0.25
    # Only questions both systems have are paired.
    assert paired_binary("correct", {"q1": True, "q2": False}, {"q1": False}).n == 1


def test_paired_binary_verdict_when_significant() -> None:
    a = {f"q{i}": False for i in range(30)}
    b = {f"q{i}": i < 8 for i in range(30)}
    result = paired_binary("correct", a, b)
    assert (result.improved, result.worsened, result.p_value < 0.05) == (8, 0, True)
    assert result.verdict == "B higher (p <= 0.05)"
    worse = paired_binary("correct", b, a)
    assert worse.verdict == "B lower (p <= 0.05)"
    same = paired_binary("correct", a, a)
    assert (same.p_value, same.verdict) == (1.0, "no difference")


def test_paired_scores_permutation() -> None:
    a = {"q1": 3, "q2": 3, "q3": 2, "q4": 1}
    b = {"q1": 3, "q2": 2, "q3": 3, "q4": 3}
    result = paired_scores(a, b)
    assert (result.n, result.improved, result.worsened, result.unchanged) == (4, 2, 1, 1)
    assert result.a_pct == pytest.approx(62.5) and result.b_pct == pytest.approx(87.5)
    assert result.diff_pct == pytest.approx(25.0)
    assert result.exact and 0 < result.p_value <= 1  # 3 changed questions: all 8 sign patterns


def test_per_type_summary() -> None:
    records = [
        *(rec("standard", q) for q in ("q1", "q2", "q3", "q4", "q5")),
        *(rec("agentic", q) for q in ("q1", "q2", "q3", "q4", "q5")),
        *(
            rec("standard", q, "no_answer", answered=n == 4, found=n == 4, status="x")
            for n, q in enumerate(("n1", "n2", "n3", "n4"), 1)
        ),
        *(
            rec(
                "agentic",
                q,
                "no_answer",
                answered=n == 4,
                found=n == 4,
                clarification=n == 3,
                status="need_clarification" if n == 3 else "x",
            )
            for n, q in enumerate(("n1", "n2", "n3", "n4"), 1)
        ),
    ]
    lines = [
        *scores("standard", {"q1": 3, "q2": 3, "q3": 2, "q4": 1, "q5": 3}),
        *scores("agentic", {"q1": 3, "q2": 2, "q3": 3, "q4": 3}),  # q5 never scored for B
    ]
    summary = per_type_summary(records, lines)
    single = summary["single_hop"]
    assert (single.n, single.unscored) == (4, 1)  # q5 is dropped from both sides
    assert single.correct is not None and single.adequate is not None and single.score is not None
    assert (single.correct.improved, single.correct.worsened) == (2, 1)
    assert (single.adequate.improved, single.adequate.worsened) == (1, 0)
    assert single.score.diff_pct == pytest.approx(25.0)
    assert single.refusal is None
    pooled = summary["answerable"]
    assert pooled.n == 4  # the three answerable types together
    # No-answer: refused = no answer given; a clarification is a refusal and counted apart.
    no_answer = summary["no_answer"]
    assert no_answer.correct is None and no_answer.refusal is not None
    assert (no_answer.refusal.a.count, no_answer.refusal.b.count, no_answer.refusal.n) == (3, 3, 4)
    assert no_answer.clarifications == {"standard": 0, "agentic": 1}


def test_refusal_rate_and_clarification_split() -> None:
    records = [
        *(rec("standard", f"n{i}", "no_answer", answered=False, found=False) for i in range(10)),
        *(
            rec("agentic", f"n{i}", "no_answer", answered=False, found=False, clarification=i < 2)
            for i in range(9)
        ),
        rec("agentic", "n9", "no_answer"),  # the graph answered a question with no answer
    ]
    summary = per_type_summary(records, [])["no_answer"]
    assert summary.refusal is not None
    assert (summary.refusal.a.count, summary.refusal.b.count) == (10, 9)
    assert summary.refusal.worsened == 1 and summary.refusal.improved == 0
    assert summary.clarifications["agentic"] == 2
    assert summary.refusal.b.low < 0.9 < summary.refusal.b.high


# ---- the loop inside the graph ------------------------------------------------------------------


def _events(*verdicts: str) -> list[HookEvent]:
    return [
        HookEvent(step="judge", round=i, data={"verdict": v}) for i, v in enumerate(verdicts, 1)
    ]


def _round(n: int, *pages: str) -> RoundInfo:
    return RoundInfo(round=n, sub_question="s", pages=list(pages), n_chunks=len(pages))


def _world() -> list[SystemRecord]:
    """Five questions answered by the graph (gold page of qN is `gqN`) and by standard RAG."""
    agentic = [
        rec("agentic", "q1", rounds=1, events=_events("enough"), round_info=[_round(1, "gq1")],
            retrieved_pages=["gq1"], context_pages=["gq1"]),
        rec("agentic", "q2", rounds=2, events=_events("partly_enough", "enough"),
            round_info=[_round(1, "x"), _round(2, "gq2")], retrieved_pages=["x", "gq2"],
            context_pages=["gq2", "x"]),
        rec("agentic", "q3", rounds=3, events=_events("not_enough", "not_enough", "not_enough"),
            round_info=[_round(1, "x"), _round(2, "y"), _round(3, "z")],
            retrieved_pages=["x", "y", "z"], context_pages=["x"]),
        rec("agentic", "q4", rounds=2, events=_events("partly_enough", "enough"),
            round_info=[_round(1, "x"), _round(2, "gq4")], retrieved_pages=["x", "gq4"],
            context_pages=["x"]),
        rec("agentic", "q5", status="need_clarification", found=False, answered=False,
            clarification=True, rounds=0),
    ]  # fmt: skip
    standard = [
        rec("standard", "q1", retrieved_pages=["gq1"], context_pages=["gq1"]),
        rec("standard", "q2", retrieved_pages=["gq2"], context_pages=["gq2"]),
        rec("standard", "q3", retrieved_pages=[], context_pages=[]),
        rec("standard", "q4", retrieved_pages=["x"], context_pages=["x"]),
        rec("standard", "q5", retrieved_pages=["gq5"], context_pages=["gq5"]),
    ]
    return [*standard, *agentic]


def test_loop_report_rescue_counts() -> None:
    report = loop_report(_world())
    single = report.by_type["single_hop"]
    assert single.n == 5
    assert single.rounds == {0: 1, 1: 1, 2: 2, 3: 1}
    assert (single.ran_round2, single.ran_round3) == (3, 1)
    assert single.first_verdicts == {"enough": 1, "partly_enough": 2, "not_enough": 1}
    assert single.final_sufficiency == {"enough": 3, "not_enough": 1}  # last verdict per question
    # Gold page in round one, in any round, in what the answer was written from.
    assert (single.hit_round1, single.hit_all_rounds, single.hit_context) == (1, 3, 2)
    assert single.round1_missed == 3  # q2, q3, q4 (q5 has no search)
    assert single.rescued == 2  # q2, q4: found in round 2 although not in round 1
    assert single.lost_to_context == 1  # q4: retrieved in round 2, not in the answer's fragments
    # The judge against the label "gold page in round one" (P3-03's label), inside the real graph.
    judged = report.judge_vs_label
    assert (judged.tp, judged.fp, judged.fn, judged.tn) == (1, 0, 0, 3)
    assert judged.accuracy.count == 4 and judged.accuracy.n == 4


def test_false_clarification_rate() -> None:
    report = loop_report(_world())
    rate = report.false_clarification["single_hop"]
    assert (rate.count, rate.n) == (1, 5)
    assert report.false_clarification["answerable"].count == 1
    assert report.clarified_questions == [("q5", "Question q5?")]


def test_rewrite_effect_paired() -> None:
    effect = rewrite_effect(_world())
    single = effect["single_hop"]
    # Raw question (standard): hit q1, q2; rewritten round one (graph): hit q1 only; q5 skipped
    # (the graph asked back and searched nothing).
    assert single.n == 4
    assert (single.a.count, single.b.count) == (2, 1)
    assert (single.improved, single.worsened) == (0, 1)


# ---- cost, latency, limits ----------------------------------------------------------------------


def _cost(calls: int, cheap: int, cny: float) -> dict[str, Any]:
    return {
        "usage": Usage(
            calls=calls + 1,
            prompt_tokens=1000 * calls,
            completion_tokens=100 * calls,
            cny=cny,
        ),
        "usage_by_kind": {"chat": Usage(calls=calls), "embedding": Usage(calls=1)},
        "usage_by_profile": {"cheap": Usage(calls=cheap), "online": Usage(calls=calls - cheap)},
    }  # fmt: skip


def test_cost_latency_tables() -> None:
    records = [
        rec("standard", "q1", total_ms=5000, **_cost(1, 0, 0.005)),
        rec("standard", "q2", total_ms=7000, **_cost(1, 0, 0.005)),
        rec("agentic", "q1", total_ms=10000, rounds=1, **_cost(3, 2, 0.01)),
        rec("agentic", "q2", total_ms=20000, rounds=2, **_cost(5, 4, 0.02)),
        rec("agentic", "q3", total_ms=40000, rounds=3, stop_reason="deadline", **_cost(7, 6, 0.03)),
    ]
    rows = {(r.system, r.qtype): r for r in cost_latency(records)}
    standard, agentic = rows["standard", "all"], rows["agentic", "all"]
    assert standard.n == 2 and standard.mean_chat_calls == 1 and standard.mean_cny == 0.005
    assert agentic.n == 3 and agentic.mean_chat_calls == 5 and agentic.mean_cheap_calls == 4
    assert agentic.mean_online_calls == 1 and agentic.mean_embedding_calls == 1
    assert agentic.mean_prompt_tokens == 5000 and agentic.total_cny == pytest.approx(0.06)
    assert agentic.mean_cny == pytest.approx(0.02) and agentic.mean_rounds == 2
    assert (agentic.p50_ms, agentic.p95_ms, agentic.max_ms) == (20000, 38000, 40000)
    assert (agentic.over_15s, agentic.over_30s) == (2, 1)
    assert ("agentic", "single_hop") in rows


def test_deadline_cases_listed() -> None:
    records = [
        rec("agentic", "q1", total_ms=9000, rounds=1),
        rec("agentic", "q2", total_ms=31000, rounds=2, stop_reason="deadline",
            coverage_unconfirmed=True, degraded=True, fallbacks=["judge:timeout"]),
        rec("agentic", "q3", total_ms=12000, rounds=3, stop_reason="max_rounds",
            coverage_unconfirmed=True),
        rec("agentic", "q4", total_ms=8000, rounds=1, rerank_timeout=True, degraded=True),
        rec("standard", "q1"),
    ]  # fmt: skip
    lines = scores("agentic", {"q1": 3, "q2": 1, "q3": 3, "q4": 3})
    report = limits_report(records, lines)
    assert report.stop_reasons == {"deadline": 1, "max_rounds": 1}
    assert report.coverage_unconfirmed == 2 and report.degraded == 2 and report.rerank_timeout == 1
    assert report.fallbacks == {"judge:timeout": 1}
    assert [c.qid for c in report.cases] == ["q2", "q3"]
    assert (report.cases[0].stop_reason, report.cases[0].total_ms, report.cases[0].score) == (
        "deadline",
        31000,
        1,
    )
    assert (report.limited_answerable, report.limited_correct) == (2, 1)
    assert report.over_deadline_ms == 1  # q2 took longer than the 30 s total deadline


def test_min_detectable_table() -> None:
    table = {row.name: row for row in min_detectable_table({"public": 65, "colloquial": 15})}
    assert table["public"].n == 65
    assert table["public"].best_case == pytest.approx(6 / 65)  # 6 better, 0 worse: p = 0.031
    assert table["public"].with_two_worse == pytest.approx(8 / 65)  # 10 better, 2 worse
    assert table["colloquial"].best_case == pytest.approx(6 / 15)


def test_judge_stability() -> None:
    lines = [
        *scores("agentic", {"q1": 3, "q2": 3, "q3": 2}, run=1),
        *scores("agentic", {"q1": 3, "q2": 2, "q3": 2}, run=2),
        line("agentic", "q4", 1, run=1, by_rule="no_answer"),
        line("agentic", "q4", 1, run=2, by_rule="no_answer"),
    ]
    result = judge_stability(lines, "agentic")
    assert result.n == 3 and result.n_rule_excluded == 1
    assert (result.score_flips.count, result.correct_flips.count) == (1, 1)
    assert result.mean_abs_diff == pytest.approx(1 / 3)


def test_clean_refusal_excludes_errors() -> None:
    """A run that crashed (a retrieval error, a gateway timeout) gave no answer, so it counts as
    a refusal by the fixed definition; the clean refusal rate leaves those out."""
    records = [
        *(rec("standard", f"n{i}", "no_answer", answered=False, found=False) for i in range(10)),
        *(
            rec("agentic", f"n{i}", "no_answer", answered=False, found=False,
                status="error" if i < 4 else "not_found",
                error="RetrievalError: x" if i < 4 else None)
            for i in range(10)
        ),
    ]  # fmt: skip
    records[0] = rec(
        "standard", "n0", "no_answer", answered=False, found=False, generation_error="llm_timeout"
    )
    summary = per_type_summary(records, [])["no_answer"]
    assert summary.refusal is not None and summary.clean_refusal is not None
    assert (summary.refusal.a.count, summary.refusal.b.count) == (10, 10)  # by the fixed definition
    assert (summary.clean_refusal.a.count, summary.clean_refusal.b.count) == (9, 6)
    assert (summary.clean_refusal.improved, summary.clean_refusal.worsened) == (0, 3)
    assert summary.errors == {"standard": 1, "agentic": 4}
