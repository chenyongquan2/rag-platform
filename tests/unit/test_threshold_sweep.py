"""Spec P2-02 AC1-AC6: the sweep maths of the rerank-threshold script (design s10-2, s14-4)."""

from __future__ import annotations

import pytest

from rag.eval.stats import wilson_interval
from rag.eval.threshold.records import Candidate, ScoreRecord
from rag.eval.threshold.sweep import (
    group_rates,
    is_refused,
    paired_vs_baseline,
    parse_thresholds,
    recommend,
    sweep,
    top_score,
)


def rec(
    qid: str,
    scores: list[float],
    *,
    kind: str = "answerable",
    correct: dict[int, str] | None = None,
    pages: list[str] | None = None,
) -> ScoreRecord:
    """A record with candidates scored `scores` (best first); candidate i is on page
    `hf_docs:p<i>` unless `correct` renames it; `pages` are the labelled correct pages."""
    names = {i: f"hf_docs:p{i}" for i in range(len(scores))} | (correct or {})
    return ScoreRecord(
        qid=qid,
        question=f"question {qid}",
        kind=kind,  # type: ignore[arg-type]
        relevant_pages=pages or [],
        candidates=[
            Candidate(chunk_id=f"{names[i]}#0", doc_id=names[i], score=s)
            for i, s in enumerate(scores)
        ],
    )


def test_refusal_is_top_score_below_threshold() -> None:
    r = rec("a", [0.30, 0.10])
    assert top_score(r) == 0.30
    assert is_refused(r, 0.31)
    assert not is_refused(r, 0.30)  # a score equal to the threshold passes (P0-08)
    assert not is_refused(r, 0.0)
    nothing = rec("b", [])
    assert top_score(nothing) is None
    assert is_refused(nothing, 0.0)  # no candidates: nothing to answer from


def test_rates_have_wilson_intervals() -> None:
    answerable = [rec(f"a{i}", [0.1 * (i + 1)]) for i in range(10)]  # tops 0.1 .. 1.0
    rates = group_rates(answerable, 0.35)  # 0.1, 0.2, 0.3 refused
    assert (rates.refused.count, rates.refused.n) == (3, 10)
    assert rates.refused.rate == pytest.approx(0.3)
    assert (rates.refused.low, rates.refused.high) == wilson_interval(3, 10)
    assert (rates.accepted.count, rates.accepted.n) == (7, 10)
    assert (rates.accepted.low, rates.accepted.high) == wilson_interval(7, 10)
    assert rates.threshold == 0.35


def test_all_refused_none_refused_and_empty_group() -> None:
    records = [rec("a", [0.2]), rec("b", [0.4])]
    none = group_rates(records, 0.0)
    assert (none.refused.count, none.accepted.count) == (0, 2)
    assert none.refused.low == 0.0 and none.refused.high == pytest.approx(wilson_interval(0, 2)[1])
    everything = group_rates(records, 1.0)
    assert (everything.refused.count, everything.accepted.count) == (2, 0)
    assert everything.refused.high == 1.0
    empty = group_rates([], 0.5)
    assert empty.n == 0
    assert empty.refused.rate is None and empty.accepted.rate is None
    assert (empty.refused.low, empty.refused.high) == (0.0, 1.0)


def test_lost_page_needs_a_scored_correct_page() -> None:
    pages = ["hf_docs:right"]
    scored_low = rec("a", [0.9, 0.2], correct={1: "hf_docs:right"}, pages=pages)
    scored_high = rec("b", [0.9, 0.7], correct={1: "hf_docs:right"}, pages=pages)
    two_copies = rec(
        "c", [0.9, 0.5, 0.2], correct={1: "hf_docs:right", 2: "hf_docs:right"}, pages=pages
    )
    not_found = rec("d", [0.9, 0.8], pages=pages)  # the retriever never found the page
    rates = group_rates([scored_low, scored_high, two_copies, not_found], 0.6)
    # a: lost (only copy 0.2 < 0.6); b: kept; c: its best copy 0.5 < 0.6 -> lost; d: not counted.
    assert (rates.lost_page.count, rates.lost_page.n) == (2, 3)
    assert group_rates([scored_low], 0.2).lost_page.count == 0  # equal to the threshold passes


def test_kept_chunks_are_capped_at_top_k() -> None:
    records = [rec("a", [0.9] * 10), rec("b", [0.9, 0.1])]
    rates = group_rates(records, 0.5, top_k=8)
    assert rates.mean_kept == pytest.approx((8 + 1) / 2)


def _dev() -> list[ScoreRecord]:
    # Top scores 0.95 (x4), 0.80 (x3), 0.55, 0.35, 0.20 -> 10 questions.
    tops = [0.95] * 4 + [0.80] * 3 + [0.55, 0.35, 0.20]
    return [rec(f"d{i}", [t]) for i, t in enumerate(tops)]


GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def test_recommend_largest_threshold_within_cap() -> None:
    rows = sweep({"dev": _dev()}, GRID)
    # false refusals: 0.1:0, 0.3:1, 0.5:2, 0.7:3, 0.9:6, 0.99:10 of 10
    assert recommend(rows, group="dev", max_false_refusal=0.0) == 0.1
    assert recommend(rows, group="dev", max_false_refusal=0.2) == 0.5
    assert recommend(rows, group="dev", max_false_refusal=0.3) == 0.7  # at the cap counts
    assert recommend(rows, group="dev", max_false_refusal=1.0) == 0.99  # none too many: the top


def test_recommend_wilson_upper_mode() -> None:
    rows = sweep({"dev": _dev()}, GRID)
    # Wilson upper bounds with n = 10: 0/10 -> 0.278, 1/10 -> 0.445, 2/10 -> 0.557.
    assert recommend(rows, group="dev", max_false_refusal=0.45, bound="wilson_upper") == 0.3
    assert recommend(rows, group="dev", max_false_refusal=0.30, bound="wilson_upper") == 0.1
    assert recommend(rows, group="dev", max_false_refusal=0.45, bound="observed") == 0.7


def test_recommend_none_when_no_threshold_fits() -> None:
    rows = sweep({"dev": _dev()}, [0.3, 0.5])
    assert recommend(rows, group="dev", max_false_refusal=0.0) is None
    with pytest.raises(ValueError):
        recommend(rows, group="dev", bound="median")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        recommend(rows, group="missing")


def test_sweep_rows_follow_the_threshold_order_and_hold_every_group() -> None:
    rows = sweep({"dev": _dev(), "no_answer": [rec("n", [0.4], kind="no_answer")]}, [0.5, 0.1])
    assert [r.threshold for r in rows] == [0.1, 0.5]
    assert set(rows[0].groups) == {"dev", "no_answer"}
    # the no-answer question (top 0.4) is accepted at 0.1 and refused at 0.5
    assert rows[0].groups["no_answer"].accepted.count == 1
    assert rows[1].groups["no_answer"].refused.count == 1


def test_parse_thresholds() -> None:
    assert parse_thresholds("0.3, 0.5,0.1") == [0.1, 0.3, 0.5]
    assert parse_thresholds("0.1:0.5:0.1") == [0.1, 0.2, 0.3, 0.4, 0.5]  # no 0.30000000000000004
    assert parse_thresholds("0.3,0.3") == [0.3]
    assert parse_thresholds("0:1:0.25") == [0.0, 0.25, 0.5, 0.75, 1.0]
    for bad in ("", "abc", "0.5:0.1:0.1", "0.1:0.5:0", "-0.1,0.5", "0.5,1.5", "0.1:0.5"):
        with pytest.raises(ValueError):
            parse_thresholds(bad)


def test_paired_change_against_a_baseline_threshold() -> None:
    records = [rec(f"q{i}", [t]) for i, t in enumerate([0.2, 0.4, 0.6, 0.8])]
    up = paired_vs_baseline(records, 0.7, baseline=0.3)  # 0.4 and 0.6 are newly refused
    assert (up.newly_refused, up.newly_accepted, up.n) == (2, 0, 4)
    assert up.p_value == pytest.approx(0.5)  # exact McNemar of 2 against 0
    down = paired_vs_baseline(records, 0.1, baseline=0.5)  # 0.2 and 0.4 are newly accepted
    assert (down.newly_refused, down.newly_accepted) == (0, 2)
    same = paired_vs_baseline(records, 0.3, baseline=0.3)
    assert (same.newly_refused, same.newly_accepted, same.p_value) == (0, 0, 1.0)
