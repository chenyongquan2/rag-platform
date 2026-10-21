"""Spec P5-01 AC9: the shadow report: would-hit rate, wrong-hit rate and their Wilson intervals
(design s12-4, s14-4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.cache.cli import main
from rag.cache.report import (
    load_gold,
    load_records,
    render_report,
    shadow_report,
    wrong_reasons,
)
from rag.cache.semantic import question_hash
from rag.eval.stats import wilson_interval
from rag.testing import observe_settings


def rec(status: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kb": "hf_docs",
        "question": "q",
        "question_hash": "h",
        "status": status,
        "fresh_top": ["a#0", "b#0"],
        "fresh_top_docs": ["a", "b"],
        "lookup_ms": 1.5,
    }
    if status == "would_hit":
        base.update(
            candidate_hash="c",
            distance=0.05,
            guard_ok=True,
            cached_top=["a#0", "b#0"],
            cached_top_docs=["a", "b"],
            same_top1=True,
            same_top_docs=True,
            overlap=1.0,
        )
    base.update(over)
    return base


def test_rates_and_wilson_intervals() -> None:
    records = (
        [rec("empty")] * 20
        + [rec("too_far", nearest_distance=0.4)] * 60
        + [rec("guard_rejected", guard_reason="identifiers")] * 5
        + [rec("would_hit")] * 14
        + [rec("would_hit", same_top1=False, cached_top=["b#0", "a#0"])] * 1
        + [rec("no_vector")] * 7
    )
    report = shadow_report(records)
    assert report["records"] == 107 and report["lookups"] == 100  # no_vector is not a lookup
    assert report["status"]["would_hit"] == 15 and report["status"]["no_vector"] == 7
    hit = report["would_hit"]
    assert hit["n"] == 15 and hit["of"] == 100 and hit["rate"] == pytest.approx(0.15)
    assert hit["wilson95"] == pytest.approx(wilson_interval(15, 100))
    wrong = report["wrong_hit"]
    assert wrong["n"] == 1 and wrong["of"] == 15 and wrong["rate"] == pytest.approx(1 / 15)
    assert wrong["wilson95"] == pytest.approx(wilson_interval(1, 15))
    assert report["guard"]["rejected"] == 5 and report["guard"]["by_reason"] == {"identifiers": 5}


def test_a_wrong_hit_is_a_different_top_chunk_or_a_lost_gold_page() -> None:
    same = rec("would_hit")
    other_top = rec("would_hit", same_top1=False, cached_top=["b#0", "a#0"])
    lost = rec(
        "would_hit",
        question_hash="q1",
        cached_top=["a#0", "c#0"],
        cached_top_docs=["a", "c"],
        fresh_top_docs=["a", "b"],
    )
    gold = {"q1": {"b"}}
    assert wrong_reasons(same, gold) == []
    assert wrong_reasons(other_top, gold) == ["top1_differs"]
    assert wrong_reasons(lost, gold) == ["gold_lost"]
    # the same record without a label: only the top-1 rule can apply
    assert wrong_reasons(lost, {}) == []
    # a gold page that is in neither list is not a loss
    assert wrong_reasons(lost, {"q1": {"z"}}) == []
    both = rec(
        "would_hit",
        question_hash="q1",
        same_top1=False,
        cached_top_docs=["c"],
        fresh_top_docs=["a", "b"],
    )
    assert wrong_reasons(both, gold) == ["top1_differs", "gold_lost"]


def test_report_counts_each_wrong_hit_once_and_names_the_reasons() -> None:
    gold = {"q1": {"b"}}
    records = [
        rec("would_hit", question_hash="q1", cached_top_docs=["a"], same_top1=True),
        rec("would_hit", question_hash="q1", same_top1=False, cached_top_docs=["c"]),
        rec("would_hit", question_hash="q2"),
    ]
    report = shadow_report(records, gold)
    assert report["wrong_hit"]["n"] == 2 and report["wrong_hit"]["of"] == 3
    assert report["wrong_hit"]["by_reason"] == {"gold_lost": 2, "top1_differs": 1}
    assert report["wrong_hit"]["labelled"] == 2


def test_the_rule_for_turning_l2_on_uses_the_upper_end_of_the_interval() -> None:
    clean = shadow_report([rec("would_hit")] * 400 + [rec("empty")] * 600)
    assert clean["decision"]["wrong_hit_upper"] < 0.01
    assert clean["decision"]["turn_on"] is True
    few = shadow_report([rec("would_hit")] * 30 + [rec("empty")] * 70)
    assert few["decision"]["turn_on"] is False  # 0 wrong in 30: the upper end is still above 1%
    low = shadow_report([rec("empty")] * 100 + [rec("too_far")] * 100)
    assert low["decision"]["turn_on"] is False and "no would-hit" in low["decision"]["why"]


def test_an_empty_file_gives_a_report_not_an_error() -> None:
    report = shadow_report([])
    assert report["records"] == 0 and report["would_hit"]["rate"] is None


def test_lookup_overhead_and_distances() -> None:
    records = [rec("would_hit", distance=d, lookup_ms=ms) for d, ms in ((0.01, 1.0), (0.09, 3.0))]
    records += [rec("too_far", nearest_distance=0.3, lookup_ms=2.0)]
    report = shadow_report(records)
    assert report["lookup_ms"]["median"] == pytest.approx(2.0)
    assert report["distance"]["would_hit_max"] == pytest.approx(0.09)
    assert report["distance"]["nearest_miss_min"] == pytest.approx(0.3)


def test_render_report_is_plain_text_with_the_numbers() -> None:
    text = render_report(shadow_report([rec("would_hit")] * 3 + [rec("empty")] * 7))
    assert "would hit" in text and "3 of 10" in text and "Wilson" in text


def test_load_records_and_gold(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    path.write_text(
        "\n".join([json.dumps(rec("empty")), "", "not json", json.dumps(rec("would_hit"))]) + "\n",
        encoding="utf-8",
    )
    assert [r["status"] for r in load_records([path])] == ["empty", "would_hit"]
    qs = tmp_path / "qs.jsonl"
    qs.write_text(
        json.dumps({"qid": "x:1", "question": "How do I push?", "relevant_pages": ["hf_docs:a.md"]})
        + "\n",
        encoding="utf-8",
    )
    assert load_gold([qs]) == {question_hash("How do I push?"): {"hf_docs:a.md"}}


def test_cli_shadow_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "shadow.jsonl"
    path.write_text(
        "\n".join(json.dumps(rec(s)) for s in ["empty", "would_hit", "too_far"]) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    code = main(
        ["shadow-report", "--records", str(path), "--json", str(out)], settings=observe_settings()
    )
    assert code == 0
    assert "would hit" in capsys.readouterr().out
    assert json.loads(out.read_text("utf-8"))["lookups"] == 3


def test_a_smaller_threshold_reads_the_far_would_hits_as_too_far() -> None:
    records = [rec("would_hit", distance=0.04)] * 3 + [rec("would_hit", distance=0.18)] * 2
    records += [rec("empty")] * 5
    wide = shadow_report(records)
    narrow = shadow_report(records, max_distance=0.1)
    assert wide["would_hit"]["n"] == 5 and narrow["would_hit"]["n"] == 3
    assert narrow["status"]["too_far"] == 2 and narrow["max_distance"] == 0.1
    assert wide["max_distance"] is None
