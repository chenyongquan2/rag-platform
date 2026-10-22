"""Spec EVAL-1 AC1, AC2: TPR, TNR, kappa, disagreements and the gate."""

from __future__ import annotations

import pytest

from rag.eval.calibration import CalibrationItem, calibrate, check_gate, render_calibration
from rag.eval.stats import wilson_interval


def _items(pairs: list[tuple[int, int]]) -> list[CalibrationItem]:
    return [CalibrationItem(label=f"r{i}", human=h, judge=j) for i, (h, j) in enumerate(pairs)]


def test_calibrate_rates_kappa_and_disagreements() -> None:
    # (human, judge): 8 human 3 (7 judge 3, 1 judge 2), 4 human 1/2 (3 judge 1/2, 1 judge 3)
    pairs = [(3, 3)] * 7 + [(3, 2)] + [(1, 1), (2, 2), (2, 1), (2, 3)]
    report = calibrate(_items(pairs))
    assert report.n == 12 and report.human_full == 8 and report.human_not_full == 4
    assert (report.tpr.count, report.tpr.n) == (7, 8)
    assert (report.tnr.count, report.tnr.n) == (3, 4)
    assert report.tpr.rate == pytest.approx(7 / 8)
    assert (report.tpr.low, report.tpr.high) == pytest.approx(wilson_interval(7, 8))
    assert (report.tnr.low, report.tnr.high) == pytest.approx(wilson_interval(3, 4))
    # Binary table (full vs not full): judge-full & human-full 7, judge-not & human-full 1,
    # judge-full & human-not 1, judge-not & human-not 3 -> po = 10/12, pe = (8*8 + 4*4)/144.
    po, pe = 10 / 12, (8 * 8 + 4 * 4) / 144
    assert report.kappa == pytest.approx((po - pe) / (1 - pe))
    assert report.exact.count == 9  # (3,3)x7, (1,1), (2,2)
    assert report.weighted_kappa is not None
    # Disagreements on the full/not-full split come first and carry both scores.
    assert [(d.label, d.human, d.judge) for d in report.disagreements] == [
        ("r7", 3, 2),
        ("r11", 2, 3),
        ("r10", 2, 1),
    ]
    assert [d.kind for d in report.disagreements] == [
        "judge_too_strict",
        "judge_too_lenient",
        "other",
    ]


def test_calibrate_without_non_full_human_scores_has_no_tnr() -> None:
    report = calibrate(_items([(3, 3)] * 9 + [(3, 2)]))
    assert report.human_not_full == 0
    assert report.tnr.n == 0 and report.tnr.rate is None
    assert report.tpr.rate == pytest.approx(0.9)
    assert report.kappa is None or report.kappa == pytest.approx(0.0, abs=1e-9)  # all humans 3


def test_calibrate_rejects_scores_outside_one_to_three() -> None:
    with pytest.raises(ValueError):
        calibrate(_items([(4, 3)]))
    with pytest.raises(ValueError):
        calibrate(_items([(3, 0)]))


def test_gate_pass_fail_not_estimable() -> None:
    good = calibrate(_items([(3, 3)] * 20 + [(1, 1)] * 6 + [(2, 2)] * 6 + [(3, 2)]))
    gate = check_gate(good)
    assert gate.verdict == "pass"
    assert gate.tpr == "pass" and gate.tnr == "pass" and gate.kappa == "pass"
    # TNR below 0.85: 9 of 12 caught.
    weak = calibrate(_items([(3, 3)] * 20 + [(1, 1)] * 6 + [(2, 3)] * 3 + [(2, 2)] * 3))
    gate = check_gate(weak)
    assert gate.tnr == "fail" and gate.verdict == "fail"
    # Exactly on the line passes (point estimate "not below").
    edge = calibrate(_items([(3, 3)] * 17 + [(3, 2)] * 3 + [(1, 1)] * 17 + [(1, 3)] * 3))
    assert edge.tpr.rate == pytest.approx(0.85)
    assert check_gate(edge).tpr == "pass" and check_gate(edge).tnr == "pass"
    # Too few non-full human answers: TNR is not estimable, so is the verdict.
    sparse = calibrate(_items([(3, 3)] * 30 + [(2, 2)]))
    gate = check_gate(sparse)
    assert gate.tnr == "not_estimable" and gate.verdict == "not_estimable"
    # A failure elsewhere still shows as fail.
    bad = calibrate(_items([(3, 3)] * 5 + [(3, 1)] * 5 + [(2, 2)]))
    assert check_gate(bad).verdict == "fail"


def test_render_labels_the_source_and_does_not_claim_a_pass_for_an_ai_pre_score() -> None:
    report = calibrate(_items([(3, 3)] * 20 + [(1, 1)] * 6 + [(2, 2)] * 6))
    gate = check_gate(report)
    ai = render_calibration(report, gate, source="ai-prescore")
    assert "AI pre-score" in ai and "not an independent human" in ai
    assert "owner" in ai  # the owner still has to confirm
    human = render_calibration(report, gate, source="human")
    assert "AI pre-score" not in human
    for text in (ai, human):
        assert "TPR" in text and "TNR" in text and "kappa" in text
        assert "0.85" in text and "0.7" in text
