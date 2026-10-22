"""Judge against human scores on the 1-3 scale, split into "full" (3) and "not full" (1 or 2)
(spec EVAL-1, design s14-3).

- TPR ("keeps good"): of the answers the human scored 3, the share the judge also scored 3.
- TNR ("catches bad"): of the answers the human scored 1 or 2, the share the judge also scored
  1 or 2.
- kappa: Cohen's kappa of the full / not-full split (the same as quadratic weighted kappa on two
  levels); `weighted_kappa` is the quadratic weighted kappa on the 1-3 scores themselves.
- Gate (the eval upgrade plan, step 1): TPR >= 0.85, TNR >= 0.85, kappa >= 0.7. A rate computed on
  fewer than `MIN_PER_CLASS` answers is "not estimable": with 6 bad answers all caught, the lower
  end of the 95% Wilson interval is still only about 61%.

The numbers say how far the judge agrees with whoever scored. When the scores are an AI's
pre-score, `render_calibration(..., source="ai-prescore")` says so and does not call it a pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from rag.eval.stats import Proportion, weighted_kappa

TPR_MIN = 0.85
TNR_MIN = 0.85
KAPPA_MIN = 0.7
MIN_PER_CLASS = 10
"""Fewest answers of a class (human 3, or human 1/2) for its rate to count."""
_TOLERANCE = 1e-9

Status = Literal["pass", "fail", "not_estimable"]
Source = Literal["human", "ai-prescore"]


class CalibrationItem(BaseModel):
    """One answer scored by a person (or a pre-scorer) and by the judge, both on 1..3."""

    label: str
    human: int
    judge: int


class Disagreement(BaseModel):
    label: str
    human: int
    judge: int
    kind: Literal["judge_too_strict", "judge_too_lenient", "other"]
    """`judge_too_strict`: human 3, judge 1 or 2. `judge_too_lenient`: human 1 or 2, judge 3.
    `other`: both not full but different (1 against 2)."""


class BinaryAgreement(BaseModel):
    n: int
    human_full: int
    human_not_full: int
    tpr: Proportion
    tnr: Proportion
    kappa: float | None
    weighted_kappa: float | None
    exact: Proportion
    disagreements: list[Disagreement]


class GateCheck(BaseModel):
    tpr: Status
    tnr: Status
    kappa: Status
    verdict: Status


def _kind(human: int, judge: int) -> Literal["judge_too_strict", "judge_too_lenient", "other"]:
    if human == 3:
        return "judge_too_strict"
    if judge == 3:
        return "judge_too_lenient"
    return "other"


def calibrate(items: Sequence[CalibrationItem]) -> BinaryAgreement:
    """Agreement of the judge with the human scores. Every score must be 1, 2 or 3."""
    for item in items:
        if item.human not in (1, 2, 3) or item.judge not in (1, 2, 3):
            raise ValueError(
                f"{item.label}: scores must be 1, 2 or 3, got {item.human}, {item.judge}"
            )
    human_full = [i for i in items if i.human == 3]
    human_bad = [i for i in items if i.human != 3]
    tpr = Proportion.of(sum(i.judge == 3 for i in human_full), len(human_full))
    tnr = Proportion.of(sum(i.judge != 3 for i in human_bad), len(human_bad))
    humans = [int(i.human == 3) for i in items]
    judges = [int(i.judge == 3) for i in items]
    pairs = [i for i in items if i.human != i.judge]
    disagreements = [
        Disagreement(label=i.label, human=i.human, judge=i.judge, kind=_kind(i.human, i.judge))
        for i in pairs
    ]
    disagreements.sort(key=lambda d: d.kind == "other")  # stable: split disagreements first
    return BinaryAgreement(
        n=len(items),
        human_full=len(human_full),
        human_not_full=len(human_bad),
        tpr=tpr,
        tnr=tnr,
        kappa=weighted_kappa(judges, humans, low=0, high=1),
        weighted_kappa=weighted_kappa(
            [i.judge for i in items], [i.human for i in items], low=1, high=3
        ),
        exact=Proportion.of(len(items) - len(pairs), len(items)),
        disagreements=disagreements,
    )


def _rate_status(rate: Proportion, minimum: float) -> Status:
    if rate.n < MIN_PER_CLASS or rate.rate is None:
        return "not_estimable"
    return "pass" if rate.rate >= minimum - _TOLERANCE else "fail"


def check_gate(
    report: BinaryAgreement,
    *,
    tpr_min: float = TPR_MIN,
    tnr_min: float = TNR_MIN,
    kappa_min: float = KAPPA_MIN,
) -> GateCheck:
    """Each of the three against its line (point estimate, not below); the verdict is `fail` when
    any one fails, else `not_estimable` when any one cannot be computed, else `pass`."""
    tpr = _rate_status(report.tpr, tpr_min)
    tnr = _rate_status(report.tnr, tnr_min)
    if report.kappa is None or min(report.human_full, report.human_not_full) < MIN_PER_CLASS:
        kappa: Status = "not_estimable"
    else:
        kappa = "pass" if report.kappa >= kappa_min - _TOLERANCE else "fail"
    parts = (tpr, tnr, kappa)
    verdict: Status = (
        "fail" if "fail" in parts else "not_estimable" if "not_estimable" in parts else "pass"
    )
    return GateCheck(tpr=tpr, tnr=tnr, kappa=kappa, verdict=verdict)


_STATUS_TEXT = {
    "pass": "meets the line",
    "fail": "below the line",
    "not_estimable": "not estimable",
}
_VERDICT_TEXT = {
    "pass": "all three meet the gate",
    "fail": "at least one is below the gate",
    "not_estimable": "cannot be decided: too few answers in a class",
}


def render_calibration(
    report: BinaryAgreement, gate: GateCheck, *, source: Source = "human", title: str = ""
) -> str:
    """Markdown of one calibration result."""
    lines = [f"# {title or 'Judge calibration'}", ""]
    if source == "ai-prescore":
        lines += [
            "> The scores compared with the judge here are an AI pre-score, not an independent "
            "human judgement. The owner still has to confirm the rows; nothing below is a "
            "passed calibration until then.",
            "",
        ]
    kappa = "n/a" if report.kappa is None else f"{report.kappa:.3f}"
    wkappa = "n/a" if report.weighted_kappa is None else f"{report.weighted_kappa:.3f}"
    lines += [
        f"Answers compared: {report.n} (scored 3: {report.human_full}; "
        f"scored 1 or 2: {report.human_not_full}).",
        "",
        "| Measure | Value (95% Wilson interval) | Line | Result |",
        "|---|---|---|---|",
        f"| TPR (human 3, judge also 3) | {report.tpr.text()} | >= {TPR_MIN} "
        f"| {_STATUS_TEXT[gate.tpr]} |",
        f"| TNR (human 1 or 2, judge also 1 or 2) | {report.tnr.text()} | >= {TNR_MIN} "
        f"| {_STATUS_TEXT[gate.tnr]} |",
        f"| kappa (3 or not) | {kappa} | >= {KAPPA_MIN} | {_STATUS_TEXT[gate.kappa]} |",
        f"| weighted kappa (1 to 3) | {wkappa} | | |",
        f"| identical scores | {report.exact.text()} | | |",
        "",
        f"Verdict against the gate: {_VERDICT_TEXT[gate.verdict]}.",
        "",
    ]
    if report.disagreements:
        lines += ["Disagreements (label, human score, judge score):", ""]
        for d in report.disagreements:
            lines.append(f"- {d.label}: human {d.human}, judge {d.judge} ({d.kind})")
    else:
        lines.append("No disagreements.")
    return "\n".join(lines) + "\n"
