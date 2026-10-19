"""The maths of the rerank-threshold sweep (spec P2-02; design s10-2, s14-4). No I/O.

A question is *refused* at threshold `t` when its best rerank score is below `t` (a score equal to
`t` passes, as in `RerankThreshold`, P0-08) or it has no candidate. So one number per question, its
top score, decides the outcome at every threshold:

- on answerable questions the refused share is the false-refusal rate;
- on no-answer questions the not-refused share is the wrong-answer-accepted rate.

Rates carry Wilson intervals of the exact counts (s14-4); with a few dozen questions they are wide.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rag.eval.stats import mcnemar_exact, wilson_interval
from rag.eval.threshold.records import ScoreRecord

Bound = Literal["observed", "wilson_upper"]
DEFAULT_TOP_K = 8
"""Final list length (`retrieval.final_k`): the mean of kept chunks is counted up to this."""


@dataclass(frozen=True)
class Rate:
    count: int
    n: int
    low: float
    high: float
    """95% Wilson interval; (0, 1) when `n` is 0."""

    @property
    def rate(self) -> float | None:
        return self.count / self.n if self.n else None


def make_rate(count: int, n: int) -> Rate:
    low, high = wilson_interval(count, n)
    return Rate(count=count, n=n, low=low, high=high)


@dataclass(frozen=True)
class GroupRates:
    """What one threshold does to one group of questions."""

    threshold: float
    n: int
    refused: Rate
    """Questions with every candidate below the threshold (false refusal for answerable ones)."""
    accepted: Rate
    """Questions not refused (wrong answer accepted for no-answer ones)."""
    lost_page: Rate
    """Of the questions whose correct page was scored, those where every scored chunk of a correct
    page is below the threshold (so the page no longer reaches the answer step)."""
    mean_kept: float
    """Mean number of candidates at or above the threshold, counted up to `top_k`."""


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    groups: dict[str, GroupRates]


def top_score(record: ScoreRecord) -> float | None:
    return max((c.score for c in record.candidates), default=None)


def is_refused(record: ScoreRecord, threshold: float) -> bool:
    top = top_score(record)
    return top is None or top < threshold


def _lost_page(record: ScoreRecord, threshold: float) -> bool | None:
    """True or False for a question whose correct page was scored; None when it was not (the
    retriever never found the page, so the threshold is not the cause)."""
    pages = set(record.relevant_pages)
    scores = [c.score for c in record.candidates if c.doc_id in pages]
    if not scores:
        return None
    return max(scores) < threshold


def group_rates(
    records: Sequence[ScoreRecord], threshold: float, *, top_k: int = DEFAULT_TOP_K
) -> GroupRates:
    refused = sum(is_refused(r, threshold) for r in records)
    lost = [x for r in records if (x := _lost_page(r, threshold)) is not None]
    kept = [min(sum(c.score >= threshold for c in r.candidates), top_k) for r in records]
    return GroupRates(
        threshold=threshold,
        n=len(records),
        refused=make_rate(refused, len(records)),
        accepted=make_rate(len(records) - refused, len(records)),
        lost_page=make_rate(sum(lost), len(lost)),
        mean_kept=sum(kept) / len(kept) if kept else 0.0,
    )


def sweep(
    groups: Mapping[str, Sequence[ScoreRecord]],
    thresholds: Sequence[float],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[SweepRow]:
    """One row per threshold (ascending), each with the rates of every named group."""
    return [
        SweepRow(
            threshold=t,
            groups={name: group_rates(records, t, top_k=top_k) for name, records in groups.items()},
        )
        for t in sorted(set(thresholds))
    ]


def recommend(
    rows: Sequence[SweepRow],
    *,
    group: str,
    max_false_refusal: float = 0.05,
    bound: Bound = "observed",
) -> float | None:
    """The largest threshold whose false-refusal rate on `group` is within `max_false_refusal`.

    `bound` "observed" compares the observed rate; "wilson_upper" the upper end of its Wilson
    interval (stricter: with 99 questions only zero refusals keep it under 5%). The rate never
    goes down as the threshold goes up, so this is the last threshold before the cap is crossed.
    Returns None when even the lowest threshold is over the cap.
    """
    if bound not in ("observed", "wilson_upper"):
        raise ValueError(f"bound must be 'observed' or 'wilson_upper', got {bound!r}")
    fits: list[float] = []
    for row in rows:
        refused = row.groups[group].refused
        value = refused.rate if bound == "observed" else refused.high
        if value is None:
            raise ValueError(f"group {group!r} has no questions")
        if value <= max_false_refusal + 1e-12:
            fits.append(row.threshold)
    return max(fits) if fits else None


@dataclass(frozen=True)
class PairedChange:
    """The same questions at two thresholds: which flipped between refused and accepted."""

    threshold: float
    baseline: float
    n: int
    newly_refused: int
    """Accepted at the baseline, refused at `threshold`."""
    newly_accepted: int
    """Refused at the baseline, accepted at `threshold`."""
    p_value: float
    """Exact McNemar test on the two flip counts (design s14-4)."""


def paired_vs_baseline(
    records: Sequence[ScoreRecord], threshold: float, *, baseline: float
) -> PairedChange:
    before = [is_refused(r, baseline) for r in records]
    after = [is_refused(r, threshold) for r in records]
    newly_refused = sum(a and not b for a, b in zip(after, before, strict=True))
    newly_accepted = sum(b and not a for a, b in zip(after, before, strict=True))
    return PairedChange(
        threshold=threshold,
        baseline=baseline,
        n=len(records),
        newly_refused=newly_refused,
        newly_accepted=newly_accepted,
        p_value=mcnemar_exact(newly_refused, newly_accepted),
    )


def parse_thresholds(text: str) -> list[float]:
    """`0.3,0.5` (a list) or `0.05:0.95:0.05` (start:stop:step, stop included); values in [0, 1].

    Range values are computed as `start + i * step` and rounded to 4 decimals, so there is no
    0.30000000000000004. Returns the sorted distinct values; raises ValueError on bad input.
    """
    text = text.strip()
    if not text:
        raise ValueError("no thresholds given")
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 3:
                raise ValueError("a range is start:stop:step")
            start, stop, step = (float(p) for p in parts)
            if step <= 0 or stop < start:
                raise ValueError("a range needs step > 0 and stop >= start")
            count = math.floor((stop - start) / step + 1e-9) + 1
            values = [round(start + i * step, 4) for i in range(count)]
        else:
            values = [round(float(p), 4) for p in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"bad thresholds {text!r}: {exc}") from exc
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"thresholds must be between 0 and 1, got {text!r}")
    return sorted(set(values))


def format_threshold(value: float) -> str:
    return f"{value:.2f}" if round(value, 2) == value else f"{value:.4f}".rstrip("0")


def _cell(rate: Rate) -> str:
    return f"{rate.count}/{rate.n} ({rate.low:.1%} to {rate.high:.1%})"


def render_sweep(
    rows: Sequence[SweepRow],
    *,
    dev: str,
    checks: Sequence[str] = (),
    no_answers: Sequence[str] = (),
    recommended: float | None,
    max_false_refusal: float,
    bound: Bound,
) -> str:
    """Markdown table (one row per threshold) followed by the rule and the recommendation."""
    columns: list[tuple[str, str, str]] = [
        (f"{dev}: false refusal", dev, "refused"),
        (f"{dev}: correct page lost", dev, "lost_page"),
        (f"{dev}: mean chunks kept", dev, "kept"),
        *((f"{name}: false refusal", name, "refused") for name in checks),
        *((f"{name}: wrong answer accepted", name, "accepted") for name in no_answers),
    ]
    lines = [
        "| threshold | " + " | ".join(label for label, _, _ in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for row in rows:
        cells = [format_threshold(row.threshold)]
        for _, name, field in columns:
            g = row.groups[name]
            cells.append(f"{g.mean_kept:.1f}" if field == "kept" else _cell(getattr(g, field)))
        lines.append("| " + " | ".join(cells) + " |")
    what = (
        "observed false-refusal rate" if bound == "observed" else "Wilson upper bound of the rate"
    )
    lines += [
        "",
        f"Rule: the largest threshold whose {what} on {dev!r} is at most {max_false_refusal:.0%}. "
        "Only the selection set chooses; the other sets are reported, never used to choose.",
    ]
    if recommended is None:
        lines.append(f"No threshold on the grid meets the rule on {dev!r}.")
    else:
        lines.append(f"Recommended threshold: {format_threshold(recommended)}")
    return "\n".join(lines) + "\n"


def render_paired(
    thresholds: Sequence[float],
    groups: Mapping[str, Sequence[ScoreRecord]],
    *,
    baseline: float,
) -> str:
    """Markdown table: per threshold and group, `+a / -b (p)` with a = questions newly refused
    and b = questions newly accepted compared with the baseline threshold."""
    names = list(groups)
    lines = [
        f"Paired change against threshold {format_threshold(baseline)} "
        "(+ = refused at this threshold but not at the baseline, - = the reverse; "
        "p = exact McNemar):",
        "",
        "| threshold | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for t in sorted(set(thresholds)):
        cells = []
        for name in names:
            c = paired_vs_baseline(groups[name], t, baseline=baseline)
            cells.append(f"+{c.newly_refused} / -{c.newly_accepted} (p {c.p_value:.3f})")
        lines.append(f"| {format_threshold(t)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
