# ruff: noqa: E501 (long lines are Markdown table rows)
"""One system's new run against an older run of the same system on the same questions (spec P3-07).

    report = compare_runs(before_records, after_records, before_lines, after_lines, system="agentic")
    text = render_compare_runs(report)

Paired by question id over the questions both runs have: correct (judge score 3, pass 1) per
answerable type with exact McNemar; refusal and clean refusal on no-answer questions; cost and
latency of the shared questions. `b` / `improved` is the new run ("after"). No model, no network.

Per-question cost is shown in full and for chat models only, because records written before
P3-07 have no embedding ledger rows in their usage (P3-05 finding 6), later ones do.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from rag.eval.agentic.analysis import (
    ANSWERABLE,
    ANSWERABLE_TYPES,
    CORRECT,
    PairedBinary,
    clean_refusal,
    judge_scores,
    paired_binary,
)
from rag.eval.agentic.records import JudgeLine, SystemRecord, is_deadline_truncated, is_retryable
from rag.eval.agentic.sets import QTYPES
from rag.eval.retrieval.runner import latency_summary


class RunCostRow(BaseModel):
    label: str
    """`before` or `after`."""
    qtype: str
    n: int
    mean_cny: float
    mean_chat_cny: float
    p50_ms: float
    p95_ms: float
    mean_rounds: float
    errors: int
    truncated: int = 0
    """Answers cut off by the total deadline (spec EVAL-0); not counted in `errors`."""
    retryable: int = 0
    """Records `--retry-errors` would run again: errors, generation errors, deadline cuts."""


class RunComparison(BaseModel):
    system: str
    shared: int
    only_before: int
    only_after: int
    correct: dict[str, PairedBinary] = Field(default_factory=dict)
    """Per answerable type and pooled (`answerable`); a = before, b = after."""
    refusal: PairedBinary | None = None
    clean_refusal: PairedBinary | None = None
    cost: list[RunCostRow] = Field(default_factory=list)


def _chat_cny(record: SystemRecord) -> float:
    usage = record.usage_by_kind.get("chat")
    return usage.cny if usage else record.usage.cny


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cost_rows(label: str, records: Sequence[SystemRecord]) -> list[RunCostRow]:
    rows: list[RunCostRow] = []
    for qtype in (*QTYPES, "all"):
        group = list(records) if qtype == "all" else [r for r in records if r.qtype == qtype]
        if not group:
            continue
        lat = latency_summary([r.total_ms for r in group])
        rows.append(
            RunCostRow(
                label=label,
                qtype=qtype,
                n=len(group),
                mean_cny=_mean([r.usage.cny for r in group]),
                mean_chat_cny=_mean([_chat_cny(r) for r in group]),
                p50_ms=lat.p50_ms,
                p95_ms=lat.p95_ms,
                mean_rounds=_mean([r.rounds for r in group]),
                errors=sum(1 for r in group if r.status == "error"),
                truncated=sum(1 for r in group if is_deadline_truncated(r)),
                retryable=sum(1 for r in group if is_retryable(r)),
            )
        )
    return rows


def compare_runs(
    before: Sequence[SystemRecord],
    after: Sequence[SystemRecord],
    before_lines: Sequence[JudgeLine],
    after_lines: Sequence[JudgeLine],
    *,
    system: str,
) -> RunComparison:
    old = {r.qid: r for r in before if r.system == system}
    new = {r.qid: r for r in after if r.system == system}
    shared = sorted(set(old) & set(new))
    s_old = judge_scores(before_lines, system)
    s_new = judge_scores(after_lines, system)
    report = RunComparison(
        system=system,
        shared=len(shared),
        only_before=len(set(old) - set(new)),
        only_after=len(set(new) - set(old)),
    )
    for qtype in (*ANSWERABLE_TYPES, ANSWERABLE):
        qids = [
            q
            for q in shared
            if (old[q].answerable if qtype == ANSWERABLE else old[q].qtype == qtype)
            and q in s_old
            and q in s_new
        ]
        if qids:
            report.correct[qtype] = paired_binary(
                f"correct {qtype}",
                {q: s_old[q] >= CORRECT for q in qids},
                {q: s_new[q] >= CORRECT for q in qids},
            )
    refuse = [q for q in shared if not old[q].answerable]
    if refuse:
        report.refusal = paired_binary(
            "refusal",
            {q: not old[q].answered for q in refuse},
            {q: not new[q].answered for q in refuse},
        )
        report.clean_refusal = paired_binary(
            "clean refusal",
            {q: clean_refusal(old[q]) for q in refuse},
            {q: clean_refusal(new[q]) for q in refuse},
        )
    report.cost = _cost_rows("before", [old[q] for q in shared]) + _cost_rows(
        "after", [new[q] for q in shared]
    )
    return report


def _pb(label: str, r: PairedBinary) -> str:
    return (
        f"| {label} | {r.n} | {r.a.count} ({r.a.rate or 0:.1%}) | {r.b.count} ({r.b.rate or 0:.1%}) "
        f"| {r.improved} | {r.worsened} | {r.p_value:.3f} | {r.verdict.replace('B ', 'after ')} |"
    )


def _failure_lines(rows: Sequence[RunCostRow]) -> list[str]:
    """One line per run: answers cut by the deadline and records `--retry-errors` would re-run."""
    by_label = {r.label: r for r in rows if r.qtype == "all"}
    return [
        f"- {label}: {r.truncated} answers cut by the deadline (not refusals), "
        f"{r.errors} errors, {r.retryable} records `--retry-errors` would run again"
        for label in ("before", "after")
        if (r := by_label.get(label)) is not None
    ]


def render_compare_runs(report: RunComparison) -> str:
    text = [
        f"# {report.system}: new run (after) against the older run (before)",
        "",
        f"- shared questions: {report.shared}; only before: {report.only_before}; "
        f"only after: {report.only_after}",
        "- correct = judge score 3 (pass 1); 'after better' = correct now, not before; exact McNemar",
        "",
        "## Correct per type",
        "",
        "| type | n | before | after | after better | after worse | McNemar p | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    text += [_pb(t, r) for t, r in report.correct.items()]
    text += ["", "## Refusal on no-answer questions", ""]
    if report.refusal and report.clean_refusal:
        text += [
            "| metric | n | before | after | after more | after fewer | McNemar p | verdict |",
            "|---|---|---|---|---|---|---|---|",
            _pb("refusal (no answer given)", report.refusal),
            _pb("clean refusal (no error, not cut by the deadline)", report.clean_refusal),
        ]
    else:
        text.append("No no-answer questions shared.")
    text += [
        "",
        "## Cost and latency (shared questions)",
        "",
        "Chat CNY leaves out embedding rows (records before P3-07 have none).",
        "",
        *_failure_lines(report.cost),
        "",
        "| run | type | n | CNY / question | chat CNY / question | P50 s | P95 s | mean rounds | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    text += [
        f"| {r.label} | {r.qtype} | {r.n} | {r.mean_cny:.4f} | {r.mean_chat_cny:.4f} | "
        f"{r.p50_ms / 1000:.1f} | {r.p95_ms / 1000:.1f} | {r.mean_rounds:.2f} | {r.errors} |"
        for r in sorted(
            report.cost, key=lambda r: ((*QTYPES, "all").index(r.qtype), r.label != "before")
        )
    ]
    return "\n".join(text) + "\n"


__all__ = ["RunComparison", "RunCostRow", "compare_runs", "render_compare_runs"]
