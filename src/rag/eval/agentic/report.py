# ruff: noqa: E501 (long lines are Markdown table rows and console messages)
"""The Markdown report of the comparison and the cost projection from a pilot (spec P3-05).

The report is English and mechanical: tables straight from `rag.eval.agentic.analysis`. The
owner-facing write-up (docs/reports/P3-report.md) is written by hand from these numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from rag.eval.agentic.analysis import (
    ANSWERABLE,
    ANSWERABLE_TYPES,
    DEADLINE_MS,
    TARGET_P95_MS,
    PairedBinary,
    by_system,
    cost_latency,
    judge_scores,
    judge_stability,
    limits_report,
    loop_report,
    min_detectable_table,
    per_type_summary,
    rewrite_effect,
)
from rag.eval.agentic.records import JudgeLine, SystemRecord
from rag.eval.agentic.sets import QTYPES

LEGEND = (
    "A = standard RAG, B = the ask_docs graph. Paired on the same questions. Rates carry the 95% "
    "Wilson interval. p = exact McNemar (binary) or paired sign-flip permutation (scores). A "
    "difference that is not significant (p > 0.05) is direction only, not a result."
)


def _p(x: float) -> str:
    return f"{x:.3f}" if x >= 0.001 else f"{x:.1e}"


def _binary_row(label: str, r: PairedBinary) -> str:
    return (
        f"| {label} | {r.n} | {r.a.text()} | {r.b.text()} | {r.improved} | {r.worsened} | "
        f"{r.difference * 100:+.1f} | {_p(r.p_value)} | {r.verdict} |"
    )


_BINARY_HEAD = [
    "| type | n | A | B | B better | B worse | diff (points) | p | verdict |",
    "|---|---|---|---|---|---|---|---|---|",
]


def render_report(records: Sequence[SystemRecord], lines: Sequence[JudgeLine]) -> str:
    out: list[str] = ["# Agentic RAG against standard RAG (spec P3-05)", "", LEGEND, ""]
    systems = by_system(records)
    sizes = Counter(r.qtype for r in systems["standard"].values())
    out += [
        "## Questions",
        "",
        "| type | standard | agentic |",
        "|---|---|---|",
        *(
            f"| {t} | {sizes[t]} | {sum(1 for r in systems['agentic'].values() if r.qtype == t)} |"
            for t in QTYPES
            if sizes[t] or any(r.qtype == t for r in systems["agentic"].values())
        ),
        "",
        "## Smallest difference each question count can show (minimum detectable difference)",
        "",
        "Net change of a binary metric (exact McNemar, p <= 0.05): when nothing got worse, and "
        "when two questions got worse.",
        "",
        "| set | n | best case | with two worse |",
        "|---|---|---|---|",
    ]
    detect_sizes = {t: sizes[t] for t in QTYPES if sizes[t]}
    answerable = sum(sizes[t] for t in ANSWERABLE_TYPES)
    if answerable:
        detect_sizes[ANSWERABLE] = answerable
    for row in min_detectable_table(detect_sizes):
        best = "none" if row.best_case is None else f"{row.best_case * 100:.1f} points"
        two = "none" if row.with_two_worse is None else f"{row.with_two_worse * 100:.1f} points"
        out.append(f"| {row.name} | {row.n} | {best} | {two} |")

    summary = per_type_summary(records, lines)
    out += ["", "## Answer quality (judge, 1-3 scale, first pass)", "", *_BINARY_HEAD]
    for key in (*ANSWERABLE_TYPES, ANSWERABLE):
        s = summary.get(key)
        if s and s.correct:
            out.append(_binary_row(f"{key}: correct (score 3)", s.correct))
    out += ["", *_BINARY_HEAD]
    for key in (*ANSWERABLE_TYPES, ANSWERABLE):
        s = summary.get(key)
        if s and s.adequate:
            out.append(_binary_row(f"{key}: adequate (score >= 2)", s.adequate))
    out += [
        "",
        "| type | n | unscored | A mean % | B mean % | diff (points) | better | worse | p (permutation) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in (*ANSWERABLE_TYPES, ANSWERABLE):
        s = summary.get(key)
        if s and s.score:
            c = s.score
            out.append(
                f"| {key} | {s.n} | {s.unscored} | {c.a_pct:.1f} | {c.b_pct:.1f} | "
                f"{c.diff_pct:+.1f} | {c.improved} | {c.worsened} | {_p(c.p_value)} |"
            )

    na = summary.get("no_answer")
    out += ["", "## No-answer questions: refusal rate (no answer given)", ""]
    if na and na.refusal:
        out += [*_BINARY_HEAD, _binary_row("no_answer: refused", na.refusal)]
        if na.clean_refusal:
            out.append(_binary_row("no_answer: refused cleanly (no error)", na.clean_refusal))
        out += [
            "",
            f"Runs that ended in an error (counted as refusals in the first row): "
            f"standard {na.errors.get('standard', 0)}, agentic {na.errors.get('agentic', 0)}.",
            "",
            f"Answers cut off by the total deadline (counted as refusals in the first row, not in "
            f"the clean row; listed below): standard {na.truncated.get('standard', 0)}, "
            f"agentic {na.truncated.get('agentic', 0)}.",
        ]
        out += [
            "",
            f"Clarifications counted as refusals: standard {na.clarifications.get('standard', 0)}, "
            f"agentic {na.clarifications.get('agentic', 0)}.",
        ]
        for name in sorted({r.set for r in systems["standard"].values() if r.qtype == "no_answer"}):
            ids = [q for q, r in systems["standard"].items() if r.set == name]
            a_n = sum(1 for q in ids if not systems["standard"][q].answered)
            b_n = sum(
                1 for q in ids if q in systems["agentic"] and not systems["agentic"][q].answered
            )
            out.append(
                f"- {name}: standard refused {a_n}/{len(ids)}, agentic refused {b_n}/{len(ids)}"
            )

    loop = loop_report(records)
    out += [
        "",
        "## Inside the graph",
        "",
        "| type | n | asked back | rounds 0/1/2/3 | ran round 2 | ran round 3 | first verdicts | "
        "gold in round 1 | gold in any round | gold in answer | round 1 missed | rescued | "
        "lost to context |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, s in loop.by_type.items():
        dist = "/".join(str(s.rounds.get(i, 0)) for i in range(4))
        out.append(
            f"| {key} | {s.n} | {s.clarifications} | {dist} | {s.ran_round2} | {s.ran_round3} | "
            f"{s.first_verdicts} | {s.hit_round1}/{s.searched_answerable} | "
            f"{s.hit_all_rounds}/{s.searched_answerable} | {s.hit_context}/{s.searched_answerable} | "
            f"{s.round1_missed} | {s.rescued} | {s.lost_to_context} |"
        )
    j = loop.judge_vs_label
    out += [
        "",
        f"Judge's round-one verdict against the label (gold page in round one): said enough and "
        f"right {j.tp}, said enough and wrong {j.fp}, said more needed and the page was there "
        f"{j.fn}, said more needed and it was not there {j.tn}; accuracy {j.accuracy.text()}.",
        "",
        "False clarification (the graph asked back on an answerable question):",
        "",
        *(f"- {k}: {v.text()}" for k, v in loop.false_clarification.items()),
    ]
    out += [f"- asked back: {qid}: {q}" for qid, q in loop.clarified_questions]

    out += [
        "",
        "## Effect of the rewrite on retrieval (gold page retrieved: raw question vs round one)",
        "",
        *_BINARY_HEAD,
    ]
    for key, r in rewrite_effect(records).items():
        out.append(_binary_row(key, r))

    out += [
        "",
        "## Cost and latency per question",
        "",
        "| system | type | n | chat calls | cheap | online | embed calls | prompt tok | completion tok | "
        "CNY/question | total CNY | rounds | P50 s | P95 s | max s | > 15 s | > 30 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cost_latency(records):
        out.append(
            f"| {c.system} | {c.qtype} | {c.n} | {c.mean_chat_calls:.2f} | {c.mean_cheap_calls:.2f} | "
            f"{c.mean_online_calls:.2f} | {c.mean_embedding_calls:.2f} | {c.mean_prompt_tokens:.0f} | "
            f"{c.mean_completion_tokens:.0f} | {c.mean_cny:.4f} | {c.total_cny:.3f} | "
            f"{c.mean_rounds:.2f} | {c.p50_ms / 1000:.1f} | {c.p95_ms / 1000:.1f} | "
            f"{c.max_ms / 1000:.1f} | {c.over_15s} | {c.over_30s} |"
        )
    out += [
        "",
        f"Design target: ask_docs P95 {TARGET_P95_MS / 1000:.0f} s; default total deadline "
        f"{DEADLINE_MS / 1000:.0f} s. Mean step time (ms) of the graph, all types:",
        "",
    ]
    for c in cost_latency(records):
        if c.system == "agentic" and c.qtype == "all":
            out.append(", ".join(f"{k} {v:.0f}" for k, v in sorted(c.mean_steps_ms.items())))

    lim = limits_report(records, lines)
    out += [
        "",
        "## Limits, fallbacks and degraded answers (graph)",
        "",
        f"- stop reasons: {lim.stop_reasons or 'none'}; coverage_unconfirmed {lim.coverage_unconfirmed}; "
        f"runs over the {DEADLINE_MS / 1000:.0f} s deadline {lim.over_deadline_ms}",
        f"- degraded {lim.degraded}, rerank_timeout {lim.rerank_timeout}, fallbacks {lim.fallbacks or 'none'}, "
        f"error categories {lim.error_categories or 'none'}",
        f"- answerable questions that stopped at a limit: {lim.limited_answerable}, of which "
        f"correct {lim.limited_correct}",
    ]
    out += [
        f"  - {c.qid} ({c.qtype}): rounds {c.rounds}, stop {c.stop_reason}, {c.total_ms / 1000:.1f} s, "
        f"score {c.score}, answered {c.answered}"
        + (" (cut by the deadline)" if c.truncated else "")
        for c in lim.cases
    ]

    out += [
        "",
        "## Answers cut by the deadline",
        "",
        "The graph stopped at the total deadline before the answer step ran, and the record says "
        "`not_found` with no error. That is not a refusal the system chose: it is left out of the "
        "clean-refusal rows and counted here (spec EVAL-0). An answerable question here has "
        "judge score 1 (no answer); `--retry-errors` runs these again.",
        "",
    ]
    out += [
        f"- {c.qid} ({c.qtype}): rounds {c.rounds}, {c.total_ms / 1000:.1f} s, score {c.score}"
        for c in lim.truncated
    ] or ["- none"]

    out += ["", "## Judge stability (two passes over the same answers)", ""]
    for system in ("standard", "agentic"):
        st = judge_stability(lines, system)
        if st.n or st.n_missing:
            out.append(
                f"- {system}: {st.n} scored in both passes; score changed {st.score_flips.text()}; "
                f"correct/not changed {st.correct_flips.text()}; rule-scored left out {st.n_rule_excluded}"
            )

    out += ["", "## Questions where the graph did worse (answerable)", ""]
    a_scores = judge_scores(lines, "standard")
    b_scores = judge_scores(lines, "agentic")
    for qid in sorted(set(a_scores) & set(b_scores)):
        if b_scores[qid] < a_scores[qid]:
            r = systems["agentic"][qid]
            out.append(
                f"- {qid} ({r.qtype}): standard {a_scores[qid]}, agentic {b_scores[qid]}; "
                f"rounds {r.rounds}, status {r.status}, stop {r.stop_reason}; {r.question[:100]}"
            )
    return "\n".join(out) + "\n"


class CostProjection(BaseModel):
    by_system: dict[str, float]
    """Expected CNY of all questions of `sizes`, per system."""
    remaining_by_system: dict[str, float]
    """The same for the questions not yet in the pilot."""
    judge: float
    total: float
    remaining_total: float
    mean_cny: dict[str, dict[str, float]]


def project_cost(
    pilot: Sequence[SystemRecord],
    sizes: Mapping[str, int],
    *,
    judge_cny_per_call: float,
    judge_passes: int = 2,
) -> CostProjection:
    """Cost of the whole run from the pilot's mean CNY per question of each type (a type the
    pilot did not reach gets the system's mean over all pilot questions)."""
    by_system_cost: dict[str, float] = {}
    remaining: dict[str, float] = {}
    means: dict[str, dict[str, float]] = {}
    for system in ("standard", "agentic"):
        mine = [r for r in pilot if r.system == system]
        overall = sum(r.usage.cny for r in mine) / len(mine) if mine else 0.0
        means[system] = {}
        total = left = 0.0
        for qtype, size in sizes.items():
            group = [r for r in mine if r.qtype == qtype]
            mean = sum(r.usage.cny for r in group) / len(group) if group else overall
            means[system][qtype] = mean
            total += size * mean
            left += max(0, size - len(group)) * mean
        by_system_cost[system], remaining[system] = total, left
    answerable = sum(n for t, n in sizes.items() if t != "no_answer")
    judge = answerable * 2 * judge_passes * judge_cny_per_call
    return CostProjection(
        by_system=by_system_cost,
        remaining_by_system=remaining,
        judge=judge,
        total=sum(by_system_cost.values()) + judge,
        remaining_total=sum(remaining.values()) + judge,
        mean_cny=means,
    )
