"""Compare two eval runs question by question (design s14-4).

The report first says how many questions got better, worse or stayed the same, per metric,
with exact McNemar for Hit@k and the paired permutation test for the continuous metrics; then
it lists the questions that changed on the chosen metric with each run's top hits.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rag.eval.retrieval.metrics import LEVELS, Level, is_binary_metric, metric_names
from rag.eval.retrieval.record import FINAL, QuestionRecord, RunRecord
from rag.eval.retrieval.report import resolution_sentence
from rag.eval.stats import (
    CHANGE_TOL,
    PairedChanges,
    mcnemar_exact,
    paired_changes,
    paired_permutation_test,
)


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    mean_a: float
    mean_b: float
    changes: PairedChanges
    test: str
    """`mcnemar` (binary metrics) or `permutation`."""
    p_value: float
    exact: bool
    iterations: int
    """Sign patterns of the permutation test (0 for McNemar)."""

    @property
    def diff(self) -> float:
        return self.mean_b - self.mean_a

    def reliable(self, alpha: float) -> bool:
        return self.p_value <= alpha


@dataclass(frozen=True)
class ChangedQuestion:
    qid: str
    question: str
    value_a: float
    value_b: float
    rank_a: int | None
    rank_b: int | None
    """Rank of the first relevant item in each run's whole list."""
    top_a: list[str]
    top_b: list[str]


@dataclass(frozen=True)
class Comparison:
    run_a: str
    run_b: str
    list_name: str
    level: Level
    metric: str
    """The metric the changed questions are listed by."""
    alpha: float
    n: int
    only_in_a: list[str]
    only_in_b: list[str]
    same_question_set: bool
    differences: list[str]
    counts: tuple[dict[str, int], dict[str, int]]
    metrics: list[MetricComparison]
    improved: list[ChangedQuestion]
    worsened: list[ChangedQuestion]


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(f"{prefix}.{key}" if prefix else str(key), item)
    else:
        yield prefix, value


def _short(value: Any) -> str:
    if isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value):
        return value[:12]  # git commits
    if isinstance(value, str) and len(value) == 64:
        return value[:12] + "..."  # sha256
    return str(value)


def run_differences(a: RunRecord, b: RunRecord) -> list[str]:
    """What differs between the runs' recorded setups (`key: a -> b`)."""
    skip = {"run_id", "created_at", "public", "lists"}
    fa = dict(_flatten("", a.run.model_dump(mode="json", exclude=skip)))
    fb = dict(_flatten("", b.run.model_dump(mode="json", exclude=skip)))
    return [
        f"{key}: {_short(fa.get(key, '(absent)'))} -> {_short(fb.get(key, '(absent)'))}"
        for key in dict.fromkeys([*fa, *fb])
        if fa.get(key) != fb.get(key)
    ]


def _top(q: QuestionRecord, list_name: str, level: Level, n: int) -> list[str]:
    hits = q.hits if list_name == FINAL else q.routes.get(list_name, [])
    return [h.doc_id if level == "page" else h.chunk_id for h in hits][:n]


def compare_runs(
    a: RunRecord,
    b: RunRecord,
    *,
    list_name: str = FINAL,
    level: Level = "page",
    metric: str | None = None,
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    top_n: int = 10,
) -> Comparison:
    """Pair the questions both runs scored on `list_name` at `level` (by qid) and test each metric.

    `metric` (default: the MRR) picks which questions are listed as improved or worsened.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    for name, run in (("A", a), ("B", b)):
        if list_name not in run.run.lists:
            raise ValueError(f"list {list_name!r} is not in run {name} (has {run.run.lists})")
    names_b = set(metric_names(b.run.ks))
    names = [m for m in metric_names(a.run.ks) if m in names_b]
    metric = metric or next(m for m in names if m.startswith("mrr@"))
    if metric not in names:
        raise ValueError(f"metric {metric!r} is not in both runs (common metrics: {names})")

    def scored(run: RunRecord) -> dict[str, QuestionRecord]:
        return {q.qid: q for q in run.questions if level in q.metrics.get(list_name, {})}

    qa, qb = scored(a), scored(b)
    paired = [qid for qid in qa if qid in qb]

    def values(qs: dict[str, QuestionRecord], m: str) -> list[float]:
        return [qs[qid].metrics[list_name][level][m] for qid in paired]

    results = []
    for m in names:
        va, vb = values(qa, m), values(qb, m)
        changes = paired_changes(va, vb)
        mean_a = sum(va) / len(va) if va else 0.0
        mean_b = sum(vb) / len(vb) if vb else 0.0
        if is_binary_metric(m):
            p = mcnemar_exact(changes.improved, changes.worsened)
            results.append(MetricComparison(m, mean_a, mean_b, changes, "mcnemar", p, True, 0))
            continue
        perm = paired_permutation_test(va, vb, iterations=iterations, seed=seed)
        results.append(
            MetricComparison(
                m, mean_a, mean_b, changes, "permutation", perm.p_value, perm.exact, perm.iterations
            )
        )

    changed = []
    for qid in paired:
        x, y = qa[qid], qb[qid]
        va, vb = x.metrics[list_name][level][metric], y.metrics[list_name][level][metric]
        if abs(vb - va) > CHANGE_TOL:
            changed.append(
                ChangedQuestion(
                    qid=qid,
                    question=x.question,
                    value_a=va,
                    value_b=vb,
                    rank_a=x.first_relevant_rank[list_name].get(level),
                    rank_b=y.first_relevant_rank[list_name].get(level),
                    top_a=_top(x, list_name, level, top_n),
                    top_b=_top(y, list_name, level, top_n),
                )
            )
    improved = sorted(
        (c for c in changed if c.value_b > c.value_a), key=lambda c: c.value_a - c.value_b
    )
    worsened = sorted(
        (c for c in changed if c.value_b < c.value_a), key=lambda c: c.value_b - c.value_a
    )
    return Comparison(
        run_a=a.run.run_id,
        run_b=b.run.run_id,
        list_name=list_name,
        level=level,
        metric=metric,
        alpha=alpha,
        n=len(paired),
        only_in_a=[qid for qid in qa if qid not in qb],
        only_in_b=[qid for qid in qb if qid not in qa],
        same_question_set=a.run.question_set.sha256 == b.run.question_set.sha256,
        differences=run_differences(a, b),
        counts=(a.counts, b.counts),
        metrics=results,
        improved=improved,
        worsened=worsened,
    )


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _test_name(m: MetricComparison) -> str:
    if m.test == "mcnemar":
        return "McNemar exact"
    if m.exact:
        return f"permutation (exact, {m.iterations} sign patterns)"
    return f"permutation ({m.iterations} random sign patterns)"


def _questions(n: int) -> str:
    return f"{n} question" + ("" if n == 1 else "s")


def render_comparison(c: Comparison) -> str:
    ca, cb = c.counts
    lines = [
        f"# Compare {c.run_a} -> {c.run_b}",
        "",
        f"List `{c.list_name}`, {c.level} level, {_questions(c.n)} paired by qid.",
    ]
    if not c.same_question_set:
        lines.append(
            "**The runs used different question sets** (sha256 differs); only the questions "
            "in both are compared."
        )
    if c.only_in_a or c.only_in_b:
        lines.append(
            f"Only in A: {', '.join(c.only_in_a) or '-'}; "
            f"only in B: {', '.join(c.only_in_b) or '-'}."
        )
    lines += [
        f"Errors A {ca.get('errors', 0)}, B {cb.get('errors', 0)}; degraded A "
        f"{ca.get('degraded', 0)}, B {cb.get('degraded', 0)}.",
        "",
        resolution_sentence(c.n, alpha=c.alpha),
        "",
        "## What differs between the runs",
        "",
        *([f"- {d}" for d in c.differences] or ["- (nothing recorded differs)"]),
        "",
        "## Metrics",
        "",
        "| metric | A | B | B - A | better | worse | same | test | p | reliable |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in c.metrics:
        ch = m.changes
        lines.append(
            f"| {m.metric} | {m.mean_a:.3f} | {m.mean_b:.3f} | {m.diff:+.3f} | {ch.improved} | "
            f"{ch.worsened} | {ch.unchanged} | {_test_name(m)} | {m.p_value:.3g} | "
            f"{'yes' if m.reliable(c.alpha) else 'no'} |"
        )
    lines += [
        "",
        f'`reliable` means p <= {c.alpha}; otherwise the difference is "no reliable '
        'difference". p-values are per metric and not corrected for looking at several '
        "metrics: decide on the metric chosen before the run.",
        "",
    ]
    for title, group in (("Improved", c.improved), ("Worsened", c.worsened)):
        lines += [f"## {title} on {c.metric}: {_questions(len(group))}", ""]
        for q in group:
            lines += [
                f"- `{q.qid}` {q.question.strip()} {c.metric} {q.value_a:.3f} -> {q.value_b:.3f}, "
                f"first relevant rank {_rank(q.rank_a)} -> {_rank(q.rank_b)}",
                f"  - A: {', '.join(f'`{i}`' for i in q.top_a) or '(nothing)'}",
                f"  - B: {', '.join(f'`{i}`' for i in q.top_b) or '(nothing)'}",
            ]
        lines.append("")
    return "\n".join(lines)
