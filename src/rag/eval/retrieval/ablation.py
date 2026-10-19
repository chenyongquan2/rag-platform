"""Ablation table: several variant runs against one baseline run (spec P2-03, design s14-4).

Each variant is compared with the baseline question by question (`compare_runs`): how many
questions got better, worse or stayed the same on each metric, the test (exact McNemar for Hit@k,
paired permutation for MRR and nDCG) and its p-value. The table also gives each run's Hit@k with
its Wilson interval and its latency, and says which variants moved less than the smallest
difference the question count can show (their "no reliable difference" means too few questions to
tell, not that the variants are equal).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.eval.retrieval.compare import MetricComparison, compare_runs
from rag.eval.retrieval.metrics import Level, is_binary_metric, metric_names
from rag.eval.retrieval.record import FINAL, RunRecord
from rag.eval.retrieval.report import resolution_sentence
from rag.eval.stats import min_detectable_difference


@dataclass(frozen=True)
class AblationRow:
    variant: str
    metric: str
    mean_baseline: float
    mean_variant: float
    better: int
    worse: int
    unchanged: int
    test: str
    p_value: float
    verdict: str
    """`better, reliable`, `worse, reliable` or `no reliable difference`."""


@dataclass(frozen=True)
class AblationTable:
    baseline: RunRecord
    variants: dict[str, RunRecord]
    list_name: str
    level: Level
    alpha: float
    n: int
    """Questions scored in the baseline and in every variant (the table's paired set)."""
    rows: list[AblationRow]


def default_metrics(run: RunRecord) -> list[str]:
    """Hit@1, Hit@5, MRR and nDCG at the longest k (those that exist in the run)."""
    names = metric_names(run.run.ks)
    k = max(run.run.ks)
    wanted = ["hit@1", "hit@5", f"mrr@{k}", f"ndcg@{k}"]
    return [m for m in wanted if m in names]


def _verdict(m: MetricComparison, alpha: float) -> str:
    if not m.reliable(alpha):
        return "no reliable difference"
    return "better, reliable" if m.diff > 0 else "worse, reliable"


def _scored(run: RunRecord, list_name: str, level: Level) -> set[str]:
    return {q.qid for q in run.questions if level in q.metrics.get(list_name, {})}


def ablation_table(
    baseline: RunRecord,
    variants: Mapping[str, RunRecord],
    *,
    metrics: Sequence[str] | None = None,
    list_name: str = FINAL,
    level: Level = "page",
    alpha: float = 0.05,
    iterations: int = 10_000,
    seed: int = 0,
) -> AblationTable:
    if not variants:
        raise ValueError("an ablation needs at least one variant")
    metrics = list(metrics) if metrics else default_metrics(baseline)
    rows: list[AblationRow] = []
    paired = _scored(baseline, list_name, level)
    for name, run in variants.items():
        paired &= _scored(run, list_name, level)
        comparison = compare_runs(
            baseline,
            run,
            list_name=list_name,
            level=level,
            iterations=iterations,
            seed=seed,
            alpha=alpha,
        )
        by_metric = {m.metric: m for m in comparison.metrics}
        for metric in metrics:
            if metric not in by_metric:
                raise ValueError(
                    f"metric {metric!r} is not in both runs (common metrics: {list(by_metric)})"
                )
            m = by_metric[metric]
            rows.append(
                AblationRow(
                    variant=name,
                    metric=metric,
                    mean_baseline=m.mean_a,
                    mean_variant=m.mean_b,
                    better=m.changes.improved,
                    worse=m.changes.worsened,
                    unchanged=m.changes.unchanged,
                    test="McNemar exact" if is_binary_metric(metric) else "permutation",
                    p_value=m.p_value,
                    verdict=_verdict(m, alpha),
                )
            )
    return AblationTable(
        baseline=baseline,
        variants=dict(variants),
        list_name=list_name,
        level=level,
        alpha=alpha,
        n=len(paired),
        rows=rows,
    )


def _hit_cells(run: RunRecord, list_name: str, level: Level) -> str:
    summary = run.summary[list_name][level]
    return "; ".join(
        f"{key} {summary.means[key]:.3f} [{lo:.2f}, {hi:.2f}]"
        for key, (lo, hi) in summary.hit_wilson.items()
    )


def render_ablation(table: AblationTable, *, baseline_name: str = "baseline") -> str:
    base = table.baseline
    lines = [
        f"# Ablation against `{baseline_name}` ({base.run.run_id})",
        "",
        f"List `{table.list_name}`, {table.level} level, {table.n} questions paired by qid.",
        "",
        resolution_sentence(table.n, alpha=table.alpha),
        "",
        "## Better, worse, unchanged",
        "",
        "| variant | metric | baseline | variant | better | worse | unchanged | test | p |"
        " verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        lines.append(
            f"| {r.variant} | {r.metric} | {r.mean_baseline:.3f} | {r.mean_variant:.3f} | "
            f"{r.better} | {r.worse} | {r.unchanged} | {r.test} | {r.p_value:.3g} | {r.verdict} |"
        )
    lines += [
        "",
        "`p` is per metric and not corrected for looking at several metrics; reliable means "
        f"p <= {table.alpha}.",
        "",
    ]
    best = min_detectable_difference(table.n, alpha=table.alpha) if table.n else None
    if best is not None:
        small = sorted(
            {
                r.variant
                for r in table.rows
                if is_binary_metric(r.metric)
                and r.verdict == "no reliable difference"
                and abs(r.mean_variant - r.mean_baseline) < best.difference
            }
        )
        if small:
            names = ", ".join(f"`{v}`" for v in small)
            lines += [
                f"Variants whose Hit change is below the smallest difference {table.n} questions "
                f"can show ({best.difference * 100:.1f} points): {names}. "
                'For them "no reliable difference" means too few questions to tell, not that '
                "the variants are equal.",
                "",
            ]
    lines += [
        "## Hit@k with Wilson 95% intervals, and latency",
        "",
        "| run | Hit@k (Wilson) | P50 ms | P95 ms | max ms | errors | rerank timeouts |",
        "|---|---|---|---|---|---|---|",
    ]
    runs = {baseline_name: base, **table.variants}
    for name, run in runs.items():
        lat, counts = run.latency, run.counts
        lines.append(
            f"| {name} | {_hit_cells(run, table.list_name, table.level)} | {lat.p50_ms:.0f} | "
            f"{lat.p95_ms:.0f} | {lat.max_ms:.0f} | {counts.get('errors', 0)} | "
            f"{counts.get('rerank_timeout', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)
