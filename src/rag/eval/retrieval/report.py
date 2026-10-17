"""The short Markdown report written next to `results.json` (design s14-4)."""

from __future__ import annotations

from rag.eval.retrieval.metrics import LEVELS, metric_names
from rag.eval.retrieval.record import RunRecord
from rag.eval.stats import mcnemar_exact, min_detectable_difference


def resolution_sentence(n: int, *, alpha: float = 0.05) -> str:
    """How big a difference `n` questions can tell apart (written at the top of every report).

    A lower bound: the best case where no question gets worse. When questions change in both
    directions a larger difference is needed.
    """
    best = min_detectable_difference(n, alpha=alpha) if n > 0 else None
    if best is None:
        p = mcnemar_exact(n, 0)
        return (
            f"Resolution: {_plural(n, 'question')} cannot show a reliable change of a binary "
            f"metric (even all {n} better gives exact McNemar p = {p:.3g} > {alpha}): every "
            'difference is "no reliable difference".'
        )
    text = (
        f"Resolution: with {_plural(n, 'question')} a binary metric (Hit@k) has to move by at "
        f"least {best.difference * 100:.1f} points to be a reliable difference (best case: "
        f"{best.improved} questions better, none worse, exact McNemar p = {best.p_value:.3g} "
        f"<= {alpha})"
    )
    one = min_detectable_difference(n, alpha=alpha, worsened=1)
    if one is not None:
        text += (
            f"; with one question worse it takes {one.difference * 100:.1f} points "
            f"({one.improved} better)"
        )
    return (
        text + '. Smaller differences are "no reliable difference"; the more questions change '
        "in both directions, the larger the difference has to be."
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_run_report(record: RunRecord) -> str:
    run = record.run
    qs = run.question_set
    commit = run.git_commit[:12] if run.git_commit else "unknown"
    dirty = " (with uncommitted changes)" if run.git_dirty else ""
    models = ", ".join(f"{k} `{v}`" for k, v in run.models.items()) or "(not recorded)"
    lines = [
        f"# Retrieval eval {run.run_id}",
        "",
        f"- Knowledge base: `{run.kb}`" + (f", label `{run.label}`" if run.label else ""),
        f"- Created {run.created_at} at git commit `{commit}`{dirty}",
        f"- Question set: `{qs.path}`, {qs.count} questions, sha256 `{qs.sha256}`",
        "- Index version: "
        + (", ".join(f"`{v}`" for v in run.index_versions) or "(none)")
        + "; data generation: "
        + (", ".join(str(g) for g in run.data_generations) or "(none)"),
        f"- Models: {models}",
        f"- KB config sha256 `{run.kb_config_sha256}`; uv.lock sha256 `{run.lock_sha256}`",
        f"- Final list cut at top {run.top_k}; Hit@k, nDCG@k, Recall@k for k in "
        f"{', '.join(map(str, run.ks))}; MRR within the top {max(run.ks)}",
        "",
    ]
    if len(run.index_versions) > 1 or len(run.data_generations) > 1:
        lines += ["**The index changed during the run; the numbers mix index versions.**", ""]

    for level in LEVELS:
        present = [name for name in run.lists if level in record.summary.get(name, {})]
        if not present:
            continue
        n = record.summary[present[0]][level].n
        lines += [
            f"## {level.capitalize()}-level metrics ({_plural(n, 'question')} with {level} labels)",
            "",
            resolution_sentence(n),  # only the questions labelled at this level are compared
            "",
            "| metric | " + " | ".join(run.lists) + " |",
            "|---" * (len(run.lists) + 1) + "|",
        ]
        for metric in metric_names(run.ks):
            cells = []
            for name in run.lists:
                summary = record.summary[name].get(level)
                if summary is None:
                    cells.append("-")
                    continue
                cell = _fmt(summary.means[metric])
                if metric in summary.hit_wilson:
                    lo, hi = summary.hit_wilson[metric]
                    cell += f" [{lo:.2f}, {hi:.2f}]"
                cells.append(cell)
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Hit@k shows the 95% Wilson interval in brackets. `final` is the list returned to "
        "callers; each route column is that route's own list before fusion.",
        "",
        "## Latency and errors",
        "",
        f"- `retrieve()` wall time: P50 {record.latency.p50_ms:.1f} ms, P95 "
        f"{record.latency.p95_ms:.1f} ms, mean {record.latency.mean_ms:.1f} ms, max "
        f"{record.latency.max_ms:.1f} ms",
        f"- Errors {record.counts['errors']}, degraded (one route failed) "
        f"{record.counts['degraded']}, rerank timeouts {record.counts['rerank_timeout']}",
        "",
    ]
    errors = [q for q in record.questions if q.error]
    if errors:
        lines += ["## Errors", ""]
        lines += [f"- `{q.qid}`: {q.error}" for q in errors]
        lines.append("")
    if record.missed_by_all_lists:
        by_qid = {q.qid: q for q in record.questions}
        lines += [
            "## Questions no list found",
            "",
            "Check the label before the system (s14-2).",
            "",
        ]
        for qid in record.missed_by_all_lists:
            q = by_qid[qid]
            pages = ", ".join(f"`{p}`" for p in q.relevant_pages)
            lines.append(f"- `{qid}` {q.question.strip()} (expected {pages})")
        lines.append("")
    return "\n".join(lines)
