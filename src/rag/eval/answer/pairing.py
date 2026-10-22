"""Two chat models paired question by question on one question set (spec P7-01, design s14-4).

Each system is a mapping from question id to a `PairedOutcome`: the judge's 1-3 score, whether
the answer cites every sentence (format compliance, design s11-2 level 1) and the citation check
(level 2: supported pairs and per-answer recall). Only question ids both systems have are paired.

- Binary metrics (score 3, found, every sentence cited): exact McNemar on the discordant questions.
- Continuous metrics (judge score, per-answer citation recall): paired sign-flip permutation test,
  on the questions where both sides have a value.
- Citation precision is pooled over the (sentence, fragment) pairs of each side; the pairs differ
  between the two models, so it gets Wilson intervals, not a paired test.
- Latency (generation step) as P50 and P95, cost as mean CNY per question.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rag.eval.answer.record import Proportion
from rag.eval.stats import CHANGE_TOL, mcnemar_exact, paired_permutation_test


@dataclass(frozen=True)
class PairedOutcome:
    """One question answered by one system."""

    qid: str
    score: int | None
    """Judge score on the 1-3 scale (None: not scored)."""
    found: bool
    fully_cited: bool
    """The answer was given and every sentence carries a valid citation marker."""
    citation_recall: float | None
    """Share of the answer's sentences supported by a cited fragment; None: not checked."""
    supported_pairs: int
    judged_pairs: int
    """(sentence, cited fragment) pairs the citation judge said `supported`, of those it judged."""
    generation_ms: float
    cny: float


@dataclass(frozen=True)
class PairedRow:
    metric: str
    test: Literal["mcnemar", "permutation"]
    n: int
    base: float
    """Count of true values (binary) or mean (continuous) on the paired questions."""
    candidate: float
    better: int
    worse: int
    unchanged: int
    mean_diff: float
    """Candidate minus base: share of questions (binary) or mean difference (continuous)."""
    p_value: float


@dataclass(frozen=True)
class PairedAnswerTable:
    n: int
    rows: list[PairedRow]
    base_precision: Proportion
    candidate_precision: Proportion
    base_latency_p50: float
    base_latency_p95: float
    candidate_latency_p50: float
    candidate_latency_p95: float
    base_cny_mean: float
    candidate_cny_mean: float


_BINARY: list[tuple[str, Callable[[PairedOutcome], bool]]] = [
    ("score 3", lambda o: o.score == 3),
    ("found", lambda o: o.found),
    ("every sentence cited", lambda o: o.fully_cited),
]
_CONTINUOUS: list[tuple[str, Callable[[PairedOutcome], float | None]]] = [
    ("judge score (1-3)", lambda o: None if o.score is None else float(o.score)),
    ("citation recall per answer", lambda o: o.citation_recall),
]


def _binary_row(metric: str, a: list[bool], b: list[bool]) -> PairedRow:
    better = sum(y and not x for x, y in zip(a, b, strict=True))
    worse = sum(x and not y for x, y in zip(a, b, strict=True))
    n = len(a)
    return PairedRow(
        metric=metric,
        test="mcnemar",
        n=n,
        base=float(sum(a)),
        candidate=float(sum(b)),
        better=better,
        worse=worse,
        unchanged=n - better - worse,
        mean_diff=(better - worse) / n if n else 0.0,
        p_value=mcnemar_exact(better, worse),
    )


def _continuous_row(
    metric: str, a: list[float], b: list[float], *, iterations: int, seed: int
) -> PairedRow:
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    better = sum(d > CHANGE_TOL for d in diffs)
    worse = sum(d < -CHANGE_TOL for d in diffs)
    if a:
        result = paired_permutation_test(a, b, iterations=iterations, seed=seed)
        mean_diff, p = result.mean_diff, result.p_value
    else:
        mean_diff, p = 0.0, 1.0
    return PairedRow(
        metric=metric,
        test="permutation",
        n=len(a),
        base=float(np.mean(a)) if a else 0.0,
        candidate=float(np.mean(b)) if b else 0.0,
        better=better,
        worse=worse,
        unchanged=len(a) - better - worse,
        mean_diff=mean_diff,
        p_value=p,
    )


def paired_answer_table(
    base: Mapping[str, PairedOutcome],
    candidate: Mapping[str, PairedOutcome],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> PairedAnswerTable:
    """Pair `candidate` against `base` on their common question ids."""
    qids = sorted(set(base) & set(candidate))
    if not qids:
        raise ValueError("no common question ids between the two systems")
    a = [base[q] for q in qids]
    b = [candidate[q] for q in qids]
    rows = [_binary_row(name, [f(x) for x in a], [f(y) for y in b]) for name, f in _BINARY]
    for name, g in _CONTINUOUS:
        both = [(g(x), g(y)) for x, y in zip(a, b, strict=True)]
        kept = [(x, y) for x, y in both if x is not None and y is not None]
        rows.append(
            _continuous_row(
                name,
                [x for x, _ in kept],
                [y for _, y in kept],
                iterations=iterations,
                seed=seed,
            )
        )

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    la, lb = [x.generation_ms for x in a], [y.generation_ms for y in b]
    return PairedAnswerTable(
        n=len(qids),
        rows=rows,
        base_precision=Proportion.of(
            sum(x.supported_pairs for x in a), sum(x.judged_pairs for x in a)
        ),
        candidate_precision=Proportion.of(
            sum(y.supported_pairs for y in b), sum(y.judged_pairs for y in b)
        ),
        base_latency_p50=pct(la, 50),
        base_latency_p95=pct(la, 95),
        candidate_latency_p50=pct(lb, 50),
        candidate_latency_p95=pct(lb, 95),
        base_cny_mean=float(np.mean([x.cny for x in a])),
        candidate_cny_mean=float(np.mean([y.cny for y in b])),
    )


def _value(row: PairedRow, v: float) -> str:
    return f"{v:.0f}/{row.n}" if row.test == "mcnemar" else f"{v:.3f}"


def render_paired_answers(table: PairedAnswerTable, *, base_name: str, candidate_name: str) -> str:
    """Markdown: one row per metric with both values, better/worse/unchanged and the test's p."""
    lines = [
        f"## {candidate_name} against {base_name} (n = {table.n} paired questions)",
        "",
        f"| metric | {base_name} | {candidate_name} | better | worse | unchanged | difference "
        "| test | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        diff = f"{r.mean_diff:+.1%}" if r.test == "mcnemar" else f"{r.mean_diff:+.3f}"
        lines.append(
            f"| {r.metric} (n {r.n}) | {_value(r, r.base)} | {_value(r, r.candidate)} | "
            f"{r.better} | {r.worse} | {r.unchanged} | {diff} | {r.test} | {r.p_value:.4f} |"
        )
    lines += [
        "",
        f"- citation precision (pooled pairs, not paired): {base_name} "
        f"{table.base_precision.text()}; {candidate_name} {table.candidate_precision.text()}",
        f"- generation latency ms: {base_name} P50 {table.base_latency_p50:.0f}, "
        f"P95 {table.base_latency_p95:.0f}; {candidate_name} P50 "
        f"{table.candidate_latency_p50:.0f}, P95 {table.candidate_latency_p95:.0f}",
        f"- CNY per question (ledger price table): {base_name} {table.base_cny_mean:.5f}; "
        f"{candidate_name} {table.candidate_cny_mean:.5f}",
        "",
    ]
    return "\n".join(lines)
