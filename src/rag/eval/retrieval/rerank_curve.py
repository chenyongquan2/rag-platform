"""How many candidates to send to the reranker: latency and nDCG per rerank_n (design s10-1).

For each question the retriever (fusion and dedupe, no rerank) is asked once for
`max(rerank_ns)` candidates. Then, for each n, the first n candidates in fusion order go to the
reranker; the call is timed and the reranked top `final_k` is scored at page level. The n values
are interleaved per question so slow drift (CPU heat, other load) hits every n alike. A first
point `rerank_n = None` scores the fusion order without rerank. Only the reranker call is timed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from rag.eval.questions import QuestionSet
from rag.eval.retrieval.metrics import cap_ks, compute_metrics, metric_names, ranked_ids
from rag.eval.retrieval.protocol import Reranker, Retriever
from rag.eval.retrieval.record import Provenance, QuestionSetInfo
from rag.eval.retrieval.runner import DEFAULT_KS, collect_provenance
from rag.types import Chunk


class CurvePoint(BaseModel):
    rerank_n: int | None
    """None: the fusion order, no rerank."""
    questions: int
    """Questions with page labels (the metric means are over these)."""
    mean_candidates: float
    """Candidates actually sent (fewer than n when the retriever returned fewer)."""
    p50_ms: float | None = None
    p95_ms: float | None = None
    mean_ms: float | None = None
    ms_per_candidate: float | None = None
    means: dict[str, float]


class RerankCurve(BaseModel):
    kb: str
    created_at: str
    question_set: QuestionSetInfo
    final_k: int
    ks: list[int]
    rerank_ns: list[int]
    warmup: int
    provenance: Provenance
    points: list[CurvePoint]


def rerank_latency_curve(
    retriever: Retriever,
    reranker: Reranker | Callable[[str, Sequence[Chunk]], Sequence[Chunk]],
    question_set: QuestionSet,
    *,
    kb: str,
    rerank_ns: Sequence[int] = (10, 20, 30),
    final_k: int = 8,
    ks: Sequence[int] = DEFAULT_KS,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    provenance: Provenance | None = None,
    now: datetime | None = None,
) -> RerankCurve:
    """Measure the reranker for each n in `rerank_ns` (see the module docstring).

    `warmup` unmeasured reranker calls on the first question's candidates come first (the first
    call of a model is slower). A k above `final_k` is capped at `final_k` with a warning.
    Retriever and reranker errors are not caught.
    """
    if not rerank_ns or min(rerank_ns) < 1:
        raise ValueError(f"rerank_ns must be positive, got {rerank_ns}")
    question_set.check_kb(kb)
    ks = cap_ks(ks, final_k)
    names = metric_names(ks)
    top = max(rerank_ns)
    candidates = [
        list(retriever.retrieve(q.question, kb=kb, top_k=top).chunks)
        for q in question_set.questions
    ]
    for _ in range(warmup):
        reranker(question_set.questions[0].question, candidates[0])

    latencies: dict[int, list[float]] = {n: [] for n in rerank_ns}
    sent: dict[int, list[int]] = {n: [] for n in rerank_ns}
    scores: dict[int | None, list[dict[str, float]]] = {None: [], **{n: [] for n in rerank_ns}}
    for question, cands in zip(question_set.questions, candidates, strict=True):
        relevant = set(question.relevant_pages)

        def score(chunks: Sequence[Chunk], n: int | None, relevant: set[str] = relevant) -> None:
            values = compute_metrics(ranked_ids(chunks[:final_k], "page"), relevant, ks)
            if values is not None:
                scores[n].append(values)

        score(cands, None)
        for n in rerank_ns:
            start = clock()
            reranked = list(reranker(question.question, cands[:n]))
            latencies[n].append((clock() - start) * 1000.0)
            sent[n].append(len(cands[:n]))
            score(reranked, n)

    def means(n: int | None) -> dict[str, float]:
        rows = scores[n]
        return {m: float(np.mean([r[m] for r in rows])) if rows else 0.0 for m in names}

    points = [
        CurvePoint(
            rerank_n=None,
            questions=len(scores[None]),
            mean_candidates=float(np.mean([len(c) for c in candidates])),
            means=means(None),
        )
    ]
    for n in rerank_ns:
        ms = np.asarray(latencies[n])
        total_sent = sum(sent[n])
        points.append(
            CurvePoint(
                rerank_n=n,
                questions=len(scores[n]),
                mean_candidates=float(np.mean(sent[n])),
                p50_ms=float(np.percentile(ms, 50)),
                p95_ms=float(np.percentile(ms, 95)),
                mean_ms=float(ms.mean()),
                ms_per_candidate=float(ms.sum() / total_sent) if total_sent else None,
                means=means(n),
            )
        )
    return RerankCurve(
        kb=kb,
        created_at=(now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        question_set=QuestionSetInfo(
            path=question_set.path, sha256=question_set.sha256, count=len(question_set.questions)
        ),
        final_k=final_k,
        ks=list(ks),
        rerank_ns=list(rerank_ns),
        warmup=warmup,
        provenance=provenance if provenance is not None else collect_provenance(),
        points=points,
    )


def _cell(value: float | None, fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


def render_rerank_curve(curve: RerankCurve) -> str:
    """The Markdown table of the curve (one row per rerank_n, plus the fusion order)."""
    k = max(curve.ks)
    columns = [*(f"hit@{x}" for x in curve.ks), f"mrr@{k}", f"ndcg@{k}"]
    models = ", ".join(f"{key} `{v}`" for key, v in curve.provenance.models.items()) or "-"
    qs = curve.question_set
    lines = [
        f"# Rerank latency curve ({curve.kb}, {curve.created_at})",
        "",
        f"Question set `{qs.path}` ({qs.count} questions, sha256 `{qs.sha256}`); models: {models}.",
        f"Latency is the reranker call only ({curve.warmup} warm-up call not measured); metrics "
        f"are page level on the reranked top {curve.final_k}.",
        "",
        "| rerank_n | candidates | P50 ms | P95 ms | mean ms | ms per candidate | "
        + " | ".join(columns)
        + " |",
        "|---" * (6 + len(columns)) + "|",
    ]
    for p in curve.points:
        label = "fusion (no rerank)" if p.rerank_n is None else str(p.rerank_n)
        cells = [
            label,
            f"{p.mean_candidates:.1f}",
            _cell(p.p50_ms, ".1f"),
            _cell(p.p95_ms, ".1f"),
            _cell(p.mean_ms, ".1f"),
            _cell(p.ms_per_candidate, ".2f"),
            *(f"{p.means[c]:.3f}" for c in columns),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_rerank_curve(curve: RerankCurve, out_dir: str | Path) -> Path:
    """Write `rerank_curve.json` and `rerank_curve.md` into `out_dir` (e.g. a run directory)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rerank_curve.json").write_text(
        curve.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "rerank_curve.md").write_text(render_rerank_curve(curve), encoding="utf-8", newline="\n")
    return out
