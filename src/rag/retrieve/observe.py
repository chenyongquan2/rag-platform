"""Metrics of one finished search, shared by `search_docs` and the ask_docs graph (s15-1)."""

from __future__ import annotations

from rag.observe import attributes as attr
from rag.observe import get_observability
from rag.types import RetrievalResult

# Steps of `RetrievalResult.timings_ms` that are not a step of their own.
_NOT_STEPS = {"total"}


def observe_retrieval(result: RetrievalResult, kb: str, tool: str) -> None:
    """Step durations from the chain's own timings (P95 per step) and the quality signals: not
    found, best rerank score."""
    metrics = get_observability().metrics
    for step, ms in result.timings_ms.items():
        if step not in _NOT_STEPS:
            metrics.record_step(step, ms / 1000.0, outcome=attr.OUTCOME_OK, kb=kb)
    scores = [h.score for h in result.rerank if h.score is not None]
    metrics.record_retrieval(
        kb=kb,
        tool=tool,
        not_found=result.not_found,
        top_rerank_score=max(scores) if scores else None,
    )
