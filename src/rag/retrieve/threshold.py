"""Drop chunks whose rerank score is below the threshold; report "not found" (design s10-2, s9-1).

`RerankThreshold` judges chunks by their rerank score (`metadata["rerank_score"]`, set by
`BatchedTimeoutRerank`). LlamaIndex's `SimilarityPostprocessor` is not used for this: it drops
nodes whose score is None or below the cutoff, so after a rerank timeout it would drop every
unscored chunk (their `score` is still a fusion score of about 0.03), and it cannot report the
best chunks when all are below (design s9-1).

- Every chunk has a rerank score (the normal case): chunks below the threshold are dropped; a
  score equal to the threshold passes. When none is left - or there were no candidates - the
  result is empty and the trace says `not_found`, with the best `keep_best` (3) chunks by rerank
  score kept apart in `below_threshold`.
- After a rerank timeout (the trace says `rerank_timeout`, or some chunk has no rerank score)
  nothing is dropped by the threshold (s9-1, s16-8): the unscored chunks may be the relevant
  ones, and the threshold cannot say "not found" about a list it has not fully judged. The order
  becomes: scored chunks at or above the threshold, then the unscored ones in fusion order, then
  the scored ones below the threshold, so chunks the reranker judged irrelevant do not take the
  places of chunks it never saw. `ThresholdDecision.applied` is False.

The decision (`ThresholdDecision`) goes into the trace for business records.
"""

from __future__ import annotations

from typing import ClassVar

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import RERANK_SCORE
from rag.types import ThresholdDecision


def rerank_score(item: NodeWithScore) -> float | None:
    value = item.node.metadata.get(RERANK_SCORE)
    return float(value) if value is not None else None


class RerankThreshold(BaseNodePostprocessor):
    """Drop chunks below `threshold` after a complete rerank; after a partial one, move them
    behind the unscored chunks instead (see the module doc)."""

    timing_name: ClassVar[str] = "threshold"

    threshold: float = Field(ge=0.0, le=1.0, description="Rerank score a chunk needs (s10-2).")
    keep_best: int = Field(
        default=3, ge=0, description="Chunks kept in `below_threshold` when none passes."
    )

    @classmethod
    def class_name(cls) -> str:
        return "RerankThreshold"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        query = query_bundle if isinstance(query_bundle, HybridQuery) else None
        scores = [rerank_score(item) for item in nodes]
        judged = [s for s in scores if s is not None]
        top = max(judged) if judged else None
        passed = (top >= self.threshold) if top is not None else None
        partial = len(judged) < len(nodes) or (query is not None and query.trace.rerank_timeout)

        if partial:
            above = [
                n
                for n, s in zip(nodes, scores, strict=True)
                if s is not None and s >= self.threshold
            ]
            unscored = [n for n, s in zip(nodes, scores, strict=True) if s is None]
            below = [
                n
                for n, s in zip(nodes, scores, strict=True)
                if s is not None and s < self.threshold
            ]
            kept = above + unscored + below
            decision = ThresholdDecision(
                threshold=self.threshold,
                top_score=top,
                applied=False,
                passed=passed,
                kept=len(kept),
                dropped=0,
            )
            if query is not None:
                query.trace.threshold = decision
                query.trace.not_found = False
            return kept

        kept = [
            n for n, s in zip(nodes, scores, strict=True) if s is not None and s >= self.threshold
        ]
        decision = ThresholdDecision(
            threshold=self.threshold,
            top_score=top,
            applied=bool(judged),
            passed=passed,
            kept=len(kept),
            dropped=len(nodes) - len(kept),
        )
        if query is not None:
            trace = query.trace
            trace.threshold = decision
            trace.not_found = not kept
            if not kept:
                ranked = sorted(range(len(nodes)), key=lambda i: (-(scores[i] or 0.0), i))
                trace.below_threshold = [nodes[i] for i in ranked[: self.keep_best]]
        return kept
