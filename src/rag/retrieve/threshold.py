"""Drop chunks whose rerank score is below the threshold; report "not found" (design s10-2, s9-1).

`RerankThreshold` judges only chunks that have a rerank score (`metadata["rerank_score"]`,
set by `BatchedTimeoutRerank`). Chunks without one - left unscored by a rerank timeout - are
kept in place: their `score` is still a fusion score of about 0.03 and says nothing about the
threshold. LlamaIndex's `SimilarityPostprocessor` is not used for this: it drops nodes whose
score is None or below the cutoff, so after a timeout it would drop every unscored chunk, and
it cannot report the best chunks when all are below (design s9-1).

When no chunk is left - every scored chunk below the threshold and none unscored, or no
candidates at all - the result is empty and the trace says `not_found`, with the best
`keep_best` (3) chunks by rerank score kept apart in `below_threshold`. A score equal to the
threshold passes. The decision (`ThresholdDecision`) goes into the trace for business records.
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
    """Keep chunks with rerank score >= `threshold` and every chunk without a rerank score."""

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
        scores = [rerank_score(item) for item in nodes]
        judged = [s for s in scores if s is not None]
        kept = [
            item for item, s in zip(nodes, scores, strict=True) if s is None or s >= self.threshold
        ]
        top = max(judged) if judged else None
        decision = ThresholdDecision(
            threshold=self.threshold,
            top_score=top,
            applied=bool(judged),
            passed=(top >= self.threshold) if top is not None else None,
            kept=len(kept),
            dropped=len(nodes) - len(kept),
        )
        if isinstance(query_bundle, HybridQuery):
            trace = query_bundle.trace
            trace.threshold = decision
            trace.not_found = not kept
            if not kept:
                scored = [(s, i) for i, s in enumerate(scores) if s is not None]
                scored.sort(key=lambda pair: (-pair[0], pair[1]))
                trace.below_threshold = [nodes[i] for _, i in scored[: self.keep_best]]
        return kept
