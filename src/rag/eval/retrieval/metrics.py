"""Retrieval metrics on one ranked list (design s14-1).

Each function takes the ranked ids at one label level (page level: doc ids, s5-4; chunk level:
chunk ids) in list order, and the set of relevant ids. Relevance is binary. An id that appears
again further down the list (another chunk of an already counted page) adds nothing, so no
metric exceeds 1. The ideal DCG of nDCG@k is capped at k items: with more relevant items than
k a perfect list still scores 1 (LlamaIndex's RetrieverEvaluator does not cap it).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from typing import Literal

from rag.types import Chunk, RouteHit

Level = Literal["page", "chunk"]
LEVELS: tuple[Level, ...] = ("page", "chunk")


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _relevant_ranks(ranked: Sequence[str], relevant: Collection[str], k: int) -> list[int]:
    """1-based ranks within the top k where a relevant id appears for the first time."""
    _check_k(k)
    seen: set[str] = set()
    ranks = []
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            seen.add(item)
            ranks.append(rank)
    return ranks


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 when any relevant id is in the top k, else 0."""
    return 1.0 if _relevant_ranks(ranked, relevant, k) else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Reciprocal rank of the first relevant id within the top k (0 when none)."""
    ranks = _relevant_ranks(ranked, relevant, k)
    return 1.0 / ranks[0] if ranks else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant ids found in the top k."""
    if not relevant:
        raise ValueError("recall needs at least one relevant id")
    return len(_relevant_ranks(ranked, relevant, k)) / len(set(relevant))


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-gain nDCG@k with the ideal DCG over `min(k, number of relevant ids)` items."""
    if not relevant:
        raise ValueError("nDCG needs at least one relevant id")
    dcg = sum(1.0 / math.log2(rank + 1) for rank in _relevant_ranks(ranked, relevant, k))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(set(relevant))) + 1))
    return dcg / ideal


def metric_names(ks: Sequence[int]) -> list[str]:
    """`hit@k` for each k, `mrr@max(ks)`, then `ndcg@k` and `recall@k` for each k."""
    if not ks:
        raise ValueError("ks must not be empty")
    for k in ks:
        _check_k(k)
    return [
        *(f"hit@{k}" for k in ks),
        f"mrr@{max(ks)}",
        *(f"ndcg@{k}" for k in ks),
        *(f"recall@{k}" for k in ks),
    ]


def compute_metrics(
    ranked: Sequence[str], relevant: Collection[str], ks: Sequence[int]
) -> dict[str, float] | None:
    """Every metric of `metric_names(ks)`; `None` when the question has no relevant ids."""
    names = metric_names(ks)
    if not relevant:
        return None
    values = [
        *(hit_at_k(ranked, relevant, k) for k in ks),
        mrr_at_k(ranked, relevant, max(ks)),
        *(ndcg_at_k(ranked, relevant, k) for k in ks),
        *(recall_at_k(ranked, relevant, k) for k in ks),
    ]
    return dict(zip(names, values, strict=True))


def is_binary_metric(name: str) -> bool:
    """Hit@k is 0 or 1 per question (exact McNemar); the others are continuous (permutation)."""
    return name.startswith("hit@")


def first_relevant_rank(ranked: Sequence[str], relevant: Collection[str]) -> int | None:
    """1-based rank of the first relevant id in the whole list, or None."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return rank
    return None


def ranked_ids(hits: Iterable[Chunk | RouteHit], level: Level) -> list[str]:
    """Ids at `level` in ranked order.

    Chunks of the final list keep their list order (scores are never used, so equal scores
    cannot reorder anything); route hits are ordered by their own `rank`, stable on equal ranks.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    items = list(hits)
    if items and all(isinstance(h, RouteHit) for h in items):
        items.sort(key=lambda h: h.rank)  # type: ignore[union-attr]
    return [h.doc_id if level == "page" else h.chunk_id for h in items]
