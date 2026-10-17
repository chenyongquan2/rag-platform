"""Reciprocal rank fusion of the per-route lists (design s9-2, s9-3).

score(chunk) = sum over the routes that found it of 1 / (k + rank), ranks counted from 1, all
routes weighted equally (s9-3: weights only matter once fewer candidates go to the reranker
than the routes return together). Only ranks are used, never the routes' own scores: BM25
scores have no upper bound and cosine similarities lie in [0, 1], so they cannot be added.

Ties are frequent (a chunk only BM25 found at rank 1 and one only the vector route found at
rank 1 score the same), and the eval harness scores the list in its order (P0-10 decision), so
the order of equal scores is fixed: higher exact score, then the better best single-route rank,
then chunk_id ascending. Scores are compared as exact fractions; floating-point sums of the
same terms in another order could differ in the last bit and flip a tie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from rag.types import RouteHit

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    doc_id: str
    score: float
    route_ranks: dict[str, int] = field(default_factory=dict)
    """Route name -> this chunk's rank in that route (only routes that found it)."""
    route_scores: dict[str, float | None] = field(default_factory=dict)
    """Route name -> the route's own score (BM25 score, kNN similarity), for traces."""


def rrf_fuse(routes: Mapping[str, Sequence[RouteHit]], *, k: int = DEFAULT_RRF_K) -> list[FusedHit]:
    """Fuse the routes' ranked lists; best first, ties broken as described in the module doc.

    A route's ranks come from `RouteHit.rank`; a chunk listed twice in one route counts once,
    at its better rank.
    """
    if k < 0:
        raise ValueError(f"rrf k must be >= 0, got {k}")
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float | None]] = {}
    doc_ids: dict[str, str] = {}
    for name in sorted(routes):
        for hit in routes[name]:
            per_route = ranks.setdefault(hit.chunk_id, {})
            if name in per_route and per_route[name] <= hit.rank:
                continue
            per_route[name] = hit.rank
            scores.setdefault(hit.chunk_id, {})[name] = hit.score
            doc_ids.setdefault(hit.chunk_id, hit.doc_id)

    exact = {
        chunk_id: sum((Fraction(1, k + rank) for rank in per_route.values()), Fraction(0))
        for chunk_id, per_route in ranks.items()
    }
    order = sorted(
        ranks,
        key=lambda chunk_id: (-exact[chunk_id], min(ranks[chunk_id].values()), chunk_id),
    )
    return [
        FusedHit(
            chunk_id=chunk_id,
            doc_id=doc_ids[chunk_id],
            score=float(exact[chunk_id]),
            route_ranks=dict(sorted(ranks[chunk_id].items())),
            route_scores=dict(sorted(scores[chunk_id].items())),
        )
        for chunk_id in order
    ]
