"""Reciprocal rank fusion of the per-route lists (design s9-2, s9-3).

score(chunk) = sum over the routes that found it of w_route / (k + rank), ranks counted from 1.
Weights are equal by default (s9-3: weights only matter once fewer candidates go to the reranker
than the routes return together); `RetrievalConfig.rrf_weights` can change them (spec P0-13).
A weight is read as an exact decimal fraction (0.5 is 1/2), so ties stay exact. Only ranks are
used, never the routes' own scores: BM25
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


def _exact_weights(
    routes: Mapping[str, Sequence[RouteHit]], weights: Mapping[str, float] | None
) -> dict[str, Fraction]:
    if weights is None:
        return {name: Fraction(1) for name in routes}
    missing = [name for name in routes if name not in weights]
    if missing:
        raise ValueError(f"rrf weight missing for route(s) {missing}")
    exact: dict[str, Fraction] = {}
    for name in routes:
        weight = float(weights[name])
        if weight < 0:
            raise ValueError(f"rrf weight of route {name!r} must be >= 0, got {weight}")
        exact[name] = Fraction(str(weight))
    return exact


def rrf_fuse(
    routes: Mapping[str, Sequence[RouteHit]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse the routes' ranked lists; best first, ties broken as described in the module doc.

    A route's ranks come from `RouteHit.rank`; a chunk listed twice in one route counts once,
    at its better rank. `weights` maps every route to its weight (>= 0; None = all 1). A route
    with weight 0 adds nothing to the score, but its chunks are still listed (at score 0, after
    the others) with their route ranks.
    """
    if k < 0:
        raise ValueError(f"rrf k must be >= 0, got {k}")
    route_weights = _exact_weights(routes, weights)
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
        chunk_id: sum(
            (route_weights[name] / (k + rank) for name, rank in per_route.items()), Fraction(0)
        )
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
