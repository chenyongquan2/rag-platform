"""Spec P0-07 AC2-AC3: reciprocal rank fusion and its tie-break (design s9-3, P0-10 decision)."""

from __future__ import annotations

import itertools

import pytest

from rag.retrieve.fusion import rrf_fuse
from rag.types import RouteHit


def _hits(*ids: str, scores: list[float] | None = None) -> list[RouteHit]:
    return [
        RouteHit(
            chunk_id=chunk_id,
            doc_id=chunk_id.split("#")[0],
            rank=rank,
            score=scores[rank - 1] if scores else None,
        )
        for rank, chunk_id in enumerate(ids, start=1)
    ]


def test_rrf_hand_computed() -> None:
    routes = {
        "bm25": _hits("a#0", "b#0", "c#0", scores=[12.5, 9.0, 3.25]),
        "vector": _hits("b#0", "d#0", "a#0", scores=[0.91, 0.88, 0.80]),
    }
    fused = rrf_fuse(routes, k=60)

    # a: 1/61 + 1/63, b: 1/62 + 1/61, c: 1/63, d: 1/62.
    expected = {
        "a#0": 1 / 61 + 1 / 63,
        "b#0": 1 / 62 + 1 / 61,
        "c#0": 1 / 63,
        "d#0": 1 / 62,
    }
    assert [h.chunk_id for h in fused] == ["b#0", "a#0", "d#0", "c#0"]
    for hit in fused:
        assert hit.score == pytest.approx(expected[hit.chunk_id], abs=1e-15)
        assert hit.doc_id == hit.chunk_id.split("#")[0]
    by_id = {h.chunk_id: h for h in fused}
    assert by_id["a#0"].route_ranks == {"bm25": 1, "vector": 3}
    assert by_id["a#0"].route_scores == {"bm25": 12.5, "vector": 0.80}
    assert by_id["d#0"].route_ranks == {"vector": 2}
    assert by_id["c#0"].route_scores == {"bm25": 3.25}

    # k is a parameter: with k=0 the first rank is worth 1.
    (only,) = rrf_fuse({"bm25": _hits("a#0")}, k=0)
    assert only.score == 1.0
    assert rrf_fuse({}, k=60) == []
    assert rrf_fuse({"bm25": [], "vector": []}, k=60) == []
    with pytest.raises(ValueError, match="k"):
        rrf_fuse(routes, k=-1)


def test_rrf_ties_are_deterministic() -> None:
    # x#1 is BM25 rank 1 only, y#1 is vector rank 1 only: equal scores 1/61, equal best rank
    # 1, so chunk_id decides (x before y). p and q are both (rank 2, rank 3) in some order:
    # equal exact scores, equal best rank 2 -> chunk_id again. z#9 has a lower score.
    routes = {
        "bm25": _hits("y#2", "p#0", "q#0", "z#9"),
        "vector": _hits("x#1", "q#0", "p#0"),
    }
    routes["bm25"][0] = RouteHit(chunk_id="x#1", doc_id="x", rank=1)
    routes["vector"][0] = RouteHit(chunk_id="y#1", doc_id="y", rank=1)
    expected = ["p#0", "q#0", "x#1", "y#1", "z#9"]
    assert [h.chunk_id for h in rrf_fuse(routes, k=60)] == expected

    # Same score, different best single-route rank: the better rank goes first although its
    # chunk_id sorts later. With k=0: b = 1/2 + 1/6 and a = 1/3 + 1/3, both exactly 2/3 (in
    # floats 0.5 + 0.1666... and 0.333... + 0.333... may differ; the fusion compares exactly).
    tie = {
        "bm25": [
            RouteHit(chunk_id="b#0", doc_id="b", rank=2),
            RouteHit(chunk_id="a#0", doc_id="a", rank=3),
        ],
        "vector": [
            RouteHit(chunk_id="a#0", doc_id="a", rank=3),
            RouteHit(chunk_id="b#0", doc_id="b", rank=6),
        ],
    }
    fused = rrf_fuse(tie, k=0)
    assert fused[0].score == fused[1].score  # 1/2 + 1/6 == 1/3 + 1/3 exactly
    assert [h.chunk_id for h in fused] == ["b#0", "a#0"]  # best rank 2 beats best rank 3

    # The order of routes and of hits inside the mapping does not change the result.
    for names in itertools.permutations(routes):
        shuffled = {name: list(reversed(routes[name])) for name in names}
        assert [h.chunk_id for h in rrf_fuse(shuffled, k=60)] == expected
