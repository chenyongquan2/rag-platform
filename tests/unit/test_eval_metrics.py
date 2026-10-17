"""Spec P0-10 AC4-AC7: retrieval metrics on hand-computed examples (design s14-1)."""

from __future__ import annotations

import math

import pytest

from rag.eval.retrieval.metrics import (
    compute_metrics,
    hit_at_k,
    metric_names,
    mrr_at_k,
    ndcg_at_k,
    ranked_ids,
    recall_at_k,
)
from rag.types import Chunk, RouteHit

L2, L3, L4, L5 = (1 / math.log2(n) for n in (2, 3, 4, 5))  # discounts of ranks 1 to 4 (L2 = 1)


def _chunk(chunk_id: str, doc_id: str, score: float | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, kb="hf_docs", text="t", score=score)


def test_metrics_hand_computed() -> None:
    ranked = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b", "c"}
    assert [hit_at_k(ranked, relevant, k) for k in (1, 2, 5)] == [0.0, 1.0, 1.0]
    assert mrr_at_k(ranked, relevant, 1) == 0.0
    assert mrr_at_k(ranked, relevant, 5) == 0.5
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)
    # DCG@5 = 1/log2(3) + 1/log2(5); ideal = three relevant items at ranks 1 to 3.
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx((L3 + L5) / (1 + L3 + L4))
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(L3 / (1 + L3))
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(0.38685, abs=1e-5)

    metrics = compute_metrics(ranked, relevant, ks=(1, 5))
    assert metrics is not None
    assert list(metrics) == metric_names((1, 5))
    assert list(metrics) == ["hit@1", "hit@5", "mrr@5", "ndcg@1", "ndcg@5", "recall@1", "recall@5"]
    assert metrics["mrr@5"] == 0.5
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        hit_at_k(ranked, relevant, 0)


def test_ndcg_ideal_dcg_capped_at_k() -> None:
    relevant = {"a", "b", "c"}  # more relevant items than k
    perfect = ["a", "b", "x", "c"]
    assert ndcg_at_k(perfect, relevant, 2) == 1.0
    # LlamaIndex's RetrieverEvaluator divides by the ideal DCG over all three relevant items,
    # so the same list could never reach 1 (design s14-1).
    uncapped = (1 + L3) / (1 + L3 + L4)
    assert uncapped == pytest.approx(0.76536, abs=1e-5)
    assert ndcg_at_k(perfect, relevant, 2) != pytest.approx(uncapped)
    # With k >= the number of relevant items the cap changes nothing.
    assert ndcg_at_k(perfect, relevant, 4) == pytest.approx((1 + L3 + L5) / (1 + L3 + L4))


def test_repeated_page_counts_once_and_ties_keep_list_order() -> None:
    # Three chunks of the same relevant page: page-level gain is counted at its first rank only.
    assert ndcg_at_k(["a", "a", "a"], {"a"}, 3) == 1.0
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 0.5
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, 3) == pytest.approx((1 + L4) / (1 + L3))
    assert hit_at_k(["a", "a"], {"a"}, 2) == 1.0

    # The final list is scored in list order; scores (equal or not) never reorder it.
    chunks = [
        _chunk("c1", "hf_docs:x.md", 0.5),
        _chunk("c2", "hf_docs:a.md", 0.5),
        _chunk("c3", "hf_docs:a.md", 0.9),
    ]
    assert ranked_ids(chunks, "page") == ["hf_docs:x.md", "hf_docs:a.md", "hf_docs:a.md"]
    assert ranked_ids(chunks, "chunk") == ["c1", "c2", "c3"]
    assert mrr_at_k(ranked_ids(chunks, "page"), {"hf_docs:a.md"}, 10) == 0.5
    # A route list is ordered by its own rank; equal ranks keep list order.
    route = [
        RouteHit(chunk_id="r2", doc_id="hf_docs:b.md", rank=2),
        RouteHit(chunk_id="r1", doc_id="hf_docs:a.md", rank=1),
        RouteHit(chunk_id="r3", doc_id="hf_docs:c.md", rank=2),
    ]
    assert ranked_ids(route, "chunk") == ["r1", "r2", "r3"]
    with pytest.raises(ValueError):
        ranked_ids(chunks, "doc")


def test_empty_results_and_unlabelled_questions() -> None:
    assert compute_metrics([], {"a"}, ks=(1, 10)) == dict.fromkeys(metric_names((1, 10)), 0.0)
    assert ndcg_at_k([], {"a"}, 10) == 0.0
    # No relevant items at this level: the metrics are undefined, not 0.
    assert compute_metrics(["a", "b"], set(), ks=(1, 10)) is None
