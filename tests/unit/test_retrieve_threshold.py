"""Spec P0-08 AC8-AC9: RerankThreshold (design s9-1, s10-2, s16-8)."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from rag.retrieve.deadline import Deadline
from rag.retrieve.query import HybridQuery
from rag.retrieve.threshold import RerankThreshold
from rag.types import ThresholdDecision


def _node(name: str, rerank: float | None, fusion: float = 0.03) -> NodeWithScore:
    meta = {"doc_id": f"hf_docs:{name}.md", "kb": "hf_docs"}
    if rerank is not None:
        meta["rerank_score"] = rerank
    return NodeWithScore(
        node=TextNode(id_=f"hf_docs:{name}.md#0", text=name, metadata=meta), score=fusion
    )


def _query(*, timed_out: bool = False) -> HybridQuery:
    query = HybridQuery("q", acl_groups=("public",), deadline=Deadline.after_ms(5000))
    query.trace.rerank_timeout = timed_out
    return query


def _ids(nodes: list[NodeWithScore]) -> list[str]:
    return [n.node.node_id.split(":")[1].split(".")[0] for n in nodes]


def test_threshold_drops_scored_below() -> None:
    nodes = [_node("a", 0.9), _node("b", 0.3), _node("c", 0.29), _node("d", 0.01)]
    query = _query()
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=query)

    assert _ids(out) == ["a", "b"]  # equal to the threshold passes
    assert query.trace.not_found is False
    assert query.trace.below_threshold == []
    assert query.trace.threshold == ThresholdDecision(
        threshold=0.3, top_score=0.9, applied=True, passed=True, kept=2, dropped=2
    )


def test_threshold_after_timeout_keeps_unscored() -> None:
    # After a rerank timeout nothing is dropped (s9-1, s16-8): scored chunks at or above the
    # threshold first, then the unscored ones in fusion order, then the scored ones below.
    nodes = [_node("a", 0.8), _node("e", 0.3), _node("b", 0.1), _node("c", None), _node("d", None)]
    query = _query(timed_out=True)
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=query)
    assert _ids(out) == ["a", "e", "c", "d", "b"]
    assert query.trace.not_found is False
    assert query.trace.threshold == ThresholdDecision(
        threshold=0.3, top_score=0.8, applied=False, passed=True, kept=5, dropped=0
    )

    # Every scored chunk below the threshold: still nothing dropped, never "not found".
    nodes = [_node("a", 0.2), _node("b", None)]
    query = _query(timed_out=True)
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=query)
    assert _ids(out) == ["b", "a"]
    assert query.trace.not_found is False
    assert query.trace.threshold is not None and query.trace.threshold.passed is False

    # Nothing scored at all (no slot before the deadline): everything kept in fusion order.
    nodes = [_node("a", None), _node("b", None)]
    query = _query(timed_out=True)
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=query)
    assert _ids(out) == ["a", "b"]
    assert query.trace.threshold == ThresholdDecision(
        threshold=0.3, top_score=None, applied=False, passed=None, kept=2, dropped=0
    )

    # Without a HybridQuery (no trace), unscored chunks alone mark a partial rerank.
    from llama_index.core.schema import QueryBundle

    nodes = [_node("a", 0.9), _node("b", 0.1), _node("c", None)]
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=QueryBundle("q"))
    assert _ids(out) == ["a", "c", "b"]


def test_threshold_all_below_returns_not_found_with_top3() -> None:
    nodes = [_node(n, s) for n, s in [("a", 0.2), ("b", 0.15), ("c", 0.1), ("d", 0.05)]]
    query = _query()
    out = RerankThreshold(threshold=0.3).postprocess_nodes(nodes, query_bundle=query)

    assert out == []
    assert query.trace.not_found is True
    assert _ids(query.trace.below_threshold) == ["a", "b", "c"]
    assert query.trace.threshold == ThresholdDecision(
        threshold=0.3, top_score=0.2, applied=True, passed=False, kept=0, dropped=4
    )

    # The best 3 by rerank score, whatever order they came in.
    nodes = [_node("x", 0.05), _node("y", 0.25), _node("z", 0.1), _node("w", 0.2)]
    query = _query()
    RerankThreshold(threshold=0.3, keep_best=2).postprocess_nodes(nodes, query_bundle=query)
    assert _ids(query.trace.below_threshold) == ["y", "w"]

    # No candidates at all is also "not found", with nothing to show.
    query = _query()
    assert RerankThreshold(threshold=0.3).postprocess_nodes([], query_bundle=query) == []
    assert query.trace.not_found is True
    assert query.trace.below_threshold == []
