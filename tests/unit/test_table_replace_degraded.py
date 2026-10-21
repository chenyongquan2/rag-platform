"""Spec P6-01 review item 4: a table fetch that failed is a degraded result, not a silent one."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from rag.cache.rules import search_cacheable
from rag.retrieve.query import HybridQuery
from rag.retrieve.search import HybridSearch
from rag.retrieve.table_groups import TableSummaryReplace
from rag.types import ErrorCategory

KB = "pdf_demo"
DOC = "pdf_demo:p.pdf#p3"
T0 = f"{DOC}#t0"


class StubRetriever:
    """What `HybridSearch` needs of the retriever: a knowledge base name and the candidates."""

    kb = KB

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes

    def retrieve(self, query: HybridQuery) -> list[NodeWithScore]:
        return list(self.nodes)

    def close(self) -> None:
        pass


def summary_hit() -> NodeWithScore:
    node = TextNode(
        id_=f"{DOC}#9",
        text="A summary written by a model.",
        metadata={"doc_id": DOC, "chunk_kind": "table_summary", "table_id": T0},
    )
    return NodeWithScore(node=node, score=0.8)


def broken_fetch(table_ids: Any, acl_groups: Any, timeout_s: Any = None) -> Any:
    raise TimeoutError("elasticsearch is slow")


def test_a_failed_table_fetch_makes_the_result_degraded_and_uncacheable() -> None:
    search = HybridSearch(
        StubRetriever([summary_hit()]),  # type: ignore[arg-type]
        acl_groups=["public"],
        deadline_ms=1000,
        postprocessors=[TableSummaryReplace(broken_fetch)],
    )
    result = search.retrieve("which model is best", kb=KB, top_k=5)
    assert [c.chunk_id for c in result.chunks] == [f"{DOC}#9"]  # the summary stays
    assert result.degraded is True
    assert ErrorCategory.DEGRADED in result.error_categories
    verdict = search_cacheable(result)
    assert verdict.ok is False and "degraded" in verdict.reason


def test_a_good_fetch_leaves_the_result_not_degraded() -> None:
    table = TextNode(
        id_=f"{DOC}#4",
        text="| a | b |",
        metadata={"doc_id": DOC, "chunk_kind": "table", "table_id": T0},
    )
    search = HybridSearch(
        StubRetriever([summary_hit()]),  # type: ignore[arg-type]
        acl_groups=["public"],
        deadline_ms=1000,
        postprocessors=[TableSummaryReplace(lambda ids, acl, timeout_s=None: {T0: [table]})],
    )
    result = search.retrieve("which model is best", kb=KB, top_k=5)
    assert [c.chunk_id for c in result.chunks] == [f"{DOC}#4"]
    assert result.degraded is False and search_cacheable(result).ok is True
