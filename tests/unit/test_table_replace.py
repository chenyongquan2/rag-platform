"""Spec P6-01 AC10, AC11: a hit on a table summary is replaced by the table's own chunk(s)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.hybrid import SOURCE_FIELDS
from rag.retrieve.page_groups import PageKeyDedupe
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline, search_config_version
from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import BatchedTimeoutRerank, RerankRunner
from rag.retrieve.search import HybridSearch, make_hybrid_search, node_to_chunk
from rag.retrieve.table_groups import EsTableFetch, TableSummaryReplace
from rag.retrieve.threshold import RerankThreshold
from rag.testing import FakeEmbedding
from rag.testing.rerank import FakeCrossEncoder
from rag.testing.services import FakeElasticsearch

KB = "pdf_demo"
INDEX = "kb_pdf_demo_v1"
ALIAS = "kb_pdf_demo"
DOC = "pdf_demo:p.pdf#p3"


def chunk(chunk_id: str, kind: str, score: float, *, table_id: str | None = None, **meta: Any):
    metadata: dict[str, Any] = {"doc_id": DOC, "chunk_kind": kind, **meta}
    if table_id:
        metadata["table_id"] = table_id
    node = TextNode(id_=chunk_id, text=f"text of {chunk_id}", metadata=metadata)
    return NodeWithScore(node=node, score=score)


def table_node(chunk_id: str, table_id: str) -> TextNode:
    return TextNode(
        id_=chunk_id,
        text=f"| the table | {chunk_id} |",
        metadata={"doc_id": DOC, "chunk_kind": "table", "table_id": table_id, "page": 3},
    )


class Fetch:
    def __init__(self, tables: dict[str, list[TextNode]]) -> None:
        self.tables = tables
        self.calls: list[tuple[list[str], tuple[str, ...]]] = []

    def __call__(self, table_ids, acl_groups, timeout_s=None):
        self.calls.append((list(table_ids), tuple(acl_groups)))
        return {t: self.tables[t] for t in table_ids if t in self.tables}


def run(nodes, fetch, **kwargs):
    query = HybridQuery("which model has the lowest error", acl_groups=("public", "team"))
    step = TableSummaryReplace(fetch=fetch, **kwargs)
    return step.postprocess_nodes(nodes, query_bundle=query), query


def ids(nodes) -> list[str]:
    return [n.node.node_id for n in nodes]


T0, T1 = f"{DOC}#t0", f"{DOC}#t1"
FETCH = {
    T0: [table_node(f"{DOC}#4", T0)],
    T1: [table_node(f"{DOC}#5", T1), table_node(f"{DOC}#6", T1)],
}


def test_table_replace_puts_the_table_chunk_where_the_summary_hit_was() -> None:
    nodes = [
        chunk(f"{DOC}#0", "text", 0.9),
        chunk(f"{DOC}#9", "table_summary", 0.8, table_id=T0),
        chunk(f"{DOC}#1", "text", 0.7),
    ]
    fetch = Fetch(FETCH)
    out, query = run(nodes, fetch)
    assert ids(out) == [f"{DOC}#0", f"{DOC}#4", f"{DOC}#1"]
    replaced = out[1]
    assert replaced.score == 0.8  # the summary's rank and score
    assert replaced.node.get_content() == f"| the table | {DOC}#4 |"  # the table, not the summary
    assert replaced.node.metadata["matched_table_summary"] == f"{DOC}#9"
    assert fetch.calls == [([T0], ("public", "team"))]  # the caller's ACL groups go along
    assert query.trace.table_replacements == {f"{DOC}#9": [f"{DOC}#4"]}


def test_table_replace_a_table_in_parts_returns_every_part_in_order() -> None:
    nodes = [chunk(f"{DOC}#9", "table_summary", 0.8, table_id=T1)]
    out, _ = run(nodes, Fetch(FETCH))
    assert ids(out) == [f"{DOC}#5", f"{DOC}#6"]
    assert [n.score for n in out] == [0.8, 0.8]


def test_table_replace_limits_the_parts_of_one_table() -> None:
    nodes = [chunk(f"{DOC}#9", "table_summary", 0.8, table_id=T1)]
    out, query = run(nodes, Fetch(FETCH), max_parts=1)
    assert ids(out) == [f"{DOC}#5"]
    assert query.trace.table_replacements == {f"{DOC}#9": [f"{DOC}#5"]}


def test_table_replace_removes_duplicates_and_keeps_the_better_rank() -> None:
    # the table chunk itself was also found, later: only the first place stays
    nodes = [
        chunk(f"{DOC}#9", "table_summary", 0.9, table_id=T0),
        chunk(f"{DOC}#4", "table", 0.5, table_id=T0),
    ]
    out, _ = run(nodes, Fetch(FETCH))
    assert ids(out) == [f"{DOC}#4"] and out[0].score == 0.9
    # the table chunk was found first: the summary adds nothing
    nodes = [
        chunk(f"{DOC}#4", "table", 0.9, table_id=T0),
        chunk(f"{DOC}#9", "table_summary", 0.5, table_id=T0),
    ]
    out, _ = run(nodes, Fetch(FETCH))
    assert ids(out) == [f"{DOC}#4"] and out[0].score == 0.9
    assert "matched_table_summary" not in out[0].node.metadata


def test_table_replace_two_summaries_of_one_table_give_one_table() -> None:
    nodes = [
        chunk(f"{DOC}#8", "table_summary", 0.9, table_id=T0),
        chunk(f"{DOC}#9", "table_summary", 0.8, table_id=T0),
    ]
    out, _ = run(nodes, Fetch(FETCH))
    assert ids(out) == [f"{DOC}#4"]


def test_table_replace_without_summary_hits_changes_nothing_and_fetches_nothing() -> None:
    nodes = [chunk(f"{DOC}#0", "text", 0.9), chunk(f"{DOC}#4", "table", 0.8, table_id=T0)]
    fetch = Fetch(FETCH)
    out, _ = run(nodes, fetch)
    assert out == nodes and fetch.calls == []
    legacy = [NodeWithScore(node=TextNode(id_="x#0", text="t", metadata={"doc_id": "x"}), score=1)]
    out, _ = run(legacy, fetch)  # chunks of other knowledge bases have no chunk_kind at all
    assert out == legacy and fetch.calls == []


def test_table_replace_drops_a_summary_whose_table_cannot_be_found() -> None:
    nodes = [
        chunk(f"{DOC}#9", "table_summary", 0.8, table_id="gone"),
        chunk(f"{DOC}#0", "text", 0.7),
    ]
    out, query = run(nodes, Fetch(FETCH))
    assert ids(out) == [f"{DOC}#0"]
    assert query.trace.table_replacements == {f"{DOC}#9": []}


def test_table_replace_fetch_error_keeps_the_summary_and_says_so() -> None:
    def broken(table_ids, acl_groups, timeout_s=None):
        raise TimeoutError("elasticsearch is slow")

    nodes = [chunk(f"{DOC}#9", "table_summary", 0.8, table_id=T0)]
    out, query = run(nodes, broken)
    assert out == nodes
    assert query.trace.table_fetch_error == "TimeoutError"


# ---- the Elasticsearch fetch ----


def _es_with_tables() -> FakeElasticsearch:
    es = FakeElasticsearch()
    es.mappings[INDEX] = {"properties": {}}
    es.docs[INDEX] = {}
    es.aliases[ALIAS] = INDEX

    def put(chunk_id: str, **source: Any) -> None:
        es.docs[INDEX][chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": DOC,
            "kb": KB,
            "acl_groups": ["public"],
            "content": f"body {chunk_id}",
            **source,
        }

    put(f"{DOC}#5", chunk_kind="table", table_id=T1, chunk_index=5)
    put(f"{DOC}#6", chunk_kind="table", table_id=T1, chunk_index=6)
    put(f"{DOC}#4", chunk_kind="table", table_id=T0, chunk_index=4)
    put(f"{DOC}#9", chunk_kind="table_summary", table_id=T1, chunk_index=9)
    es.docs[INDEX][f"{DOC}#7"] = {
        "chunk_id": f"{DOC}#7",
        "doc_id": DOC,
        "kb": KB,
        "acl_groups": ["secret"],
        "content": "hidden",
        "chunk_kind": "table",
        "table_id": T0,
        "chunk_index": 7,
    }
    return es


def test_es_table_fetch_reads_table_chunks_in_order_for_the_callers_groups() -> None:
    fetch = EsTableFetch(_es_with_tables(), kb=KB, index=ALIAS)
    found = fetch([T1, T0], ("public",))
    assert [n.node_id for n in found[T1]] == [f"{DOC}#5", f"{DOC}#6"]
    assert [n.node_id for n in found[T0]] == [f"{DOC}#4"]  # #7 belongs to another group
    node = found[T1][0]
    assert node.get_content() == f"body {DOC}#5" and node.metadata["table_id"] == T1
    assert "content" not in node.metadata
    # a summary chunk is never returned as a table
    assert all(n.metadata["chunk_kind"] == "table" for parts in found.values() for n in parts)


def test_es_table_fetch_needs_acl_groups() -> None:
    fetch = EsTableFetch(_es_with_tables(), kb=KB, index=ALIAS)
    with pytest.raises(ValueError, match="acl_groups"):
        fetch([T0], ())


# ---- wiring, fields, config version (AC10, AC11) ----


def _es() -> FakeElasticsearch:
    es = FakeElasticsearch()
    es.mappings[INDEX] = {"_meta": {"embed_model": "fake-embedding@4"}, "properties": {}}
    es.aliases[ALIAS] = INDEX
    return es


def _build(cfg: KnowledgeBaseConfig) -> SearchPipeline:
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]
    return make_search_pipeline(
        cfg,
        settings,
        acl_groups=["public"],
        es=_es(),
        embed_model=FakeEmbedding(dim=4),
        count_tokens=lambda s: len(s.split()),
        runner=RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1),
    )


def test_table_replace_is_off_by_default_and_a_retrieval_setting() -> None:
    assert RetrievalConfig().replace_table_summaries is False
    assert RetrievalConfig(replace_table_summaries=True).replace_table_summaries is True
    with pytest.raises(ValueError, match="replace_table_summaries"):
        RetrievalConfig.model_validate({"replace_table_summaries": "maybe"})


def test_table_replace_is_wired_before_the_reranker_only_when_configured() -> None:
    off = _build(KnowledgeBaseConfig(kb=KB))
    assert not any(isinstance(p, TableSummaryReplace) for p in off.postprocessors)
    cfg = KnowledgeBaseConfig(
        kb=KB, retrieval=RetrievalConfig(replace_table_summaries=True, dedupe_by_page_key=True)
    )
    on = _build(cfg)
    assert [type(p) for p in on.postprocessors] == [
        TextDedupe,
        PageKeyDedupe,
        TableSummaryReplace,
        BatchedTimeoutRerank,
        RerankThreshold,
    ]


def test_table_replace_is_wired_into_the_hybrid_search_too() -> None:
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]

    def build(cfg: KnowledgeBaseConfig) -> HybridSearch:
        return make_hybrid_search(
            cfg, settings, acl_groups=["public"], es=_es(), embed_model=FakeEmbedding(dim=4)
        )

    assert [type(p) for p in build(KnowledgeBaseConfig(kb=KB)).postprocessors] == [TextDedupe]
    on = build(KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(replace_table_summaries=True)))
    assert [type(p) for p in on.postprocessors] == [TextDedupe, TableSummaryReplace]


def test_search_config_version_ignores_replace_table_summaries_when_off() -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    cfg = KnowledgeBaseConfig(kb=KB)
    dump = cfg.model_dump(mode="json", exclude={"description", "agent", "cache"})
    dump["retrieval"].pop("dedupe_by_page_key")
    dump["retrieval"].pop("replace_table_summaries")
    dump["retrieval"].pop("replace_table_summaries_after_rerank")  # spec P6-01
    payload = {
        "kb_config": dump,
        "embed_model": "e@1",
        "reranker": {"model": settings.reranker_model, "max_length": settings.reranker_max_length},
    }
    legacy = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert search_config_version(cfg, settings, "e@1") == legacy
    on = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(replace_table_summaries=True))
    assert search_config_version(on, settings, "e@1") != legacy


def test_search_reads_pdf_fields_from_the_index_and_returns_them_with_the_chunk() -> None:
    for field in ("page", "chunk_kind", "table_id", "has_generated_text"):
        assert field in SOURCE_FIELDS
    item = chunk(f"{DOC}#4", "table", 0.8, table_id=T0, page=3, has_generated_text=False)
    chunk_out = node_to_chunk(item, KB)
    assert chunk_out.metadata["page"] == 3
    assert chunk_out.metadata["chunk_kind"] == "table" and chunk_out.metadata["table_id"] == T0
