"""Spec P6-01 AC10: the replacement can also run after the reranker.

Before the reranker the cross-encoder reads the table text; after it the cross-encoder reads the
summary (a clean sentence it judges better than a Markdown table) and the table takes the
summary's place and score afterwards.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.pipeline import make_search_pipeline, search_config_version
from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import RERANK_SCORE, BatchedTimeoutRerank, RerankRunner
from rag.retrieve.table_groups import TableSummaryReplace
from rag.retrieve.threshold import RerankThreshold
from rag.testing import FakeEmbedding
from rag.testing.rerank import FakeCrossEncoder
from rag.testing.services import FakeElasticsearch

KB = "pdf_demo"
DOC = "pdf_demo:p.pdf#p3"
T0 = f"{DOC}#t0"


def test_replacement_after_the_reranker_keeps_the_summarys_rerank_score() -> None:
    summary = TextNode(
        id_=f"{DOC}#9",
        text="summary",
        metadata={
            "doc_id": DOC,
            "chunk_kind": "table_summary",
            "table_id": T0,
            RERANK_SCORE: 0.91,
        },
    )
    table = TextNode(
        id_=f"{DOC}#4",
        text="| the | table |",
        metadata={"doc_id": DOC, "chunk_kind": "table", "table_id": T0},
    )

    def fetch(table_ids: Any, acl_groups: Any, timeout_s: Any = None) -> dict[str, list[TextNode]]:
        return {T0: [table]}

    out = TableSummaryReplace(fetch).postprocess_nodes(
        [NodeWithScore(node=summary, score=0.5)], query_bundle=HybridQuery("q", acl_groups=("a",))
    )
    assert [n.node.node_id for n in out] == [f"{DOC}#4"]
    assert out[0].node.metadata[RERANK_SCORE] == 0.91
    assert out[0].score == 0.5


def _build(cfg: KnowledgeBaseConfig):
    es = FakeElasticsearch()
    es.mappings["kb_pdf_demo_v1"] = {"_meta": {"embed_model": "fake-embedding@4"}, "properties": {}}
    es.aliases["kb_pdf_demo"] = "kb_pdf_demo_v1"
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]
    return make_search_pipeline(
        cfg,
        settings,
        acl_groups=["public"],
        es=es,
        embed_model=FakeEmbedding(dim=4),
        count_tokens=lambda s: len(s.split()),
        runner=RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1),
    )


def test_replacement_stage_is_a_setting_that_is_off_by_default() -> None:
    assert RetrievalConfig().replace_table_summaries_after_rerank is False
    assert RetrievalConfig(
        replace_table_summaries_after_rerank=True
    ).replace_table_summaries_after_rerank


def test_replacement_after_the_reranker_is_wired_after_the_threshold() -> None:
    cfg = KnowledgeBaseConfig(
        kb=KB,
        retrieval=RetrievalConfig(
            replace_table_summaries=True, replace_table_summaries_after_rerank=True
        ),
    )
    kinds = [type(p) for p in _build(cfg).postprocessors]
    assert kinds == [TextDedupe, BatchedTimeoutRerank, RerankThreshold, TableSummaryReplace]
    before = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(replace_table_summaries=True))
    kinds = [type(p) for p in _build(before).postprocessors]
    assert kinds == [TextDedupe, TableSummaryReplace, BatchedTimeoutRerank, RerankThreshold]


def test_the_stage_setting_does_nothing_while_the_replacement_is_off() -> None:
    cfg = KnowledgeBaseConfig(
        kb=KB, retrieval=RetrievalConfig(replace_table_summaries_after_rerank=True)
    )
    assert not any(isinstance(p, TableSummaryReplace) for p in _build(cfg).postprocessors)


def test_config_version_ignores_the_stage_setting_when_it_is_off() -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    cfg = KnowledgeBaseConfig(kb=KB)
    dump = cfg.model_dump(mode="json", exclude={"description", "agent", "cache"})
    for key in (
        "dedupe_by_page_key",
        "replace_table_summaries",
        "replace_table_summaries_after_rerank",
    ):
        dump["retrieval"].pop(key)
    payload = {
        "kb_config": dump,
        "embed_model": "e@1",
        "reranker": {"model": settings.reranker_model, "max_length": settings.reranker_max_length},
    }
    legacy = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert search_config_version(cfg, settings, "e@1") == legacy
    on = KnowledgeBaseConfig(
        kb=KB,
        retrieval=RetrievalConfig(
            replace_table_summaries=True, replace_table_summaries_after_rerank=True
        ),
    )
    only_replace = KnowledgeBaseConfig(
        kb=KB, retrieval=RetrievalConfig(replace_table_summaries=True)
    )
    assert search_config_version(on, settings, "e@1") != search_config_version(
        only_replace, settings, "e@1"
    )
