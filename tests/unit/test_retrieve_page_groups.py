"""One page = one group: keep one language version per `page_key` (spec P4-01 AC12; design s8-4).

The Chinese and the English manual have the same pages; a search sees both versions of a page
and both would take places in the result. With `retrieval.dedupe_by_page_key` the version in the
question's language stays. Nodes below are fakes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.page_groups import PageKeyDedupe
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline, search_config_version
from rag.retrieve.query import HybridQuery
from rag.retrieve.rerank import BatchedTimeoutRerank, RerankRunner
from rag.retrieve.search import HybridSearch, make_hybrid_search
from rag.retrieve.threshold import RerankThreshold
from rag.testing import FakeEmbedding
from rag.testing.rerank import FakeCrossEncoder
from rag.testing.services import FakeElasticsearch


def node(chunk_id: str, language: str, page_key: str | None, score: float) -> NodeWithScore:
    meta: dict[str, Any] = {"doc_id": chunk_id.split("#")[0], "language": language}
    if page_key is not None:
        meta["page_key"] = page_key
    return NodeWithScore(
        node=TextNode(id_=chunk_id, text=f"text of {chunk_id}", metadata=meta), score=score
    )


def ids(nodes: list[NodeWithScore]) -> list[str]:
    return [n.node.node_id for n in nodes]


def run(nodes: list[NodeWithScore], question: str) -> tuple[list[NodeWithScore], HybridQuery]:
    query = HybridQuery(question, acl_groups=("public",))
    return PageKeyDedupe().postprocess_nodes(nodes, query_bundle=query), query


FUSED = [
    node("en/a#0", "en", "a", 0.9),
    node("zh/a#0", "zh", "a", 0.8),
    node("en/b#0", "en", "b", 0.7),
    node("zh/c#0", "zh", "c", 0.6),
    node("en/c#0", "en", "c", 0.5),
]


def test_page_key_dedupe_keeps_the_version_in_the_questions_language() -> None:
    kept, _ = run(FUSED, "怎么向队列里添加一个部件")
    assert ids(kept) == ["zh/a#0", "en/b#0", "zh/c#0"]  # a: zh wins, b: only en, c: zh wins
    kept, _ = run(FUSED, "How do I add a widget to the queue")
    assert ids(kept) == ["en/a#0", "en/b#0", "en/c#0"]


def test_page_key_dedupe_keeps_the_best_ranked_version_when_the_question_has_no_language() -> None:
    kept, _ = run(FUSED, "12345 ?!")
    assert ids(kept) == ["en/a#0", "en/b#0", "zh/c#0"]  # a: en ranks first, c: zh ranks first


def test_page_key_dedupe_keeps_the_best_ranked_version_when_the_question_language_is_absent() -> (
    None
):
    nodes = [node("de/a#0", "de", "a", 0.9), node("fr/a#0", "fr", "a", 0.8)]
    kept, _ = run(nodes, "How do I add a widget")  # English question, neither version is English
    assert ids(kept) == ["de/a#0"]


def test_page_key_dedupe_keeps_a_lone_version_whatever_the_question_language() -> None:
    # A Chinese question that only found the English page: it stays (cross-language retrieval).
    kept, query = run([node("en/b#0", "en", "b", 0.7)], "怎么向队列里添加一个部件")
    assert ids(kept) == ["en/b#0"]
    assert query.trace.page_key_duplicates == {}


def test_page_key_dedupe_keeps_every_chunk_of_the_chosen_version() -> None:
    nodes = [
        node("zh/a#0", "zh", "a", 0.9),
        node("en/a#0", "en", "a", 0.85),
        node("zh/a#1", "zh", "a", 0.8),
        node("en/a#1", "en", "a", 0.75),
        node("zh/b#0", "zh", "b", 0.7),
    ]
    kept, _ = run(nodes, "添加部件的方法")
    assert ids(kept) == ["zh/a#0", "zh/a#1", "zh/b#0"]


def test_page_key_dedupe_leaves_order_scores_and_pages_without_a_key_alone() -> None:
    nodes = [
        node("hf/x#0", "en", None, 0.9),
        node("en/a#0", "en", "a", 0.8),
        node("zh/a#0", "zh", "a", 0.7),
        node("hf/y#0", "en", None, 0.6),
    ]
    kept, _ = run(nodes, "How to add")
    assert ids(kept) == ["hf/x#0", "en/a#0", "hf/y#0"]
    assert [n.score for n in kept] == [0.9, 0.8, 0.6]
    assert PageKeyDedupe().postprocess_nodes([]) == []


def test_page_key_dedupe_records_what_it_dropped_and_which_languages_exist() -> None:
    kept, query = run(FUSED, "怎么向队列里添加一个部件")
    assert query.trace.page_key_duplicates == {"en/a#0": "a", "en/c#0": "c"}
    meta = {n.node.node_id: n.node.metadata for n in kept}
    assert meta["zh/a#0"]["also_matched_languages"] == ["en"]  # the en version matched too
    assert meta["zh/c#0"]["also_matched_languages"] == ["en"]
    assert "also_matched_languages" not in meta["en/b#0"]


def test_page_key_dedupe_works_without_a_hybrid_query() -> None:
    kept = PageKeyDedupe().postprocess_nodes(FUSED, query_bundle=QueryBundle("添加一个部件"))
    assert ids(kept) == ["zh/a#0", "en/b#0", "zh/c#0"]
    assert ids(PageKeyDedupe().postprocess_nodes(FUSED)) == ["en/a#0", "en/b#0", "zh/c#0"]


def test_question_language_boundary_is_two_chinese_characters() -> None:
    from rag.retrieve.page_groups import question_language

    assert question_language("What does the 栈 term mean in this manual") == "en"  # one character
    assert question_language("What does the 队列 term mean in this manual") == "zh"  # two
    assert question_language("12345") == "und"


def test_page_key_dedupe_mixed_language_question_counts_chinese_characters() -> None:
    kept, _ = run(FUSED, "WidgetQueue::Add 怎么用")  # mostly an identifier, but a Chinese question
    assert ids(kept) == ["zh/a#0", "en/b#0", "zh/c#0"]


# --- configuration and wiring ------------------------------------------------------------------

KB = "pk_test"
ALIAS = "kb_pk_test"
INDEX = "kb_pk_test_v1"


def test_dedupe_by_page_key_is_off_by_default_and_a_retrieval_setting() -> None:
    assert RetrievalConfig().dedupe_by_page_key is False
    assert RetrievalConfig(dedupe_by_page_key=True).dedupe_by_page_key is True
    with pytest.raises(ValueError):
        RetrievalConfig.model_validate({"dedupe_by_page_key": "maybe"})


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


def test_search_reads_page_key_from_the_index_and_returns_it_with_the_chunk() -> None:
    """The retriever asks Elasticsearch only for the fields of `SOURCE_FIELDS`; without `page_key`
    there the postprocessor would see no page keys at all (found by the integration test)."""
    from rag.retrieve.hybrid import SOURCE_FIELDS
    from rag.retrieve.search import node_to_chunk

    assert "page_key" in SOURCE_FIELDS and "language" in SOURCE_FIELDS
    item = node("zh/a#0", "zh", "a", 0.8)
    chunk = node_to_chunk(item, "mt5")
    assert chunk.metadata["page_key"] == "a" and chunk.metadata["language"] == "zh"


def test_page_key_dedupe_is_wired_into_the_pipeline_only_when_configured() -> None:
    off = _build(KnowledgeBaseConfig(kb=KB))
    assert type(off.postprocessors[0]) is TextDedupe
    assert not any(isinstance(p, PageKeyDedupe) for p in off.postprocessors)

    cfg_on = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(dedupe_by_page_key=True))
    on = _build(cfg_on)
    kinds = [type(p) for p in on.postprocessors]
    # after the text dedupe, before the reranker: both versions must not take rerank places
    assert kinds == [TextDedupe, PageKeyDedupe, BatchedTimeoutRerank, RerankThreshold]
    assert on.config_version != off.config_version


def test_page_key_dedupe_is_wired_into_the_hybrid_search_too() -> None:
    settings = ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4)  # type: ignore[call-arg]

    def build(cfg: KnowledgeBaseConfig) -> HybridSearch:
        return make_hybrid_search(
            cfg, settings, acl_groups=["public"], es=_es(), embed_model=FakeEmbedding(dim=4)
        )

    assert [type(p) for p in build(KnowledgeBaseConfig(kb=KB)).postprocessors] == [TextDedupe]
    on = build(KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(dedupe_by_page_key=True)))
    assert [type(p) for p in on.postprocessors] == [TextDedupe, PageKeyDedupe]


def test_config_version_of_a_knowledge_base_without_the_flag_is_unchanged() -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    cfg = KnowledgeBaseConfig(kb=KB)
    # The version as it was computed before the setting existed: the setting is not in the dump.
    dump = cfg.model_dump(mode="json", exclude={"description", "agent", "cache"})
    for pipe in dump["pipelines"].values():  # no reader_params: not in the version
        pipe.pop("reader_params")
    dump["retrieval"].pop("dedupe_by_page_key")
    # added by spec P6-01: off = not in the version
    dump["retrieval"].pop("replace_table_summaries")
    dump["retrieval"].pop("replace_table_summaries_after_rerank")
    payload = {
        "kb_config": dump,
        "embed_model": "e@1",
        "reranker": {"model": settings.reranker_model, "max_length": settings.reranker_max_length},
    }
    legacy = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert search_config_version(cfg, settings, "e@1") == legacy
    on = KnowledgeBaseConfig(kb=KB, retrieval=RetrievalConfig(dedupe_by_page_key=True))
    assert search_config_version(on, settings, "e@1") != legacy
