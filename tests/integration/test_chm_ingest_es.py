"""A tiny invented bilingual CHM manual through the real pipeline into Elasticsearch (P4-01 AC14).

The demo manual (`rag.testing.chm`) is invented; embeddings are fakes of 8 dimensions. Every index
carries this run's unique prefix and is deleted afterwards (see `conftest`); the real
`kb_hf_docs` indexes are never touched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from redis import Redis

from rag.config import KnowledgeBaseConfig, RetrievalConfig, ServiceSettings
from rag.index.names import KbIndexNames
from rag.ingest import KbProcessor
from rag.ingest.job import IngestJob
from rag.registry import connectors
from rag.retrieve.search import make_hybrid_search
from rag.testing.chm import demo_kb_dict, demo_manual
from rag.testing.fakes import FakeEmbedding

pytestmark = pytest.mark.integration

DIMS = 8


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


class Demo:
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        redis: Redis,
        names: KbIndexNames,
        settings: ServiceSettings,
        tmp: Path,
        **overrides: Any,
    ) -> None:
        self.es, self.redis, self.names, self.settings = elasticsearch, redis, names, settings
        volumes = demo_manual(tmp / "manual")
        data = demo_kb_dict(volumes, kb=names.kb, **overrides)
        self.cfg = KnowledgeBaseConfig.model_validate(data)

    def ingest(self) -> Any:
        connector = connectors.create(
            "chm_dir",
            kb=self.cfg.kb,
            settings=self.settings,
            **(self.cfg.source.model_extra or {}),
        )
        job = IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            embed_model=FakeEmbedding(dim=DIMS),
            connector=connector,
            processor=KbProcessor(self.cfg, settings=self.settings, tokenizer=words),
            names=self.names,
        )
        return job.run()

    @property
    def index(self) -> str:
        return self.names.index(1)

    def search(self, question: str, **filters: Any) -> list[Any]:
        search = make_hybrid_search(
            self.cfg,
            self.settings,
            acl_groups=["public"],
            index_prefix=self.names.prefix,
            es=self.es,
            embed_model=FakeEmbedding(dim=DIMS),
        )
        try:
            result = search.retrieve(question, kb=self.cfg.kb, top_k=10, filters=filters or None)
        finally:
            search.close()
        return result.chunks


@pytest.fixture
def make_demo(
    es: Elasticsearch,
    redis_client: Redis,
    make_names: Callable[[], KbIndexNames],
    service_settings: ServiceSettings,
    tmp_path: Path,
) -> Callable[..., Demo]:
    settings = service_settings.model_copy(
        update={
            "docstore_url": f"file://{(tmp_path / 'docstore').as_posix()}",
            "lock_url": f"file://{(tmp_path / 'locks').as_posix()}",
            "ingest_report_dir": (tmp_path / "reports").as_posix(),
            "embed_dim": DIMS,
            "embed_backend": "fake",
        }
    )

    def factory(**overrides: Any) -> Demo:
        return Demo(es, redis_client, make_names(), settings, tmp_path, **overrides)

    return factory


def _sources(demo: Demo) -> dict[str, dict[str, Any]]:
    demo.es.indices.refresh(index=demo.index)
    hits = demo.es.search(index=demo.index, query={"match_all": {}}, size=100)["hits"]["hits"]
    return {h["_id"]: h["_source"] for h in hits}


def test_chunks_carry_language_page_key_links_and_title_path(
    make_demo: Callable[..., Demo],
) -> None:
    demo = make_demo()
    report = demo.ingest()
    chunks = _sources(demo)
    assert report.chunks_written == len(chunks) == 7
    assert {c["language"] for c in chunks.values()} == {"en", "zh"}
    a_zh = chunks[f"{demo.cfg.kb}:zh/a.htm#0"]
    a_en = chunks[f"{demo.cfg.kb}:en/a.htm#0"]
    assert a_zh["page_key"] == a_en["page_key"] == "a"
    assert a_zh["title_path"].startswith("甲") and a_en["title_path"].startswith("A")
    assert a_zh["doc_type"] == "chm" and a_zh["unit"] == "a.htm"
    assert a_zh["source_rel"] == "zh/a.htm" and a_zh["acl_groups"] == ["public"]
    assert a_zh["links_to"] == a_en["links_to"] == ["b"]
    linking = demo.es.search(index=demo.index, query={"term": {"links_to": "b"}})["hits"]["hits"]
    assert sorted(h["_source"]["doc_id"] for h in linking) == [
        f"{demo.cfg.kb}:en/a.htm",
        f"{demo.cfg.kb}:zh/a.htm",
    ]
    aggs = demo.es.search(
        index=demo.index, size=0, aggs={"keys": {"terms": {"field": "page_key"}}}
    )["aggregations"]["keys"]["buckets"]
    counts = {b["key"]: b["doc_count"] for b in aggs}
    assert counts["a"] == 2 and counts["b"] == 2 and counts["only_en"] == 1


def test_the_analyzer_in_the_yaml_reaches_the_index_and_searches_work(
    make_demo: Callable[..., Demo],
) -> None:
    demo = make_demo(es={"analyzer": "cjk"})
    demo.ingest()
    properties = demo.es.indices.get_mapping(index=demo.index)[demo.index]["mappings"]["properties"]
    assert properties["content"]["analyzer"] == properties["title_path"]["analyzer"] == "cjk"
    demo.es.indices.refresh(index=demo.index)
    found = demo.es.search(index=demo.index, query={"match": {"content": "内容"}})["hits"]["hits"]
    assert sorted(h["_source"]["doc_id"] for h in found) == [
        f"{demo.cfg.kb}:zh/b.htm",
        f"{demo.cfg.kb}:zh/lie.htm",
    ]


def test_language_filter_returns_one_language(make_demo: Callable[..., Demo]) -> None:
    demo = make_demo()
    demo.ingest()
    zh = demo.search("gizmo", language="zh")
    en = demo.search("gizmo", language="en")
    assert zh and {c.metadata["language"] for c in zh} == {"zh"}
    assert en and {c.metadata["language"] for c in en} == {"en"}


def test_search_without_the_page_key_option_returns_both_versions_of_a_page(
    make_demo: Callable[..., Demo],
) -> None:
    demo = make_demo()
    demo.ingest()
    chunks = demo.search("gizmo")
    pages_b = {c.metadata["language"] for c in chunks if c.metadata.get("page_key") == "b"}
    assert pages_b == {"zh", "en"}


def test_dedupe_by_page_key_keeps_the_version_in_the_questions_language(
    make_demo: Callable[..., Demo],
) -> None:
    demo = make_demo(retrieval=RetrievalConfig(dedupe_by_page_key=True).model_dump())
    demo.ingest()
    zh_question = demo.search("gizmo 怎么用")
    en_question = demo.search("How do I use the gizmo")
    for chunks, language, other in ((zh_question, "zh", "en"), (en_question, "en", "zh")):
        b = [c for c in chunks if c.metadata.get("page_key") == "b"]
        assert b and {c.metadata["language"] for c in b} == {language}
        assert b[0].metadata["also_matched_languages"] == [other]
        keys = [c.metadata.get("page_key") for c in chunks]
        assert len(keys) == len(set(keys))  # one chunk per page here: one result per page
