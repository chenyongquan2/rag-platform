"""Ingest job against the compose Elasticsearch and Redis (spec P0-06, design s7-4, s7-6).

Every test gets a fresh knowledge base `k<random>` under this run's index prefix; its first job
run creates `<prefix>kb_<kb>_v1` and the alias. Embeddings are fakes of a small size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import VectorStoreQuery
from redis import Redis

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation
from rag.index.manage import create_kb_index, switch_alias
from rag.index.mapping import build_index_body
from rag.index.names import KbIndexNames
from rag.index.store import BulkWriteError, EsVectorStore
from rag.ingest import KbProcessor
from rag.ingest.job import IngestJob, IngestReport, MassDeleteError
from rag.testing.fakes import FakeEmbedding
from rag.testing.ingest import ListConnector, page, section

pytestmark = pytest.mark.integration

DIMS = 8
STEPS = [
    "markdown_split",
    {"token_split": {"max_tokens": 40}},
    {"merge_tiny": {"min_tokens": 3}},
    "assign_chunk_ids",
]
PIPELINE = {"reader": "markdown", "clean": ["md_clean"], "steps": STEPS}
NOTEBOOK_JSON = '{"cells": [], "metadata": {}, "nbformat": 4}'


def words(text: str) -> list[str]:
    return text.split()


class PoisonedEmbedding(FakeEmbedding):
    """Gives texts containing `poison` a vector one element short: Elasticsearch rejects that
    chunk (and only that chunk) in the bulk request."""

    def _vector(self, text: str) -> list[float]:
        vector = super()._vector(text)
        return vector[:-1] if "poison" in text else vector


@dataclass
class Kb:
    es: Elasticsearch
    redis: Redis
    names: KbIndexNames
    settings: ServiceSettings
    tmp: Path

    def __post_init__(self) -> None:
        self.cfg = KnowledgeBaseConfig.model_validate(
            {
                "kb": self.names.kb,
                "source": {"connector": "test_list"},
                "language": "en",
                "pipelines": {".md": PIPELINE, ".ipynb": PIPELINE},
            }
        )
        self.connector = ListConnector(self.names.kb)

    def job(self, **kwargs: Any) -> IngestJob:
        kwargs.setdefault("embed_model", FakeEmbedding(dim=DIMS))
        return IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            connector=self.connector,
            processor=self.processor(),
            names=self.names,
            **kwargs,
        )

    def run(self, **kwargs: Any) -> IngestReport:
        return self.job(**kwargs).run()

    def processor(self) -> KbProcessor:
        return KbProcessor(self.cfg, settings=self.settings, tokenizer=words)

    @property
    def index(self) -> str:
        return self.names.index(1)

    @property
    def docstore_file(self) -> Path:
        return self.tmp / "docstore" / f"{self.index}.json"

    def chunks(self) -> dict[str, dict[str, Any]]:
        """Every chunk in the index (after a refresh), `_id -> _source`."""
        self.es.indices.refresh(index=self.index)
        resp = self.es.search(index=self.index, query={"match_all": {}}, size=1000)
        return {hit["_id"]: hit["_source"] for hit in resp["hits"]["hits"]}

    def contents(self) -> dict[str, str]:
        return {i: c["content"] for i, c in self.chunks().items()}

    def expected(self) -> dict[str, str]:
        """The chunks the current source should give: read, clean and chunk without ES."""
        proc = self.processor()
        docs = proc.clean(proc.read(self.connector.documents()).documents)
        nodes: Sequence[BaseNode] = proc.chunk(docs)
        return {n.node_id: n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes}

    def doc_chunks(self, name: str) -> dict[str, dict[str, Any]]:
        prefix = self.connector.doc_id(name) + "#"
        return {i: c for i, c in self.chunks().items() if i.startswith(prefix)}

    def generation(self) -> int:
        return get_data_generation(
            self.redis, self.names.kb, key_prefix=f"{DEFAULT_KEY_PREFIX}{self.names.prefix}"
        )


@pytest.fixture
def make_kb(
    es: Elasticsearch,
    redis_client: Redis,
    make_names: Callable[[], KbIndexNames],
    service_settings: ServiceSettings,
    tmp_path: Path,
) -> Callable[..., Kb]:
    def factory(**overrides: Any) -> Kb:
        settings = service_settings.model_copy(
            update={
                "docstore_url": f"file://{(tmp_path / 'docstore').as_posix()}",
                "lock_url": f"file://{(tmp_path / 'locks').as_posix()}",
                "ingest_report_dir": (tmp_path / "reports").as_posix(),
                "embed_dim": DIMS,
                "embed_backend": "fake",
                **overrides,
            }
        )
        return Kb(es, redis_client, make_names(), settings, tmp_path)

    return factory


def test_five_change_cases(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    a_v1 = page("A", section("a"), section("aa"), section("aaa"))
    kb.connector.put("a.md", a_v1)
    kb.connector.put("b.md", page("B", section("b"), section("bb")))
    kb.connector.put("c.md", page("C", section("c")))

    added = kb.run()
    assert len(added.added) == 3
    assert kb.contents() == kb.expected()
    assert len(kb.doc_chunks("a.md")) == 3

    kb.connector.put("a.md", page("A", section("a-new")))  # modify: three chunks become one
    modified = kb.run()
    assert modified.updated == [kb.connector.doc_id("a.md")]
    assert kb.contents() == kb.expected()
    assert {c["ingest_run_id"] for c in kb.doc_chunks("a.md").values()} == {modified.run_id}

    kb.connector.remove("b.md")  # delete
    deleted = kb.run()
    assert deleted.deleted == [kb.connector.doc_id("b.md")]
    assert kb.doc_chunks("b.md") == {}
    assert kb.contents() == kb.expected()

    text = kb.connector.docs[kb.connector.doc_id("c.md")].content.decode()
    kb.connector.remove("c.md")  # rename: same text under a new path
    kb.connector.put("c-renamed.md", text)
    renamed = kb.run()
    assert renamed.added == [kb.connector.doc_id("c-renamed.md")]
    assert renamed.deleted == [kb.connector.doc_id("c.md")]
    assert kb.doc_chunks("c.md") == {}
    assert kb.contents() == kb.expected()

    kb.connector.put("a.md", a_v1)  # revert to the first version
    reverted = kb.run()
    assert reverted.updated == [kb.connector.doc_id("a.md")]
    assert kb.contents() == kb.expected()
    assert len(kb.doc_chunks("a.md")) == 3


def test_query_between_write_and_finalize_finds_doc(
    make_kb: Callable[..., Kb], monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = make_kb()
    kb.connector.put("a.md", page("A", section("old"), section("older"), section("oldest")))
    kb.connector.put("b.md", page("B", section("b")))
    kb.run()
    a_id = kb.connector.doc_id("a.md")
    kb.connector.put("a.md", page("A", section("fresh"), section("fresher")))

    seen_before_add: list[int] = []
    original_add = EsVectorStore.add

    def add_after_a_query(self: EsVectorStore, nodes: Sequence[BaseNode], **kw: Any) -> list[str]:
        # LlamaIndex has already called delete(a) at this point.
        assert a_id in self.pending_deletes
        seen_before_add.append(len(kb.doc_chunks("a.md")))
        return original_add(self, nodes, **kw)

    monkeypatch.setattr(EsVectorStore, "add", add_after_a_query)
    job = kb.job()
    job.prepare()
    job.write()

    assert seen_before_add == [3]  # the old version is still complete before the new one lands
    chunks = kb.doc_chunks("a.md")
    assert len(chunks) == 3  # two new chunks and one leftover until finalize
    store = EsVectorStore(kb.es, kb.index)
    hits = store.query(
        VectorStoreQuery(
            query_embedding=FakeEmbedding(dim=DIMS).get_query_embedding("anything"),
            similarity_top_k=5,
            doc_ids=[a_id],
        )
    )
    assert hits.ids and all(i.startswith(a_id + "#") for i in hits.ids)
    fresh = kb.es.search(index=kb.index, query={"match": {"content": "fresh0"}})
    assert [h["_id"] for h in fresh["hits"]["hits"]] == [f"{a_id}#0"]

    job.commit()
    assert len(kb.doc_chunks("a.md")) == 2
    assert kb.contents() == kb.expected()


def test_bulk_partial_failure_then_next_run_repairs(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.connector.put("a.md", page("A", section("a"), section("aa")))
    kb.connector.put("b.md", page("B", section("b")))
    kb.run()
    saved = kb.docstore_file.read_bytes()
    generation = kb.generation()

    kb.connector.put("a.md", page("A", section("poison"), section("a2"), section("a3")))
    kb.connector.put("b.md", page("B", section("b2")))
    with pytest.raises(BulkWriteError, match="1 chunks"):
        kb.run(embed_model=PoisonedEmbedding(dim=DIMS))
    assert kb.docstore_file.read_bytes() == saved
    assert kb.generation() == generation
    assert kb.contents() != kb.expected()  # half written, nothing deleted

    kb.connector.put("a.md", page("A", section("cured"), section("a2"), section("a3")))
    repaired = kb.run()
    assert sorted(repaired.updated) == sorted(kb.connector.doc_id(n) for n in ("a.md", "b.md"))
    assert kb.contents() == kb.expected()
    assert {c["ingest_run_id"] for c in kb.chunks().values()} == {repaired.run_id}
    assert kb.generation() == generation + 1


def test_one_changed_file_leaves_others_intact(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    names = [f"doc{i}.md" for i in range(6)]
    for i, name in enumerate(names):
        kb.connector.put(name, page(name, section(f"d{i}x"), section(f"d{i}y")))
    first = kb.run()
    before = kb.chunks()

    kb.connector.put("doc3.md", page("doc3.md", section("changed")))
    second = kb.run()

    assert second.updated == [kb.connector.doc_id("doc3.md")]
    after = kb.chunks()
    for name in names:
        if name == "doc3.md":
            continue
        prefix = kb.connector.doc_id(name) + "#"
        old = {i: c for i, c in before.items() if i.startswith(prefix)}
        new = {i: c for i, c in after.items() if i.startswith(prefix)}
        assert new == old and len(new) == 2
        assert {c["ingest_run_id"] for c in new.values()} == {first.run_id}
    assert kb.contents() == kb.expected()


def test_read_failure_keeps_chunks(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb(ingest_max_read_failure_ratio=0.5)
    kb.connector.put("a.md", page("A", section("a")))
    kb.connector.put("b.md", page("B", section("b")))
    kb.connector.put("nb.ipynb", page("Notebook", section("nb"), section("nbb")))
    kb.run()
    before = kb.doc_chunks("nb.ipynb")
    assert len(before) == 2

    kb.connector.put("nb.ipynb", NOTEBOOK_JSON)  # the markdown reader refuses JSON notebooks
    report = kb.run()

    assert report.kept_old_version == [kb.connector.doc_id("nb.ipynb")]
    assert report.deleted == []
    assert kb.doc_chunks("nb.ipynb") == before


def test_mass_delete_guard_against_es(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    for name in ("a.md", "b.md", "c.md", "d.md"):
        kb.connector.put(name, page(name, section(name[0])))
    kb.run()
    saved = kb.docstore_file.read_bytes()
    before = kb.chunks()

    for name in ("b.md", "c.md", "d.md"):
        kb.connector.remove(name)
    with pytest.raises(MassDeleteError):
        kb.run()

    assert kb.docstore_file.read_bytes() == saved
    assert kb.chunks() == before


def test_data_generation_once_per_changing_run(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    start = kb.generation()
    kb.connector.put("a.md", page("A", section("a")))

    assert kb.run().data_generation == start + 1  # added
    assert kb.run().data_generation == start + 1  # nothing changed: caches stay valid
    kb.connector.put("a.md", page("A", section("a2")))
    assert kb.run().data_generation == start + 2  # modified
    kb.connector.put("a.md", page("A", section("poison")))
    with pytest.raises(BulkWriteError):
        kb.run(embed_model=PoisonedEmbedding(dim=DIMS))
    assert kb.generation() == start + 2  # failed run


def test_existing_index_gets_new_fields(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    old_body = build_index_body(analyzer="standard", dims=DIMS)
    for name in ("images", "front_matter"):  # an index made before these fields existed
        del old_body["mappings"]["properties"][name]
    create_kb_index(kb.es, kb.names, 1, old_body)
    switch_alias(kb.es, kb.names, kb.index, expected_current=None)
    front_matter = "---\ntitle: Alpha\ntags: [x, y]\n---\n"
    kb.connector.put("a.md", front_matter + page("A", section("a")))

    report = kb.run()

    assert report.new_index_fields == ["front_matter", "images"]
    props = kb.es.indices.get_mapping(index=kb.index)[kb.index]["mappings"]["properties"]
    assert props["images"] == {"type": "keyword"}
    assert props["front_matter"] == {"type": "object", "enabled": False}
    assert {c["front_matter"]["title"] for c in kb.chunks().values()} == {"Alpha"}
    meta = kb.es.indices.get_mapping(index=kb.index)[kb.index]["mappings"]["_meta"]
    assert meta["embed_model"] == f"fake-embedding@{DIMS}"
    assert all(c["images"] == [] for c in kb.chunks().values())
