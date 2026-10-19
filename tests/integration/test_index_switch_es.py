"""The five change cases, the new-index flow, the switch and the rollback against the compose
Elasticsearch and Redis (spec P2-01, design s7-4, s7-5, s7-6, s19 P2 acceptance).

Every test gets a fresh knowledge base `k<random>` under this run's index prefix. Embeddings are
fakes. A background thread queries the alias in a loop while the code under test runs, to show
that a page is never missing and the alias never fails.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from llama_index.core.schema import BaseNode, MetadataMode
from redis import Redis

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation
from rag.index.names import KbIndexNames
from rag.index.rebuild import (
    GateContext,
    GateFailedError,
    GateResult,
    RebuildError,
    RebuildReport,
    rebuild_kb,
    rollback_kb,
)
from rag.index.store import EsVectorStore
from rag.ingest import KbProcessor
from rag.ingest.job import IngestJob, IngestReport
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
SMALL_STEPS = [
    "markdown_split",
    {"token_split": {"max_tokens": 14}},
    {"merge_tiny": {"min_tokens": 3}},
    "assign_chunk_ids",
]
NOTEBOOK_JSON = '{"cells": [], "metadata": {}, "nbformat": 4}'


def words(text: str) -> list[str]:
    return text.split()


def kb_yaml(kb: str, steps: list[Any]) -> KnowledgeBaseConfig:
    pipeline = {"reader": "markdown", "clean": ["md_clean"], "steps": steps}
    return KnowledgeBaseConfig.model_validate(
        {
            "kb": kb,
            "source": {"connector": "test_list"},
            "language": "en",
            "pipelines": {".md": pipeline, ".ipynb": pipeline},
        }
    )


def accept(_ctx: GateContext) -> bool:
    return True


def embed_factory(_run_id: str) -> tuple[FakeEmbedding, None]:
    return FakeEmbedding(dim=DIMS), None


class Poller:
    """Counts the hits of one query through the alias in a loop, from its own thread."""

    def __init__(self, es: Elasticsearch, alias: str, query: dict[str, Any]) -> None:
        self.es, self.alias, self.query = es, alias, query
        self.counts: list[int] = []
        self.errors: list[str] = []
        self.indices: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                resp = self.es.search(index=self.alias, query=self.query, size=1)
                self.counts.append(int(resp["hits"]["total"]["value"]))
                self.indices.update(h["_index"] for h in resp["hits"]["hits"])
            except Exception as exc:  # any failure of the alias is what the test looks for
                self.errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.003)

    def __enter__(self) -> Poller:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def assert_never_empty(self, *, at_least_polls: int = 5) -> None:
        assert self.errors == []
        assert len(self.counts) >= at_least_polls, f"only {len(self.counts)} polls ran"
        assert min(self.counts) >= 1, f"a poll found nothing: {self.counts}"


@dataclass
class Kb:
    es: Elasticsearch
    redis: Redis
    names: KbIndexNames
    settings: ServiceSettings
    tmp: Path
    cfg: KnowledgeBaseConfig

    def __post_init__(self) -> None:
        self.connector = ListConnector(self.names.kb)

    def processor(self, cfg: KnowledgeBaseConfig | None = None) -> KbProcessor:
        return KbProcessor(cfg or self.cfg, settings=self.settings, tokenizer=words)

    def job(self, **kwargs: Any) -> IngestJob:
        return IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            embed_model=FakeEmbedding(dim=DIMS),
            connector=self.connector,
            processor=self.processor(),
            names=self.names,
            **kwargs,
        )

    def run(self, **kwargs: Any) -> IngestReport:
        """An incremental run against the alias (what runs on v(N) all the time)."""
        return self.job(**kwargs).run()

    def rebuild(self, cfg: KnowledgeBaseConfig | None = None, **kwargs: Any) -> RebuildReport:
        kwargs.setdefault("gate", accept)
        return rebuild_kb(
            cfg or self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            connector=self.connector,
            names=self.names,
            embed_factory=embed_factory,
            processor=self.processor(cfg),
            **kwargs,
        )

    def rollback(self, **kwargs: Any) -> Any:
        return rollback_kb(self.settings, es=self.es, redis=self.redis, names=self.names, **kwargs)

    def index(self, version: int) -> str:
        return self.names.index(version)

    def alias_target(self) -> list[str]:
        resp = self.es.options(ignore_status=404).indices.get_alias(name=self.names.alias)
        return [] if resp.meta.status == 404 else sorted(resp.body)

    def chunks(self, version: int = 1) -> dict[str, dict[str, Any]]:
        """Every chunk of one index version (after a refresh), `_id -> _source`."""
        index = self.index(version)
        self.es.indices.refresh(index=index)
        resp = self.es.search(index=index, query={"match_all": {}}, size=1000)
        return {hit["_id"]: hit["_source"] for hit in resp["hits"]["hits"]}

    def contents(self, version: int = 1) -> dict[str, str]:
        return {i: c["content"] for i, c in self.chunks(version).items()}

    def expected(self, cfg: KnowledgeBaseConfig | None = None) -> dict[str, str]:
        """The chunks the current source should give: read, clean and chunk without ES."""
        proc = self.processor(cfg)
        docs = proc.clean(proc.read(self.connector.documents()).documents)
        nodes: Sequence[BaseNode] = proc.chunk(docs)
        return {n.node_id: n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes}

    def doc_chunks(self, name: str, version: int = 1) -> dict[str, dict[str, Any]]:
        prefix = self.connector.doc_id(name) + "#"
        return {i: c for i, c in self.chunks(version).items() if i.startswith(prefix)}

    def others(self, version: int, *, without: str) -> dict[str, dict[str, Any]]:
        prefix = self.connector.doc_id(without) + "#"
        return {i: c for i, c in self.chunks(version).items() if not i.startswith(prefix)}

    def generation(self) -> int:
        return get_data_generation(
            self.redis, self.names.kb, key_prefix=f"{DEFAULT_KEY_PREFIX}{self.names.prefix}"
        )

    def doc_query(self, name: str) -> dict[str, Any]:
        return {"term": {"doc_id": self.connector.doc_id(name)}}

    def seed(self, names: Sequence[str] = ("a.md", "b.md", "c.md")) -> None:
        sections = {"a.md": 3, "b.md": 2, "c.md": 2}
        for name in names:
            seed = name[0]
            parts = [section(seed * (n + 1)) for n in range(sections.get(name, 2))]
            self.connector.put(name, page(name, *parts))
        self.run()


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
        names = make_names()
        return Kb(es, redis_client, names, settings, tmp_path, kb_yaml(names.kb, STEPS))

    return factory


@pytest.fixture
def defer_delete(monkeypatch: pytest.MonkeyPatch) -> list[Callable[[], None]]:
    """Calls registered here run between "new chunks written and visible" and "old chunks
    deleted" of every ingest run (just before `EsVectorStore.finalize`)."""
    hooks: list[Callable[[], None]] = []
    original = EsVectorStore.finalize

    def finalize(self: EsVectorStore, also_clean: Any = ()) -> Any:
        for hook in hooks:
            hook()
        return original(self, also_clean)

    monkeypatch.setattr(EsVectorStore, "finalize", finalize)
    return hooks


# ---- the five change cases (incremental runs on the live alias) -----------------------------


def test_case_edit_one_file(
    make_kb: Callable[..., Kb], defer_delete: list[Callable[[], None]]
) -> None:
    kb = make_kb()
    kb.seed()
    a_id = kb.connector.doc_id("a.md")
    before_a = kb.doc_chunks("a.md")
    before_others = kb.others(1, without="a.md")
    assert len(before_a) == 3
    seen: list[dict[str, int]] = []

    def query_between_write_and_delete() -> None:
        found_new = kb.es.search(index=kb.names.alias, query={"match": {"content": "fresh0"}})
        found_old = kb.es.search(index=kb.names.alias, query={"match": {"content": "aa0"}})
        seen.append(
            {
                "new": found_new["hits"]["total"]["value"],
                "old": found_old["hits"]["total"]["value"],
                "chunks": len(kb.doc_chunks("a.md")),
            }
        )

    defer_delete.append(query_between_write_and_delete)
    kb.connector.put("a.md", page("A", section("fresh")))  # three chunks become one

    with Poller(kb.es, kb.names.alias, kb.doc_query("a.md")) as poller:
        report = kb.run()

    assert report.updated == [a_id]
    assert seen == [{"new": 1, "old": 1, "chunks": 3}]  # both versions findable, nothing deleted
    poller.assert_never_empty()  # the page was found at every moment of the run
    assert len(kb.doc_chunks("a.md")) == 1
    assert kb.others(1, without="a.md") == before_others  # every other chunk: same ids, same run
    assert kb.contents() == kb.expected()


def test_case_add_one_file(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()
    before = kb.chunks()
    kb.connector.put("d.md", page("D", section("dd"), section("ddd")))

    with Poller(kb.es, kb.names.alias, kb.doc_query("a.md")) as poller:
        report = kb.run()

    assert report.added == [kb.connector.doc_id("d.md")]
    poller.assert_never_empty()
    assert len(kb.doc_chunks("d.md")) == 2
    found = kb.es.search(index=kb.names.alias, query={"match": {"content": "dd0"}})
    assert found["hits"]["total"]["value"] >= 1
    assert kb.others(1, without="d.md") == before
    assert kb.contents() == kb.expected()


def test_case_delete_one_file(
    make_kb: Callable[..., Kb], defer_delete: list[Callable[[], None]]
) -> None:
    kb = make_kb()
    kb.seed()
    before_others = kb.others(1, without="b.md")
    still_there: list[int] = []
    defer_delete.append(lambda: still_there.append(len(kb.doc_chunks("b.md"))))
    kb.connector.remove("b.md")

    with Poller(kb.es, kb.names.alias, kb.doc_query("a.md")) as poller:
        report = kb.run()

    assert report.deleted == [kb.connector.doc_id("b.md")]
    assert still_there == [2]  # the old chunks go in the last step, not before the new ones
    poller.assert_never_empty()
    assert kb.doc_chunks("b.md") == {}
    assert kb.others(1, without="b.md") == before_others
    assert kb.contents() == kb.expected()


def _ingest_notebook(kb: Kb) -> None:
    kb.connector.put("a.md", page("A", section("a")))
    kb.connector.put("b.md", page("B", section("b")))
    kb.connector.put("nb.ipynb", page("Notebook", section("nb"), section("nbb")))
    kb.run()


def test_case_unreadable_file_keeps_old_version(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb(ingest_max_read_failure_ratio=0.5)
    _ingest_notebook(kb)
    before = kb.chunks()
    assert len(kb.doc_chunks("nb.ipynb")) == 2
    kb.connector.put("nb.ipynb", NOTEBOOK_JSON)  # the markdown reader refuses JSON notebooks

    with Poller(kb.es, kb.names.alias, kb.doc_query("nb.ipynb")) as poller:
        report = kb.run()

    assert report.kept_old_version == [kb.connector.doc_id("nb.ipynb")]
    assert report.deleted == [] and report.updated == []
    poller.assert_never_empty()
    assert kb.chunks() == before  # same chunks, same ids, same run ids: nothing was touched


def test_unreadable_file_blocks_rebuild_switch(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb(ingest_max_read_failure_ratio=0.5)
    _ingest_notebook(kb)
    generation = kb.generation()
    kb.connector.put("nb.ipynb", NOTEBOOK_JSON)

    with pytest.raises(RebuildError) as excinfo:
        kb.rebuild()

    assert excinfo.value.step == "check" and "nb.ipynb" in str(excinfo.value)
    assert kb.alias_target() == [kb.index(1)]
    assert kb.generation() == generation
    assert len(kb.doc_chunks("nb.ipynb")) == 2  # the old index still has the old version

    report = kb.rebuild(version=2, allow_unreadable_docs=True)  # the leftover is resumed
    assert report.switched and kb.alias_target() == [kb.index(2)]
    assert kb.doc_chunks("nb.ipynb", version=2) == {}


def test_case_chunking_change_goes_through_new_version(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()
    old = kb.chunks()
    old_generation = kb.generation()
    new_cfg = kb_yaml(kb.names.kb, SMALL_STEPS)  # sections now split into two chunks each
    during: list[tuple[bool, bool]] = []

    def look(step: str) -> None:
        if step == "full_write":
            during.append((kb.alias_target() == [kb.index(1)], kb.chunks(1) == old))

    polls = [Poller(kb.es, kb.names.alias, kb.doc_query(n)) for n in ("a.md", "b.md", "c.md")]
    with ExitStack() as stack:
        for poller in polls:
            stack.enter_context(poller)
        report = kb.rebuild(new_cfg, after_step=look)

    assert during == [(True, True)]  # while v2 was written, queries and v1 were as before
    for poller in polls:
        poller.assert_never_empty()
    new = kb.chunks(2)
    assert report.switched and kb.alias_target() == [kb.index(2)]
    assert len(new) > len(old)
    assert not {i for i in new if i in old and new[i]["content"] == old[i]["content"]}
    assert kb.contents(2) == kb.expected(new_cfg) != kb.expected()
    assert kb.chunks(1) == old  # the old index is kept untouched for a rollback
    assert kb.generation() == old_generation + 1
    assert kb.es.count(index=kb.names.alias)["count"] == len(new)


# ---- the switch, the rollback, the gate ------------------------------------------------------


def test_search_is_never_interrupted_during_the_switch(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()
    query = {"match": {"content": "bb0"}}  # a word in b.md, which both versions hold

    with Poller(kb.es, kb.names.alias, query) as poller:
        kb.rebuild()  # v1 -> v2
        kb.rollback()  # v2 -> v1
        kb.rollback(to_version=2)  # v1 -> v2
        kb.rollback(to_version=1)  # v2 -> v1
        kb.rebuild(version=2)  # v1 -> v2 again (the version exists, nothing to write)
        time.sleep(0.1)

    poller.assert_never_empty(at_least_polls=20)
    assert poller.indices == {kb.index(1), kb.index(2)}  # it was served from both, never failed
    assert kb.alias_target() == [kb.index(2)]


def test_rollback_against_es(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()
    kb.rebuild()
    old, new = kb.chunks(1), kb.chunks(2)
    generation = kb.generation()

    result = kb.rollback()

    assert (result.from_index, result.to_index) == (kb.index(2), kb.index(1))
    assert kb.alias_target() == [kb.index(1)]
    assert kb.chunks(1) == old and kb.chunks(2) == new  # nothing deleted, nothing changed
    assert kb.generation() == generation + 1 == result.generation
    # The live index takes incremental runs again (its docstore is the one it always had).
    kb.connector.put("a.md", page("A", section("again")))
    report = kb.run()
    assert report.updated == [kb.connector.doc_id("a.md")] and report.index == kb.index(1)


def test_gate_failure_keeps_alias_against_es(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()
    old = kb.chunks(1)
    generation = kb.generation()

    def reject(ctx: GateContext) -> GateResult:
        assert ctx.new_index == kb.index(2) and ctx.old_index == kb.index(1)
        return GateResult(False, "Hit@5 fell", {"hit_at_5": 0.4})

    with pytest.raises(GateFailedError, match="Hit@5 fell"):
        kb.rebuild(gate=reject)

    assert kb.alias_target() == [kb.index(1)]
    assert kb.chunks(1) == old and kb.generation() == generation
    assert kb.contents(2) == kb.expected()  # the new version is complete, kept for a look


def test_force_merge_removes_deleted_documents(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()

    def edit_everything_meanwhile(step: str) -> None:
        if step == "full_write":  # the sync run then replaces and deletes chunks in v2
            for name in ("a.md", "b.md", "c.md"):
                kb.connector.put(name, page(name, section(name[0] + "new")))

    report = kb.rebuild(after_step=edit_everything_meanwhile)

    assert report.merge is not None and report.merge.deleted_docs_before > 0
    assert report.merge.deleted_docs_after == 0 and report.merge.segments_after == 1
    segments = kb.es.indices.segments(index=kb.index(2))["indices"][kb.index(2)]["shards"]
    stored = [seg for copies in segments.values() for c in copies for seg in c["segments"].values()]
    assert len(stored) == 1 and stored[0]["deleted_docs"] == 0
    # What the merge is for: BM25 counts only live chunks (N), not the deleted history.
    live = len(kb.chunks(2))
    resp = kb.es.search(
        index=kb.index(2), query={"match": {"content": "anew0"}}, explain=True, size=1
    )
    totals: list[float] = []

    def collect(node: dict[str, Any]) -> None:
        if "total number of documents with field" in node["description"]:
            totals.append(node["value"])
        for child in node.get("details", []):
            collect(child)

    collect(resp["hits"]["hits"][0]["_explanation"])
    assert totals == [live]
    assert kb.contents(2) == kb.expected()


def test_resume_reopens_a_version_left_closed(make_kb: Callable[..., Kb]) -> None:
    kb = make_kb()
    kb.seed()

    def reject(_ctx: GateContext) -> bool:
        return False

    with pytest.raises(GateFailedError):
        kb.rebuild(gate=reject)
    kb.es.indices.close(index=kb.index(2))  # what a failed merge could leave behind

    report = kb.rebuild(version=2)

    assert report.resumed and report.switched and kb.alias_target() == [kb.index(2)]
    assert kb.contents(2) == kb.expected()
