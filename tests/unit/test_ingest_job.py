"""Ingestion job with in-memory Elasticsearch, Redis and embeddings (spec P0-06, design s7-4)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.cost import BudgetExceeded
from rag.data import SourceDocument, make_doc_id
from rag.embed import make_embedding
from rag.factories import make_lock_provider
from rag.index.generation import DEFAULT_KEY_PREFIX
from rag.index.names import KbIndexNames
from rag.index.store import BulkWriteError
from rag.ingest import KbProcessor
from rag.ingest import build as ingest_build
from rag.ingest.job import (
    IndexModelMismatchError,
    IngestError,
    IngestJob,
    MassDeleteError,
    ReadFailureError,
    RunCostLedger,
    main,
)
from rag.testing.fakes import FakeEmbedding
from rag.testing.services import FakeElasticsearch, FakeRedis

KB = "t"
DIMS = 8
STEPS = [
    "markdown_split",
    {"token_split": {"max_tokens": 40}},
    {"merge_tiny": {"min_tokens": 3}},
    "assign_chunk_ids",
]
PIPELINE = {"reader": "markdown", "clean": ["md_clean"], "steps": STEPS}
KB_YAML = {
    "kb": KB,
    "source": {"connector": "test_list"},
    "language": "en",
    "pipelines": {".md": PIPELINE, ".ipynb": PIPELINE},
}
NOTEBOOK_JSON = '{"cells": [], "metadata": {}, "nbformat": 4}'


def words(text: str) -> list[str]:
    return text.split()


def section(seed: str, n: int = 12) -> str:
    return " ".join(f"{seed}{i}" for i in range(n)) + "."


def page(name: str, *sections: str) -> str:
    body = "\n\n".join(f"## Part {i}\n\n{text}" for i, text in enumerate(sections))
    return f"# {name}\n\n{body}\n"


def doc_id(name: str) -> str:
    return make_doc_id(KB, f"docs/{name}")


def source(name: str, text: str, version: str = "src@1") -> SourceDocument:
    rel = f"docs/{name}"
    return SourceDocument(
        doc_id=make_doc_id(KB, rel),
        kb=KB,
        source_rel=rel,
        source_url=None,
        ext=PurePosixPath(name).suffix,
        content=text.encode("utf-8"),
        source_version=version,
    )


class ListConnector:
    """Snapshot connector over an editable list of documents."""

    mode = "snapshot"

    def __init__(self) -> None:
        self.docs: dict[str, SourceDocument] = {}

    def put(self, name: str, text: str, version: str = "src@1") -> None:
        doc = source(name, text, version)
        self.docs[doc.doc_id] = doc

    def remove(self, name: str) -> None:
        del self.docs[doc_id(name)]

    def documents(self) -> Iterator[SourceDocument]:
        yield from list(self.docs.values())


@dataclass
class Env:
    tmp: Path
    settings: ServiceSettings
    cfg: KnowledgeBaseConfig
    es: FakeElasticsearch = field(default_factory=FakeElasticsearch)
    redis: FakeRedis = field(default_factory=FakeRedis)
    connector: ListConnector = field(default_factory=ListConnector)
    names: KbIndexNames = field(default_factory=lambda: KbIndexNames(KB))

    def job(self, **kwargs: Any) -> IngestJob:
        kwargs.setdefault("embed_model", FakeEmbedding(dim=DIMS))
        kwargs.setdefault(
            "processor", KbProcessor(self.cfg, settings=self.settings, tokenizer=words)
        )
        return IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            connector=self.connector,
            names=self.names,
            **kwargs,
        )

    @property
    def index(self) -> str:
        return self.names.index(1)

    @property
    def docstore_file(self) -> Path:
        return self.tmp / "docstore" / f"{self.index}.json"

    def chunks(self) -> dict[str, dict[str, Any]]:
        return self.es.docs.get(self.index, {})

    def chunk_ids(self, name: str) -> list[str]:
        prefix = doc_id(name) + "#"
        return sorted(i for i in self.chunks() if i.startswith(prefix))

    def generation(self) -> int | None:
        return self.redis.values.get(f"{DEFAULT_KEY_PREFIX}{self.names.prefix}{KB}")


def make_settings(tmp: Path, **overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "docstore_url": f"file://{(tmp / 'docstore').as_posix()}",
        "lock_url": f"file://{(tmp / 'locks').as_posix()}",
        "ingest_report_dir": (tmp / "reports").as_posix(),
        "cost_ledger_path": (tmp / "cost" / "ledger.jsonl").as_posix(),
        "embed_dim": DIMS,
        "embed_backend": "fake",
        "kb_dir": (tmp / "kb").as_posix(),
        **overrides,
    }
    return ServiceSettings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path, make_settings(tmp_path), KnowledgeBaseConfig.model_validate(KB_YAML))


def test_job_counts_and_persists_docstore(env: Env) -> None:
    env.connector.put("a.md", page("A", section("a"), section("aa")) + "<!-- old note -->\n")
    env.connector.put("b.md", page("B", section("b"), section("bb")))
    env.connector.put("c.md", page("C", section("c")))

    first = env.job().run()

    assert (len(first.added), len(first.updated), len(first.unchanged), len(first.deleted)) == (
        3,
        0,
        0,
        0,
    )
    assert first.chunks_written == len(env.chunks()) > 3
    assert env.chunk_ids("a.md") == [f"{doc_id('a.md')}#0", f"{doc_id('a.md')}#1"]
    assert all(c["ingest_run_id"] == first.run_id for c in env.chunks().values())
    assert all("old note" not in c["content"] for c in env.chunks().values())  # cleaned first
    assert env.docstore_file.is_file()
    assert first.data_generation == env.generation()
    saved = json.loads(Path(first.report_path or "").read_text("utf-8"))
    assert saved["counts"] == {
        "sources": 3,
        "added": 3,
        "updated": 0,
        "unchanged": 0,
        "deleted": 0,
        "read_failures": 0,
        "chunks_written": first.chunks_written,
        "chunks_deleted": 0,
    }
    assert saved["cost_cny"] == 0.0

    # Only noise that cleaning removes changed in a; b changed; c deleted; d added.
    env.connector.put("a.md", page("A", section("a"), section("aa")) + "<!-- new note -->\n")
    env.connector.put("b.md", page("B", section("b2")))
    env.connector.remove("c.md")
    env.connector.put("d.md", page("D", section("d")))

    second = env.job().run()

    assert second.added == [doc_id("d.md")]
    assert second.updated == [doc_id("b.md")]
    assert second.unchanged == [doc_id("a.md")]
    assert second.deleted == [doc_id("c.md")]
    assert env.chunk_ids("c.md") == []
    assert env.chunk_ids("b.md") == [f"{doc_id('b.md')}#0"]  # the leftover b#1 is gone
    assert env.chunks()[f"{doc_id('b.md')}#0"]["ingest_run_id"] == second.run_id
    assert env.chunks()[f"{doc_id('a.md')}#0"]["ingest_run_id"] == first.run_id
    assert second.chunks_deleted == 2  # the leftover b#1 and c's only chunk
    assert second.data_generation == first.data_generation + 1


def test_read_failure_uses_stored_copy(tmp_path: Path) -> None:
    env = Env(
        tmp_path,
        make_settings(tmp_path, ingest_max_read_failure_ratio=0.5),
        KnowledgeBaseConfig.model_validate(KB_YAML),
    )
    env.connector.put("a.md", page("A", section("a")))
    env.connector.put("b.md", page("B", section("b")))
    env.connector.put("nb.ipynb", page("Notebook", section("nb")))
    first = env.job().run()
    nb_chunks = {i: env.chunks()[i] for i in env.chunk_ids("nb.ipynb")}
    assert nb_chunks

    env.connector.put("nb.ipynb", NOTEBOOK_JSON)  # the markdown reader refuses JSON notebooks
    env.connector.put("new.ipynb", NOTEBOOK_JSON)  # never read successfully

    second = env.job().run()

    assert set(second.read_failures) == {doc_id("nb.ipynb"), doc_id("new.ipynb")}
    assert "NotebookJsonError" in second.read_failures[doc_id("nb.ipynb")]
    assert second.kept_old_version == [doc_id("nb.ipynb")]
    assert doc_id("nb.ipynb") in second.unchanged
    assert second.deleted == []
    assert {i: env.chunks()[i] for i in env.chunk_ids("nb.ipynb")} == nb_chunks
    assert env.chunk_ids("new.ipynb") == []
    assert first.run_id != second.run_id


def test_too_many_read_failures_abort(env: Env) -> None:
    env.connector.put("a.md", page("A", section("a")))
    env.connector.put("nb.ipynb", page("Notebook", section("nb")))
    env.job().run()
    saved = env.docstore_file.read_bytes()
    chunks = dict(env.chunks())
    generation = env.generation()

    env.connector.put("nb.ipynb", NOTEBOOK_JSON)  # 1 of 2 documents fails: 50% > 5%
    with pytest.raises(ReadFailureError, match="1 of 2"):
        env.job().run()

    assert env.docstore_file.read_bytes() == saved
    assert env.chunks() == chunks
    assert env.generation() == generation


def test_mass_delete_guard_aborts_without_changes(env: Env) -> None:
    for name in ("a.md", "b.md", "c.md", "d.md"):
        env.connector.put(name, page(name, section(name[0])))
    env.job().run()
    saved = env.docstore_file.read_bytes()
    chunks = dict(env.chunks())

    for name in ("b.md", "c.md", "d.md"):
        env.connector.remove(name)
    with pytest.raises(MassDeleteError, match=r"pass 1 documents.*the 4 in the docstore"):
        env.job().run()

    assert env.docstore_file.read_bytes() == saved
    assert env.chunks() == chunks
    assert env.es.bulk_calls[1:] == []  # nothing written by the aborted run

    confirmed = env.job(allow_mass_delete=True).run()
    assert len(confirmed.deleted) == 3
    assert set(env.chunks()) == set(env.chunk_ids("a.md"))


def test_failed_run_does_not_persist_docstore(env: Env) -> None:
    env.connector.put("a.md", page("A", section("a")))
    env.connector.put("b.md", page("B", section("b")))
    env.job().run()
    saved = env.docstore_file.read_bytes()
    generation = env.generation()

    env.connector.put("a.md", page("A", section("a2"), section("a3")))
    env.es.fail_ids = {f"{doc_id('a.md')}#1"}
    with pytest.raises(BulkWriteError):
        env.job().run()
    assert env.docstore_file.read_bytes() == saved
    assert env.generation() == generation
    assert not list((env.tmp / "docstore").glob("*.tmp"))

    env.es.fail_ids = set()
    repaired = env.job().run()
    assert repaired.updated == [doc_id("a.md")]
    assert env.chunk_ids("a.md") == [f"{doc_id('a.md')}#0", f"{doc_id('a.md')}#1"]
    assert {env.chunks()[i]["ingest_run_id"] for i in env.chunk_ids("a.md")} == {repaired.run_id}


def _gateway(requests: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"index": i, "embedding": FakeEmbedding(dim=DIMS).get_text_embedding(t)}
            for i, t in enumerate(body["input"])
        ]
        tokens = sum(len(t.split()) for t in body["input"])
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": tokens}})

    return httpx.MockTransport(handler)


def test_budget_check_before_embedding(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        embed_backend="gateway",
        embed_base_url="http://gateway.test/v1",
        embed_api_key="unit-test-placeholder",
        embed_ingest_cache_url="memory://",
        budget_stop_cny=0.0,
        budget_cny=0.0,
    )
    env = Env(tmp_path, settings, KnowledgeBaseConfig.model_validate(KB_YAML))
    env.connector.put("a.md", page("A", section("a")))
    env.connector.put("b.md", page("B", section("b")))
    requests: list[dict[str, Any]] = []

    ledger = RunCostLedger.from_settings(settings, run_id="run-budget")
    embed = make_embedding("ingest", settings, ledger=ledger, transport=_gateway(requests))
    with pytest.raises(BudgetExceeded):
        env.job(embed_model=embed, ledger=ledger).run()
    assert requests == []
    assert env.chunks() == {}
    assert not env.docstore_file.exists()

    roomy = settings.model_copy(update={"budget_stop_cny": 28.0, "budget_cny": 30.0})
    env.settings = roomy
    ledger = RunCostLedger.from_settings(roomy, run_id="run-paid")
    embed = make_embedding("ingest", roomy, ledger=ledger, transport=_gateway(requests))
    report = env.job(embed_model=embed, ledger=ledger).run()

    assert requests
    lines = ledger.entries()
    assert lines and all(line["ingest_run_id"] == "run-paid" for line in lines)
    assert report.run_id == "run-paid"
    assert report.cost_cny == pytest.approx(sum(line["cny"] for line in lines))
    assert report.cost_cny > 0
    assert report.embed_model.startswith(settings.embed_model)


def test_ingest_lock_is_exclusive(tmp_path: Path) -> None:
    env = Env(
        tmp_path,
        make_settings(tmp_path, ingest_lock_timeout_s=0.0),
        KnowledgeBaseConfig.model_validate(KB_YAML),
    )
    env.connector.put("a.md", page("A", section("a")))
    locks = make_lock_provider(env.settings.lock_url)

    with locks.lock(f"ingest:{env.names.alias}"), pytest.raises(IngestError, match="another"):
        env.job().run()
    assert env.chunks() == {}

    env.job().run()  # the lock is free again
    assert env.chunk_ids("a.md")


def test_index_refuses_other_embed_model(env: Env) -> None:
    env.connector.put("a.md", page("A", section("a")))
    first = env.job().run()
    mapping = env.es.mappings[env.index]
    assert mapping["_meta"]["embed_model"] == first.embed_model == f"fake-embedding@{DIMS}"
    chunks = dict(env.chunks())

    env.connector.put("a.md", page("A", section("changed")))
    other = FakeEmbedding(dim=DIMS, model_name="other-model")
    with pytest.raises(IndexModelMismatchError, match="other-model"):
        env.job(embed_model=other).run()
    assert env.chunks() == chunks


def _write_kb_yaml(tmp: Path) -> None:
    kb_dir = tmp / "kb" / "public"
    kb_dir.mkdir(parents=True)
    (kb_dir / f"{KB}.yaml").write_text(json.dumps(KB_YAML), "utf-8")  # JSON is valid YAML


def test_cli_runs_and_writes_report(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_kb_yaml(env.tmp)
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda _model: words)
    env.connector.put("a.md", page("A", section("a")))
    deps = {"settings": env.settings, "es": env.es, "redis": env.redis, "connector": env.connector}

    assert main(["--kb", KB, "--index-prefix", ""], **deps) == 0
    out = capsys.readouterr().out
    assert "added 1, updated 0, unchanged 0, deleted 0" in out
    reports = list((env.tmp / "reports" / KB).glob("ingest-*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text("utf-8"))["index"] == env.index

    assert main(["--kb", "nope"], **deps) == 2
    assert main(["--kb", KB, "--limit", "0"], **deps) == 2
    env.connector.docs.clear()
    assert main(["--kb", KB, "--index-prefix", ""], **deps) == 1  # mass-delete guard
    assert "mass delete" in capsys.readouterr().err


def test_cli_fake_embed_uses_fake_prefix(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_kb_yaml(env.tmp)
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda _model: words)
    env.connector.put("a.md", page("A", section("a")))
    real = env.settings.model_copy(update={"embed_backend": "gateway"})
    deps = {"settings": real, "es": env.es, "redis": env.redis, "connector": env.connector}

    assert main(["--kb", KB, "--fake-embed"], **deps) == 0

    assert env.es.aliases == {f"fake_kb_{KB}": f"fake_kb_{KB}_v1"}
    assert (env.tmp / "docstore" / f"fake_kb_{KB}_v1.json").is_file()
    assert f"{DEFAULT_KEY_PREFIX}fake_{KB}" in env.redis.values
    assert env.es.mappings[f"fake_kb_{KB}_v1"]["_meta"]["embed_model"].startswith("fake-embedding")
    assert "fake_kb_t_v1" in capsys.readouterr().out
