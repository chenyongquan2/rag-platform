"""New index version, gate, alias switch and rollback with in-memory services (spec P2-01, s7-5)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.data import make_doc_id
from rag.factories import InProcessLockProvider
from rag.index import cli
from rag.index.generation import DEFAULT_KEY_PREFIX
from rag.index.manage import create_kb_index, index_meta, switch_alias
from rag.index.mapping import index_body_for_kb
from rag.index.names import KbIndexNames
from rag.index.rebuild import (
    GateContext,
    GateFailedError,
    GateResult,
    RebuildError,
    RollbackError,
    chunk_count_gate,
    rebuild_kb,
    rollback_kb,
)
from rag.ingest import KbProcessor
from rag.ingest import build as ingest_build
from rag.ingest.job import IngestError, IngestJob
from rag.ingest.job import main as ingest_main
from rag.testing.fakes import FakeEmbedding
from rag.testing.ingest import ListConnector, page, section
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


def doc_id(name: str) -> str:
    return make_doc_id(KB, f"docs/{name}")


def embed_factory(_run_id: str) -> tuple[FakeEmbedding, None]:
    return FakeEmbedding(dim=DIMS), None


def accept(_ctx: GateContext) -> bool:
    return True


@dataclass
class Env:
    tmp: Path
    settings: ServiceSettings
    cfg: KnowledgeBaseConfig
    es: FakeElasticsearch = field(default_factory=FakeElasticsearch)
    redis: FakeRedis = field(default_factory=FakeRedis)
    connector: ListConnector = field(default_factory=lambda: ListConnector(KB))
    names: KbIndexNames = field(default_factory=lambda: KbIndexNames(KB))
    locks: InProcessLockProvider = field(default_factory=InProcessLockProvider)

    def processor(self) -> KbProcessor:
        return KbProcessor(self.cfg, settings=self.settings, tokenizer=words)

    def ingest(self, **kwargs: Any) -> Any:
        """A normal incremental run against the alias (what runs on v(N) during a rebuild)."""
        return IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            connector=self.connector,
            embed_model=FakeEmbedding(dim=DIMS),
            processor=self.processor(),
            names=self.names,
            lock_provider=self.locks,
            **kwargs,
        ).run()

    def rebuild(self, **kwargs: Any) -> Any:
        kwargs.setdefault("gate", accept)
        kwargs.setdefault("embed_factory", embed_factory)
        return rebuild_kb(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            connector=self.connector,
            names=self.names,
            processor=self.processor(),
            lock_provider=self.locks,
            **kwargs,
        )

    def rollback(self, **kwargs: Any) -> Any:
        return rollback_kb(
            self.settings,
            es=self.es,
            redis=self.redis,
            names=self.names,
            lock_provider=self.locks,
            **kwargs,
        )

    def index(self, version: int) -> str:
        return self.names.index(version)

    def alias(self) -> str | None:
        return self.es.aliases.get(self.names.alias)

    def chunks(self, version: int) -> dict[str, dict[str, Any]]:
        return self.es.docs.get(self.index(version), {})

    def contents(self, version: int) -> dict[str, str]:
        return {i: c["content"] for i, c in self.chunks(version).items()}

    def expected(self) -> dict[str, str]:
        proc = self.processor()
        docs = proc.clean(proc.read(self.connector.documents()).documents)
        return {n.node_id: n.get_content() for n in proc.chunk(docs)}  # type: ignore[arg-type]

    def generation(self) -> int:
        return self.redis.values[f"{DEFAULT_KEY_PREFIX}{self.names.prefix}{KB}"]

    def docstore_file(self, version: int) -> Path:
        return self.tmp / "docstore" / f"{self.index(version)}.json"


def make_settings(tmp: Path, **overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "docstore_url": f"file://{(tmp / 'docstore').as_posix()}",
        "lock_url": f"file://{(tmp / 'locks').as_posix()}",
        "ingest_report_dir": (tmp / "reports").as_posix(),
        "cost_ledger_path": (tmp / "cost" / "ledger.jsonl").as_posix(),
        "embed_dim": DIMS,
        "embed_backend": "fake",
        "kb_dir": (tmp / "kb").as_posix(),
        "ingest_lock_timeout_s": 0.2,
        **overrides,
    }
    return ServiceSettings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    env = Env(tmp_path, make_settings(tmp_path), KnowledgeBaseConfig.model_validate(KB_YAML))
    env.connector.put("a.md", page("A", section("a"), section("aa"), section("aaa")))
    env.connector.put("b.md", page("B", section("b"), section("bb")))
    env.connector.put("c.md", page("C", section("c")))
    env.ingest()  # v1 and the alias
    return env


def test_rebuild_creates_next_version_with_own_docstore_and_switches(env: Env) -> None:
    v1_chunks = dict(env.chunks(1))
    v1_docstore = env.docstore_file(1).read_bytes()

    report = env.rebuild()

    assert (report.old_index, report.new_index) == (env.index(1), env.index(2))
    assert report.switched is True
    assert env.alias() == env.index(2)
    assert env.chunks(1) == v1_chunks  # the old index is kept as it was
    assert env.docstore_file(1).read_bytes() == v1_docstore
    assert env.docstore_file(2).is_file() and env.docstore_file(2).read_bytes() != v1_docstore
    assert env.contents(2) == env.expected()
    assert len(report.full_write.added) == 3 and report.sync.added == []
    assert report.full_write.run_id != report.sync.run_id
    assert index_meta(env.es, env.index(2))["previous_index"] == env.index(1)
    saved = json.loads(Path(report.report_path or "").read_text("utf-8"))
    assert saved["new_index"] == env.index(2) and saved["switched"] is True


def test_sync_picks_up_changes_made_during_the_full_write(env: Env) -> None:
    during_v1: list[Any] = []

    def source_changes_meanwhile(step: str) -> None:
        if step != "full_write":
            return
        env.connector.put("a.md", page("A", section("a-new")))  # modified
        env.connector.put("d.md", page("D", section("d")))  # added
        env.connector.remove("b.md")  # deleted
        # The old index keeps taking incremental updates: the rebuild must not hold its lock.
        during_v1.append(env.ingest())

    report = env.rebuild(after_step=source_changes_meanwhile)

    assert during_v1[0].updated == [doc_id("a.md")]
    assert env.contents(1) == env.expected()
    assert report.sync.updated == [doc_id("a.md")]
    assert report.sync.added == [doc_id("d.md")]
    assert report.sync.deleted == [doc_id("b.md")]
    assert env.contents(2) == env.expected()  # no stale chunk of a or b
    runs = {c["ingest_run_id"] for c in env.chunks(2).values()}
    assert runs == {report.full_write.run_id, report.sync.run_id}


def test_rebuild_does_not_block_incremental_runs_on_the_old_index(env: Env) -> None:
    """An incremental run on the alias, started while a phase of the rebuild is writing."""
    during: list[Any] = []
    inner = threading.Event()

    class IngestWhileEmbedding(FakeEmbedding):
        def _vector(self, text: str) -> list[float]:
            if not inner.is_set():  # first vector of the phase: the job holds its lock now
                inner.set()
                env.connector.put("c.md", page("C", section("c-meanwhile")))
                during.append(env.ingest())
            return super()._vector(text)

    def factory(_run_id: str) -> tuple[FakeEmbedding, None]:
        return IngestWhileEmbedding(dim=DIMS), None

    report = env.rebuild(embed_factory=factory)

    assert during and during[0].updated == [doc_id("c.md")] and during[0].index == env.index(1)
    assert report.switched and env.contents(2) == env.expected()


def test_force_merge_after_last_write_before_gate(env: Env) -> None:
    seen: dict[str, Any] = {}

    def gate(ctx: GateContext) -> bool:
        seen["merges"] = list(env.es.merges)
        seen["bulks"] = len(env.es.bulk_calls)
        seen["new"] = ctx.new_index
        seen["old"] = ctx.old_index
        return True

    env.rebuild(gate=gate)

    assert [m["index"] for m in seen["merges"]] == [env.index(2)]
    assert seen["merges"][0]["bulk_calls"] == seen["bulks"] == len(env.es.bulk_calls)
    assert seen["merges"][0]["max_num_segments"] == 1
    assert seen["merges"][0]["reopens"] == 1  # closed and opened first: drops the history
    assert (seen["new"], seen["old"]) == (env.index(2), env.index(1))
    assert env.alias() == env.index(2)  # the gate ran before the switch, the switch followed


def test_gate_failure_leaves_alias_untouched(env: Env) -> None:
    before = env.generation()
    old_chunks = dict(env.chunks(1))

    def reject(_ctx: GateContext) -> GateResult:
        return GateResult(False, "hit@5 0.52 is below 0.60", {"hit_at_5": 0.52})

    with pytest.raises(GateFailedError) as excinfo:
        env.rebuild(gate=reject)

    error = excinfo.value
    assert error.step == "gate" and error.alias_untouched is True
    assert "hit@5 0.52" in str(error) and "unchanged" in str(error)
    assert "without --version" in str(error)  # changed settings need a new version, not a resume
    assert error.report is not None and error.report.gate is not None
    assert error.report.gate.metrics == {"hit_at_5": 0.52}
    assert env.alias() == env.index(1)
    assert env.generation() == before
    assert env.chunks(1) == old_chunks
    assert env.chunks(2)  # the new version is kept for a look or a resume

    again = env.rebuild()  # a leftover version is not reused unless asked: v3
    assert again.new_index == env.index(3) and env.alias() == env.index(3)


def test_gate_exception_leaves_alias_untouched(env: Env) -> None:
    def broken(_ctx: GateContext) -> bool:
        raise RuntimeError("eval service down")

    with pytest.raises(RebuildError, match="eval service down") as excinfo:
        env.rebuild(gate=broken)

    assert excinfo.value.step == "gate" and not isinstance(excinfo.value, GateFailedError)
    assert env.alias() == env.index(1)


@pytest.mark.parametrize("step", ["start", "create", "full_write", "sync", "merge"])
def test_failure_before_switch_leaves_alias_untouched(
    env: Env, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = env.generation()
    callback: Callable[[str], None] | None = None
    kwargs: dict[str, Any] = {}
    if step == "start":
        kwargs["version"] = 1  # the version the alias serves
    elif step == "create":

        def no_create(**_: Any) -> None:
            raise RuntimeError("cannot create the index")

        monkeypatch.setattr(env.es.indices, "create", no_create)
    elif step == "full_write":
        env.es.fail_ids = {f"{doc_id('a.md')}#0"}
    elif step == "sync":

        def poison_after_full_write(done: str) -> None:
            if done == "full_write":
                env.connector.put("a.md", page("A", section("a-new")))
                env.es.fail_ids = {f"{doc_id('a.md')}#0"}

        callback = poison_after_full_write
    else:

        def no_merge(**_: Any) -> None:
            raise RuntimeError("merge failed")

        monkeypatch.setattr(env.es.indices, "forcemerge", no_merge)

    with pytest.raises(RebuildError) as excinfo:
        env.rebuild(after_step=callback, **kwargs)

    error = excinfo.value
    assert error.step == step and error.alias_untouched is True
    assert f"step {step}" in str(error) and "unchanged" in str(error)
    assert error.new_index == env.index(1 if step == "start" else 2)
    assert env.alias() == env.index(1)
    assert env.generation() == before


def test_switch_raises_generation_once(env: Env) -> None:
    before = env.generation()

    report = env.rebuild()

    assert report.generation_before == before
    assert report.generation_after == env.generation() == before + 1


def test_alias_moved_during_rebuild_fails_without_switching(env: Env) -> None:
    before = env.generation()

    def someone_else_switches(step: str) -> None:
        if step == "merge":
            create_kb_index(env.es, env.names, 3, index_body_for_kb(env.cfg, dims=DIMS))
            switch_alias(env.es, env.names, env.index(3))

    with pytest.raises(RebuildError) as excinfo:
        env.rebuild(after_step=someone_else_switches)

    assert excinfo.value.step == "switch"
    assert env.alias() == env.index(3)  # left where the other party put it
    assert env.generation() == before
    assert "not changed by this rebuild" in str(excinfo.value)


def test_unreadable_page_of_old_index_blocks_switch(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ingest_max_read_failure_ratio=0.5)
    env = Env(tmp_path, settings, KnowledgeBaseConfig.model_validate(KB_YAML))
    env.connector.put("a.md", page("A", section("a")))
    env.connector.put("b.md", page("B", section("b")))
    env.connector.put("nb.ipynb", page("Notebook", section("nb")))
    env.ingest()
    env.connector.put("nb.ipynb", NOTEBOOK_JSON)  # the markdown reader refuses JSON notebooks

    with pytest.raises(RebuildError) as excinfo:
        env.rebuild()

    assert excinfo.value.step == "check"
    assert doc_id("nb.ipynb") in str(excinfo.value) and "allow-unreadable-docs" in str(
        excinfo.value
    )
    assert env.alias() == env.index(1)

    report = env.rebuild(version=2, allow_unreadable_docs=True)  # resume the leftover version
    assert report.switched and doc_id("nb.ipynb") in report.sync.read_failures
    assert not any(i.startswith(doc_id("nb.ipynb")) for i in env.chunks(2))


def test_rollback_points_alias_back_without_deleting(env: Env) -> None:
    env.rebuild()
    old, new = dict(env.chunks(1)), dict(env.chunks(2))
    generation = env.generation()

    result = env.rollback()

    assert (result.from_index, result.to_index) == (env.index(2), env.index(1))
    assert env.alias() == env.index(1)
    assert env.chunks(1) == old and env.chunks(2) == new  # nothing deleted, nothing changed
    assert result.generation == env.generation() == generation + 1  # forward, never restored
    assert "previous_index" not in index_meta(env.es, env.index(1))
    with pytest.raises(RollbackError, match="no previous index"):
        env.rollback()
    assert env.alias() == env.index(1)

    assert index_meta(env.es, env.index(2))["rolled_back_to"] == env.index(1)
    forward = env.rollback(to_version=2)  # explicit target: back to the new index
    assert forward.to_index == env.index(2) and env.alias() == env.index(2)
    assert "rolled_back_to" not in index_meta(env.es, env.index(2))  # no stale record


def test_rollback_refuses_without_switch_record(env: Env) -> None:
    with pytest.raises(RollbackError, match="no previous index"):
        env.rollback()
    assert env.alias() == env.index(1)

    env.rebuild()
    env.es.indices.delete(index=env.index(1))  # someone deleted the old version meanwhile
    with pytest.raises(RollbackError, match="no longer exists"):
        env.rollback()
    assert env.alias() == env.index(2)


def test_second_rebuild_is_refused(env: Env) -> None:
    errors: list[RebuildError] = []

    def start_another(step: str) -> None:
        if step == "full_write":
            with pytest.raises(RebuildError) as excinfo:
                env.rebuild()
            errors.append(excinfo.value)

    env.rebuild(after_step=start_another)

    assert errors and errors[0].step == "start" and "another rebuild" in str(errors[0])
    assert env.alias() == env.index(2) and env.index(3) not in env.es.mappings


def test_rebuild_needs_an_alias(tmp_path: Path) -> None:
    env = Env(tmp_path, make_settings(tmp_path), KnowledgeBaseConfig.model_validate(KB_YAML))
    with pytest.raises(RebuildError, match="no alias") as excinfo:
        env.rebuild()
    assert excinfo.value.step == "start" and env.es.mappings == {}


def test_resume_existing_unaliased_version(env: Env) -> None:
    def reject(_ctx: GateContext) -> bool:
        return False

    with pytest.raises(GateFailedError):
        env.rebuild(gate=reject)
    writes = len(env.es.bulk_calls)

    with pytest.raises(RebuildError, match="serves queries") as excinfo:
        env.rebuild(version=1)
    assert excinfo.value.step == "start"

    report = env.rebuild(version=2)

    assert report.new_index == env.index(2) and env.alias() == env.index(2)
    assert env.index(3) not in env.es.mappings
    assert len(env.es.bulk_calls) == writes  # nothing to write again: the docstore knew it all
    assert report.full_write.unchanged and not report.full_write.added


def test_force_merge_reopens_the_index_when_close_fails_after_applying(env: Env) -> None:
    env.es.close_fails_after_applying = True  # a client time-out: Elasticsearch did close it

    with pytest.raises(RebuildError, match="injected time-out") as excinfo:
        env.rebuild()

    assert excinfo.value.step == "merge"
    assert env.es.closed == set()  # the failure handler opened it again
    assert env.alias() == env.index(1)


def test_resume_opens_a_version_left_closed(env: Env) -> None:
    env.es.open_failures = 2  # the open and the retry in the failure handler both fail

    with pytest.raises(RebuildError) as excinfo:
        env.rebuild()

    assert excinfo.value.step == "merge" and env.es.closed == {env.index(2)}
    report = env.rebuild(version=2)  # the resume opens it before it writes
    assert report.resumed and report.switched
    assert env.es.closed == set() and env.alias() == env.index(2)


def test_unrecorded_switch_is_visible_and_rollback_still_works(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def break_meta_after_gate(step: str) -> None:
        if step == "gate":

            def no_meta(**_: Any) -> None:
                raise RuntimeError("meta write failed")

            monkeypatch.setattr(env.es.indices, "put_mapping", no_meta)

    report = env.rebuild(after_step=break_meta_after_gate)

    assert report.switched and report.previous_recorded is False
    assert any("not recorded" in line and "--to-version" in line for line in report.summary_lines())
    monkeypatch.undo()
    with pytest.raises(RollbackError, match="no previous index"):
        env.rollback()
    assert env.rollback(to_version=1).to_index == env.index(1)


def test_rollback_reports_when_it_could_not_record(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.rebuild()

    def no_meta(**_: Any) -> None:
        raise RuntimeError("meta write failed")

    monkeypatch.setattr(env.es.indices, "put_mapping", no_meta)
    result = env.rollback()

    assert env.alias() == env.index(1) and result.recorded is False


def test_chunk_count_gate(env: Env) -> None:
    ctx = GateContext(es=env.es, names=env.names, new_index=env.index(2), old_index=env.index(1))
    body = index_body_for_kb(env.cfg, dims=DIMS)
    create_kb_index(env.es, env.names, 2, body)
    gate = chunk_count_gate(0.5)
    old_count = len(env.chunks(1))

    assert gate(ctx).passed is False  # empty
    for n in range(old_count // 2 - 1):
        env.es.docs[env.index(2)][f"x#{n}"] = {"doc_id": "x"}
    result = gate(ctx)
    assert result.passed is False and "chunks" in result.detail
    env.es.docs[env.index(2)]["x#last"] = {"doc_id": "x"}
    env.es.docs[env.index(2)]["x#more"] = {"doc_id": "x"}
    result = gate(ctx)
    assert result.passed is True
    assert result.metrics["old_chunks"] == old_count


def test_ingest_job_without_generation_raise_and_own_lock(env: Env) -> None:
    before = env.generation()
    env.connector.put("d.md", page("D", section("d")))

    quiet = IngestJob(
        env.cfg,
        env.settings,
        es=env.es,
        redis=env.redis,
        connector=env.connector,
        embed_model=FakeEmbedding(dim=DIMS),
        processor=env.processor(),
        names=env.names,
        lock_provider=env.locks,
        raise_generation=False,
    ).run()

    assert quiet.added == [doc_id("d.md")]
    assert quiet.generation_raised is False and env.generation() == before

    with env.locks.lock(f"ingest:{env.names.alias}", timeout=1):
        with pytest.raises(IngestError, match="another ingest"):
            env.ingest()
        env.connector.put("e.md", page("E", section("e")))
        other = env.ingest(lock_name="ingest:somewhere-else")
    assert other.added == [doc_id("e.md")] and env.generation() == before + 1


def test_cli_rebuild_and_rollback(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (env.tmp / "kb" / "public").mkdir(parents=True)
    (env.tmp / "kb" / "public" / f"{KB}.yaml").write_text(json.dumps(KB_YAML), "utf-8")
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda _model: words)
    deps = {"settings": env.settings, "es": env.es, "redis": env.redis, "connector": env.connector}
    common = ["--kb", KB, "--index-prefix", ""]
    capsys.readouterr()

    assert cli.main(["rebuild", *common, "--fake-embed"], **deps) == 0
    out = capsys.readouterr().out
    assert f"{env.names.alias} -> {env.index(2)}" in out and "switched" in out
    assert env.alias() == env.index(2)
    reports = list((env.tmp / "reports" / KB).glob("rebuild-*.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text("utf-8"))["switched"] is True

    assert cli.main(["status", *common], **deps) == 0
    assert f"rollback target: {env.index(1)}" in capsys.readouterr().out

    assert cli.main(["rollback", *common], **deps) == 0
    assert env.alias() == env.index(1) and env.index(2) in env.es.mappings
    assert "rolled back" in capsys.readouterr().out
    assert cli.main(["rollback", *common], **deps) == 1  # nothing recorded to go back to
    assert "no previous index" in capsys.readouterr().err

    for name in ("b.md", "c.md"):
        env.connector.remove(name)  # a source that shrank: the chunk-count gate rejects it
    env.connector.put("a.md", page("A", section("a")))
    assert cli.main(["rebuild", *common, "--fake-embed"], **deps) == 1
    assert "gate" in capsys.readouterr().err and env.alias() == env.index(1)
    assert cli.main(["rebuild", *common, "--fake-embed", "--no-gate"], **deps) == 0
    assert env.alias() == env.index(4)

    assert cli.main(["rebuild", "--kb", "nope"], **deps) == 2
    assert cli.main(["rebuild", *common, "--limit", "0"], **deps) == 2
    with pytest.raises(SystemExit) as excinfo:  # argparse refuses the two together
        cli.main(["rebuild", *common, "--no-gate", "--min-chunk-ratio", "0.5"], **deps)
    assert excinfo.value.code == 2
    assert cli.main(["rebuild", *common, "--min-chunk-ratio", "1.5"], **deps) == 2


def test_cli_rebuild_prints_the_index_prefix(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (env.tmp / "kb" / "public").mkdir(parents=True)
    (env.tmp / "kb" / "public" / f"{KB}.yaml").write_text(json.dumps(KB_YAML), "utf-8")
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda _model: words)
    deps = {"settings": env.settings, "es": env.es, "redis": env.redis, "connector": env.connector}
    assert ingest_main(["--kb", KB, "--fake-embed"], **deps) == 0  # creates fake_kb_t_v1
    capsys.readouterr()

    assert cli.main(["rebuild", "--kb", KB, "--fake-embed"], **deps) == 0  # no --index-prefix

    out = capsys.readouterr().out
    assert "fake_kb_t -> fake_kb_t_v2" in out
    assert "index prefix: fake_" in out and "--index-prefix fake_" in out
