"""A new index version, built next to the live one, then switched in; and the way back (spec P2-01).

Design s7-5. A new embedding model or a new chunking makes the old vectors or chunk ids unusable,
so the knowledge base is written again into `kb_<kb>_v<N+1>` while queries keep using the alias
on `v<N>`. `rebuild_kb` runs these steps in order:

1. `start`: the alias exists and points to one versioned index; pick the next free version (or
   resume a given unaliased one).
2. `create`: the index with the mapping of the current YAML. Its docstore is its own file
   (`<index>.json`, bound to the index uuid), so the old index's docstore is not touched.
3. `full_write`: an ingest job writes every document into the new index.
4. `sync`: the same job once more, to pick up what the source changed during the full write
   (the old index kept taking incremental updates meanwhile; the jobs of the new index use their
   own lock name, so those updates are not blocked).
5. `check`: a page that cannot be read now but is in the old index would vanish with the switch
   (the new docstore has no copy to keep, unlike P0-06's incremental rule); stop unless allowed.
6. `merge`: force merge of the new index, which nothing writes to any more: deleted chunks go,
   BM25's IDF is exact again (s7-4).
7. `gate`: the injected check (the real one is the retrieval eval, P2-03) decides. A gate that
   raises counts as a failed run, not as a pass.
8. `switch`: with the alias's ingest lock held, one atomic `_aliases` request moves the alias
   (`expected_current` makes it fail as a whole if someone moved it meanwhile), the switch is
   recorded in the new index's `_meta` (`previous_index`), the data generation goes up by one.

A failure before step 8 leaves the alias where it was; the error says so. The old index is never
deleted here. `rollback_kb` moves the alias back to the recorded previous index (or an index
given by version), also deleting nothing.

The data generation only ever goes up. Rolling back raises it again instead of restoring the old
number: cache entries written while the new index was live carry numbers above the old one, and
a restored number would make them match again after later ingests (see docs/decisions.md).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.factories import LockProvider, LockTimeoutError, make_lock_provider
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation, incr_data_generation
from rag.index.manage import (
    AliasConflictError,
    IndexStateError,
    MergeResult,
    aliased_indices,
    create_kb_index,
    force_merge_index,
    index_meta,
    list_versions,
    switch_alias,
    update_index_meta,
)
from rag.index.mapping import index_body_for_kb
from rag.index.names import KbIndexNames
from rag.index.store import EsVectorStore
from rag.ingest.build import KbProcessor
from rag.ingest.job import (
    Connector,
    IngestJob,
    IngestReport,
    RunCostLedger,
    close_embedding,
    new_run_id,
)

STEPS = ("start", "create", "full_write", "sync", "check", "merge", "gate", "switch")
PREVIOUS_INDEX_KEY = "previous_index"
"""`_meta` key of an index that an alias switch made live: the index the alias pointed to before."""

EmbedFactory = Callable[[str], tuple[BaseEmbedding, RunCostLedger | None]]
"""`run_id -> (embedding model, its cost ledger or None)`; called once per ingest phase, because a
ledger tags its lines with one run id and every phase of a rebuild is a run of its own."""


class RebuildError(RuntimeError):
    """The rebuild stopped; the message says which step failed and where the alias is."""

    def __init__(
        self,
        message: str,
        *,
        step: str,
        new_index: str | None = None,
        alias_target: str | None = None,
        alias_untouched: bool = True,
        report: RebuildReport | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.new_index = new_index
        self.alias_target = alias_target
        self.alias_untouched = alias_untouched
        self.report = report


class GateFailedError(RebuildError):
    """The gate answered "not good enough"; the new index is kept, the alias was not moved."""


class RollbackError(RuntimeError):
    """The alias was not moved back."""


class GenerationNotRaisedError(RuntimeError):
    """The alias was switched but the data generation could not be raised (Redis down): caches
    may still serve old answers. Run `status`, and raise it by running the same switch again."""

    def __init__(self, message: str, *, report: RebuildReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class GateContext:
    """What a gate may look at: the finished new index (by its own name, not the alias)."""

    es: Any
    names: KbIndexNames
    new_index: str
    old_index: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)


Gate = Callable[[GateContext], GateResult | bool]


def chunk_count_gate(min_ratio: float) -> Gate:
    """Sanity floor: the new index is not empty and holds at least `min_ratio` of the old
    index's chunks. It catches a source that failed to load or a broken chunking, not a worse
    retrieval; the retrieval eval on the dev set is the real gate (P2-03)."""

    def gate(ctx: GateContext) -> GateResult:
        new = int(ctx.es.count(index=ctx.new_index)["count"])
        old = int(ctx.es.count(index=ctx.old_index)["count"])
        metrics = {"new_chunks": float(new), "old_chunks": float(old)}
        if new == 0:
            return GateResult(False, "the new index holds no chunks", metrics)
        if new < min_ratio * old:
            return GateResult(
                False,
                f"the new index holds {new} chunks, fewer than {min_ratio:.0%} of the old "
                f"index's {old} chunks",
                metrics,
            )
        return GateResult(True, f"{new} chunks against {old} in the old index", metrics)

    return gate


@dataclass
class RebuildReport:
    run_id: str
    kb: str
    alias: str
    old_index: str
    new_index: str
    started_at: str
    duration_s: float = 0.0
    resumed: bool = False
    step_seconds: dict[str, float] = field(default_factory=dict)
    full_write: IngestReport | None = None
    sync: IngestReport | None = None
    merge: MergeResult | None = None
    gate: GateResult | None = None
    switched: bool = False
    generation_before: int | None = None
    generation_after: int | None = None
    previous_recorded: bool = False
    """The switch was written to the new index's `_meta` (what `rollback` reads)."""
    error: str | None = None
    report_path: str | None = None

    @property
    def cost_cny(self) -> float:
        return sum(r.cost_cny for r in (self.full_write, self.sync) if r is not None)

    def to_dict(self) -> dict[str, Any]:
        def counts(report: IngestReport | None) -> dict[str, Any] | None:
            return (
                None if report is None else report.to_dict()["counts"] | {"run_id": report.run_id}
            )

        merge = None
        if self.merge is not None:
            merge = {
                "deleted_docs_before": self.merge.deleted_docs_before,
                "deleted_docs_after": self.merge.deleted_docs_after,
                "segments_after": self.merge.segments_after,
            }
        gate = None
        if self.gate is not None:
            gate = {
                "passed": self.gate.passed,
                "detail": self.gate.detail,
                "metrics": dict(self.gate.metrics),
            }
        return {
            "run_id": self.run_id,
            "kb": self.kb,
            "alias": self.alias,
            "old_index": self.old_index,
            "new_index": self.new_index,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "resumed": self.resumed,
            "step_seconds": {k: round(v, 3) for k, v in self.step_seconds.items()},
            "full_write": counts(self.full_write),
            "sync": counts(self.sync),
            "merge": merge,
            "gate": gate,
            "switched": self.switched,
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
            "previous_recorded": self.previous_recorded,
            "cost_cny": round(self.cost_cny, 6),
            "error": self.error,
        }

    def summary_lines(self) -> list[str]:
        lines = [f"rebuild {self.run_id}: kb {self.kb}, {self.old_index} -> {self.new_index}"]
        for label, report in (("full write", self.full_write), ("sync", self.sync)):
            if report is not None:
                lines.append(
                    f"{label}: sources {report.sources}, added {len(report.added)}, updated "
                    f"{len(report.updated)}, deleted {len(report.deleted)}, chunks written "
                    f"{report.chunks_written}, read failures {len(report.read_failures)}"
                )
        if self.merge is not None:
            lines.append(
                f"force merge: deleted documents {self.merge.deleted_docs_before} -> "
                f"{self.merge.deleted_docs_after}, segments {self.merge.segments_after}"
            )
        if self.gate is not None:
            verdict = "passed" if self.gate.passed else "FAILED"
            lines.append(f"gate {verdict}: {self.gate.detail}")
        if self.switched:
            lines.append(f"{self.alias} -> {self.new_index} (switched; {self.old_index} is kept)")
            lines.append(
                f"data generation: {self.generation_before} -> {self.generation_after}"
                + ("" if self.previous_recorded else " (switch not recorded in the index _meta)")
            )
        lines.append(f"cost: {self.cost_cny:.6f} CNY; duration {self.duration_s:.1f} s")
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        return lines


@dataclass(frozen=True)
class RollbackResult:
    alias: str
    from_index: str
    to_index: str
    generation: int


class _Rebuild:
    """One run of `rebuild_kb`; keeps the state the error messages need."""

    def __init__(
        self,
        cfg: KnowledgeBaseConfig,
        settings: ServiceSettings,
        *,
        es: Any,
        redis: Any,
        connector: Connector,
        names: KbIndexNames,
        embed_factory: EmbedFactory,
        gate: Gate,
        processor: KbProcessor | None,
        version: int | None,
        allow_mass_delete: bool,
        allow_unreadable_docs: bool,
        lock_provider: LockProvider | None,
        after_step: Callable[[str], None] | None,
    ) -> None:
        self.cfg = cfg
        self.settings = settings
        self.es = es
        self.redis = redis
        self.connector = connector
        self.names = names
        self.embed_factory = embed_factory
        self.gate = gate
        self.processor = processor
        self.version = version
        self.allow_mass_delete = allow_mass_delete
        self.allow_unreadable_docs = allow_unreadable_docs
        self.locks = lock_provider or make_lock_provider(settings.lock_url)
        self.after_step = after_step
        self.run_id = new_run_id()
        self.step = "start"
        self.old_index = ""
        self.new_index = ""
        self.report = RebuildReport(
            run_id=self.run_id,
            kb=cfg.kb,
            alias=names.alias,
            old_index="",
            new_index="",
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.generation_prefix = f"{DEFAULT_KEY_PREFIX}{names.prefix}"

    # ---- errors -----------------------------------------------------------------------------

    def _alias_sentence(self) -> tuple[str | None, str]:
        alias = self.names.alias
        try:
            current = aliased_indices(self.es, self.names)
        except Exception as exc:  # the alias state is unknown, say so instead of guessing
            return None, f"The state of the alias {alias} could not be read ({exc})."
        target = current[0] if len(current) == 1 else None
        if not self.old_index:  # failed before the alias target was read: nothing was done
            where = ", ".join(current) if current else "nothing"
            return target, f"The alias {alias} points to {where} (unchanged)."
        if current == [self.old_index]:
            return target, f"The alias {alias} still points to {self.old_index} (unchanged)."
        return target, (
            f"The alias {alias} now points to {', '.join(current) or 'nothing'}; it was not "
            "changed by this rebuild."
        )

    def fail(
        self,
        reason: str,
        *,
        step: str | None = None,
        cls: type[RebuildError] = RebuildError,
    ) -> RebuildError:
        step = step or self.step
        target, sentence = self._alias_sentence()
        parts = [f"rebuild of {self.names.alias} failed at step {step}: {reason}.", sentence]
        if self.step != "start" and self.new_index and self.new_index in self._existing():
            parts.append(
                f"The new index {self.new_index} was kept: resume it with --version "
                f"{self.names.version_of(self.new_index)}, or delete it."
            )
        message = " ".join(parts)
        self.report.error = message
        return cls(
            message,
            step=step,
            new_index=self.new_index or None,
            alias_target=target,
            alias_untouched=True,
            report=self.report,
        )

    def _existing(self) -> set[str]:
        try:
            return set(list_versions(self.es, self.names).values())
        except Exception:
            return set()

    # ---- the run ----------------------------------------------------------------------------

    def run(self) -> RebuildReport:
        started = time.perf_counter()
        try:
            with ExitStack() as stack:
                self._enter_lock(stack)
                self._steps()
        except (RebuildError, GenerationNotRaisedError):
            raise
        except Exception as exc:
            raise self.fail(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.report.duration_s = time.perf_counter() - started
            self._write_report()
        return self.report

    def _enter_lock(self, stack: ExitStack) -> None:
        name = f"rebuild:{self.names.alias}"
        try:
            stack.enter_context(self.locks.lock(name, timeout=self.settings.ingest_lock_timeout_s))
        except LockTimeoutError as exc:
            raise self.fail(f"another rebuild of {self.names.alias} is running ({exc})") from exc

    def _phase(self, name: str) -> _Phase:
        return _Phase(self, name)

    def _steps(self) -> None:
        with self._phase("start"):
            self._start()
        with self._phase("create"):
            self._create()
        with self._phase("full_write"):
            self.report.full_write = self._ingest()
        with self._phase("sync"):
            self.report.sync = self._ingest()
        with self._phase("check"):
            self._check()
        with self._phase("merge"):
            self.report.merge = force_merge_index(
                self.es, self.new_index, timeout_s=self.settings.rebuild_force_merge_timeout_s
            )
        with self._phase("gate"):
            self._gate()
        with self._phase("switch"):
            self._switch()

    def _start(self) -> None:
        names = self.names
        current = aliased_indices(self.es, names)
        if not current:
            raise self.fail(
                f"{names.alias} has no alias yet, so there is nothing to rebuild; the first "
                "ingest (python -m rag.ingest.job) creates v1 and the alias"
            )
        if len(current) > 1:
            raise self.fail(f"{names.alias} points to several indices: {current}")
        self.old_index = current[0]
        if names.version_of(self.old_index) is None:
            raise self.fail(f"{names.alias} points to {self.old_index}, not a versioned index")
        self.report.old_index = self.old_index
        versions = list_versions(self.es, names)
        if self.version is None:
            self.new_index = names.index(max(versions, default=0) + 1)
        else:
            self.new_index = names.index(self.version)
            if self.new_index == self.old_index:
                raise self.fail(
                    f"{self.old_index} is the version that {names.alias} serves queries from; "
                    "pick another version or leave the version out"
                )
        self.report.new_index = self.new_index

    def _create(self) -> None:
        if self.new_index in self._existing():  # a leftover of a failed rebuild: continue it
            self.report.resumed = True
            return
        version = self.names.version_of(self.new_index)
        assert version is not None
        body = index_body_for_kb(self.cfg, dims=self.settings.embed_dim)
        create_kb_index(self.es, self.names, version, body)

    def _ingest(self) -> IngestReport:
        run_id = new_run_id()
        embed, ledger = self.embed_factory(run_id)
        try:
            job = IngestJob(
                self.cfg,
                self.settings,
                es=self.es,
                redis=self.redis,
                embed_model=embed,
                connector=self.connector,
                processor=self.processor,
                names=self.names,
                index=self.new_index,
                allow_mass_delete=self.allow_mass_delete,
                lock_provider=self.locks,
                ledger=ledger,
                run_id=run_id,
                lock_name=f"ingest:{self.new_index}",
                raise_generation=False,
            )
            return job.run()
        finally:
            close_embedding(embed)

    def _check(self) -> None:
        sync = self.report.sync
        assert sync is not None
        unread = set(sync.read_failures) - set(sync.kept_old_version)
        if not unread or self.allow_unreadable_docs:
            return
        old_docs = EsVectorStore(self.es, self.old_index).indexed_doc_ids()
        new_docs = EsVectorStore(self.es, self.new_index).indexed_doc_ids()
        dropped = sorted((unread & old_docs) - new_docs)
        if dropped:
            shown = ", ".join(dropped[:5]) + (
                f" (and {len(dropped) - 5} more)" if dropped[5:] else ""
            )
            raise self.fail(
                f"{len(dropped)} pages cannot be read now but are in {self.old_index}, so "
                f"switching would drop them: {shown}. Fix the source, or allow it "
                "(--allow-unreadable-docs)"
            )

    def _gate(self) -> None:
        ctx = GateContext(
            es=self.es, names=self.names, new_index=self.new_index, old_index=self.old_index
        )
        try:
            raw = self.gate(ctx)
        except Exception as exc:
            raise self.fail(f"the gate raised {type(exc).__name__}: {exc}") from exc
        result = GateResult(raw) if isinstance(raw, bool) else raw
        self.report.gate = result
        if not result.passed:
            raise self.fail(
                f"the gate rejected {self.new_index}: {result.detail or 'no reason given'}",
                cls=GateFailedError,
            )

    def _switch(self) -> None:
        alias = self.names.alias
        try:
            with self.locks.lock(f"ingest:{alias}", timeout=self.settings.ingest_lock_timeout_s):
                self._switch_locked()
        except LockTimeoutError as exc:
            raise self.fail(
                f"an ingest of {alias} is still running ({exc}); resume with --version "
                f"{self.names.version_of(self.new_index)} when it is done"
            ) from exc

    def _switch_locked(self) -> None:
        alias = self.names.alias
        before = get_data_generation(self.redis, self.cfg.kb, key_prefix=self.generation_prefix)
        try:
            switch_alias(self.es, self.names, self.new_index, expected_current=self.old_index)
        except (AliasConflictError, IndexStateError) as exc:
            raise self.fail(str(exc)) from exc
        self.report.switched = True
        self.report.generation_before = before
        try:
            update_index_meta(
                self.es,
                self.new_index,
                **{
                    PREVIOUS_INDEX_KEY: self.old_index,
                    "switched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "generation_before_switch": before,
                    "rebuild_run_id": self.run_id,
                },
            )
            self.report.previous_recorded = True
        except Exception:  # the switch stands; `rollback --to-version` still works
            self.report.previous_recorded = False
        try:
            self.report.generation_after = incr_data_generation(
                self.redis, self.cfg.kb, key_prefix=self.generation_prefix
            )
        except Exception as exc:
            message = (
                f"{alias} was switched to {self.new_index}, but the data generation was not "
                f"raised ({type(exc).__name__}: {exc}); caches may serve old results. Raise it "
                "once Redis is up, for example with `rag.index.cli switch --version N` on the "
                "current version"
            )
            self.report.error = message
            raise GenerationNotRaisedError(message, report=self.report) from exc

    def _write_report(self) -> None:
        path = Path(self.settings.ingest_report_dir) / self.cfg.kb / f"rebuild-{self.run_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.report.to_dict(), indent=2, ensure_ascii=False), "utf-8"
            )
            self.report.report_path = path.as_posix()
        except OSError:
            self.report.report_path = None


class _Phase:
    """Context manager of one step: names it for error messages, times it, then calls the hook."""

    def __init__(self, run: _Rebuild, name: str) -> None:
        self.run = run
        self.name = name

    def __enter__(self) -> None:
        self.run.step = self.name
        self.started = time.perf_counter()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.run.report.step_seconds[self.name] = time.perf_counter() - self.started
        if exc_type is None and self.run.after_step is not None:
            self.run.after_step(self.name)


def rebuild_kb(
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    *,
    es: Any,
    redis: Any,
    connector: Connector,
    names: KbIndexNames,
    embed_factory: EmbedFactory,
    gate: Gate,
    processor: KbProcessor | None = None,
    version: int | None = None,
    allow_mass_delete: bool = False,
    allow_unreadable_docs: bool = False,
    lock_provider: LockProvider | None = None,
    after_step: Callable[[str], None] | None = None,
) -> RebuildReport:
    """Build the next version of the knowledge base's index and switch the alias to it if
    `gate` accepts it. Returns the report; raises `RebuildError` (alias untouched, the message
    says so), `GateFailedError`, or `GenerationNotRaisedError` (alias switched, Redis failed).

    `after_step(name)` is called after each step has finished (tests change the source or run an
    incremental ingest on the old index there). `version` resumes an unaliased version left by a
    failed rebuild, or creates that number. `names.kb` must be `cfg.kb`.
    """
    if names.kb != cfg.kb:
        raise ValueError(f"index names are for kb {names.kb!r}, not {cfg.kb!r}")
    return _Rebuild(
        cfg,
        settings,
        es=es,
        redis=redis,
        connector=connector,
        names=names,
        embed_factory=embed_factory,
        gate=gate,
        processor=processor,
        version=version,
        allow_mass_delete=allow_mass_delete,
        allow_unreadable_docs=allow_unreadable_docs,
        lock_provider=lock_provider,
        after_step=after_step,
    ).run()


def rollback_kb(
    settings: ServiceSettings,
    *,
    es: Any,
    redis: Any,
    names: KbIndexNames,
    to_version: int | None = None,
    lock_provider: LockProvider | None = None,
) -> RollbackResult:
    """Point the alias back at the index it pointed to before the last switch (read from the
    live index's `_meta`), or at `to_version`. Deletes nothing; raises the data generation.

    Raises `RollbackError` (alias untouched) when there is nothing recorded to go back to, the
    target no longer exists, or the alias moved in the meantime.
    """
    alias = names.alias
    locks = lock_provider or make_lock_provider(settings.lock_url)
    try:
        with locks.lock(f"ingest:{alias}", timeout=settings.ingest_lock_timeout_s):
            return _rollback(settings, es, redis, names, to_version)
    except LockTimeoutError as exc:
        raise RollbackError(f"an ingest of {alias} is running ({exc}); try again later") from exc


def _rollback(
    settings: ServiceSettings, es: Any, redis: Any, names: KbIndexNames, to_version: int | None
) -> RollbackResult:
    alias = names.alias
    current_list = aliased_indices(es, names)
    if len(current_list) != 1:
        raise RollbackError(f"{alias} points to {current_list or 'nothing'}; nothing to roll back")
    current = current_list[0]
    if to_version is not None:
        target = names.index(to_version)
        if target == current:
            raise RollbackError(f"{alias} already points to {current}")
    else:
        previous = index_meta(es, current).get(PREVIOUS_INDEX_KEY)
        if not previous:
            raise RollbackError(
                f"{current} has no previous index recorded (it was not made by a rebuild, or "
                "was rolled back already); choose one with --to-version N"
            )
        target = str(previous)
    if not es.indices.exists(index=target):
        raise RollbackError(f"the previous index {target} no longer exists; {alias} is unchanged")
    try:
        switch_alias(es, names, target, expected_current=current)
    except (AliasConflictError, IndexStateError, ValueError) as exc:
        raise RollbackError(f"{exc}; {alias} is unchanged") from exc
    now = datetime.now(UTC).isoformat(timespec="seconds")
    # Best effort: the records only serve the next `status` and `rollback`.
    try:
        update_index_meta(es, current, rolled_back_to=target, rolled_back_at=now)
        update_index_meta(es, target, **{PREVIOUS_INDEX_KEY: None, "rolled_back_from": current})
    except Exception:
        pass
    prefix = f"{DEFAULT_KEY_PREFIX}{names.prefix}"
    try:
        generation = incr_data_generation(redis, names.kb, key_prefix=prefix)
    except Exception as exc:
        raise GenerationNotRaisedError(
            f"{alias} was moved back to {target}, but the data generation was not raised "
            f"({type(exc).__name__}: {exc}); caches may serve results of {current}"
        ) from exc
    return RollbackResult(alias=alias, from_index=current, to_index=target, generation=generation)
