"""Ingestion job of one knowledge base (spec P0-06; design s7-4, s7-5, s5-4, s16-8).

    uv run python -m rag.ingest.job --kb hf_docs                          # the YAML's `limit`
    uv run python -m rag.ingest.job --kb hf_docs --limit 50 --fake-embed  # no model calls
    uv run python -m rag.ingest.job --kb hf_docs --all                    # every document

One run, in this order (s7-4):
1. Take the knowledge base's ingest lock. Find the index behind the alias (the first run creates
   v1 and the alias), add mapping fields it lacks, and check that it was written with the same
   embedding model.
2. Read every document from the connector and clean it. Cleaning runs before the pipeline because
   LlamaIndex hashes the input documents before it runs any step. A document that fails to read
   is replaced by its copy in the docstore, so it counts as unchanged instead of deleted; more
   than 5% failures stop the run.
3. Compare with the docstore (added, updated, unchanged, deleted). Stop if the run would pass
   fewer than half of the docstore's documents: UPSERTS_AND_DELETE deletes docstore entries at
   once, so this guard must come before `run()`. With real embeddings, stop if the estimated
   cost does not fit the budget.
4. `IngestionPipeline(chunk steps + embedding, docstore, EsVectorStore, UPSERTS_AND_DELETE).run()`
   writes new chunks; then refresh.
5. `EsVectorStore.finalize()` deletes old chunks, the data generation goes up by one when the index
   changed, the docstore is saved, a report is written.
If anything fails before the docstore is saved, the saved docstore is the previous one and the
next run does the same work again: chunk ids are fixed, writes overwrite, vectors come from the
L0 cache. Exit codes: 0 ok, 1 failed (nothing saved), 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import BaseNode, Document, TransformComponent
from pydantic import PrivateAttr

import rag.data  # noqa: F401  (registers the connectors named in knowledge-base YAML)
from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_configs
from rag.config.settings import mask_url
from rag.cost import BudgetExceeded, CostLedger
from rag.data import SourceDocument
from rag.embed import make_embedding
from rag.factories import (
    JsonFileDocumentStore,
    LockProvider,
    LockTimeoutError,
    make_docstore,
    make_es_client,
    make_lock_provider,
    make_redis_client,
)
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation, incr_data_generation
from rag.index.manage import (
    aliased_indices,
    ensure_index_fields,
    index_embed_model,
    init_kb,
    set_index_embed_model,
)
from rag.index.mapping import index_body_for_kb
from rag.index.names import KbIndexNames
from rag.index.store import EsVectorStore, WriteFields
from rag.ingest.build import KbProcessor
from rag.registry import connectors

FAKE_INDEX_PREFIX = "fake_"
"""Index prefix of runs with fake embeddings, so fake vectors never land in the real index."""


class IngestError(RuntimeError):
    """The run stopped; the docstore file was not saved."""


class ReadFailureError(IngestError):
    """Too many documents failed to read; nothing was changed."""


class MassDeleteError(IngestError):
    """The run would delete most of the knowledge base; nothing was changed."""


class IndexModelMismatchError(IngestError):
    """The index holds vectors of another embedding model; nothing was changed."""


class Connector(Protocol):
    def documents(self) -> Iterable[SourceDocument]: ...


def new_run_id() -> str:
    """`20261017T153012Z-ab12cd`: sortable by time, unique across processes."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def embed_model_id(embed: BaseEmbedding) -> str:
    """Name and size of the vectors an embedding model makes (`model@dim`, plus the prefix hash
    for the cached gateway client): two ids differ exactly when vectors are not comparable."""
    cache_model = getattr(embed, "cache_model", None)
    if cache_model:
        return str(cache_model)
    dim = getattr(embed, "dim", None)
    return f"{embed.model_name}@{dim}" if dim else str(embed.model_name)


def content_hash(text: str) -> str:
    """sha256 of the cleaned body text of a page (s5-4)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunCostLedger(CostLedger):
    """Cost ledger that tags each line with the ingest run and adds up the run's cost."""

    def __init__(self, path: str | Path, *, run_id: str, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self.run_id = run_id
        self.run_cny = 0.0
        self._run_lock = threading.Lock()

    @classmethod
    def from_settings(  # type: ignore[override]
        cls, settings: ServiceSettings | None = None, *, run_id: str
    ) -> RunCostLedger:
        settings = settings or ServiceSettings()
        return cls(
            settings.cost_ledger_path,
            run_id=run_id,
            prices=settings.model_prices,
            budget_stop_cny=settings.budget_stop_cny,
            budget_cny=settings.budget_cny,
        )

    def record(
        self,
        *,
        kind: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        **extra: Any,
    ) -> dict[str, Any]:
        row = super().record(
            kind=kind,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ingest_run_id=self.run_id,
            **extra,
        )
        with self._run_lock:
            self.run_cny += float(row["cny"])
        return row


class KbChunker(TransformComponent):
    """The chunk steps of the knowledge base as one transformation.

    Extensions may have different steps but the IngestionPipeline takes one list, so each
    document goes through the steps of its own `pipeline_version` (`KbProcessor.chunk`).
    """

    _processor: KbProcessor = PrivateAttr()

    def __init__(self, processor: KbProcessor, **data: Any) -> None:
        super().__init__(**data)
        self._processor = processor

    @classmethod
    def class_name(cls) -> str:
        return "KbChunker"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        return self._processor.chunk(nodes)  # type: ignore[arg-type]


@dataclass
class IngestReport:
    run_id: str
    kb: str
    index: str
    embed_model: str
    started_at: str
    duration_s: float
    sources: int
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    deleted: list[str]
    chunks_written: int
    chunks_deleted: int
    data_generation: int
    generation_raised: bool
    cost_cny: float
    docstore: str
    read_failures: dict[str, str] = field(default_factory=dict)
    kept_old_version: list[str] = field(default_factory=list)
    """Documents that failed to read and kept their previous version from the docstore."""
    no_pipeline: list[str] = field(default_factory=list)
    skipped_by_cleaning: dict[str, str] = field(default_factory=dict)
    new_index_fields: list[str] = field(default_factory=list)
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kb": self.kb,
            "index": self.index,
            "embed_model": self.embed_model,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "counts": {
                "sources": self.sources,
                "added": len(self.added),
                "updated": len(self.updated),
                "unchanged": len(self.unchanged),
                "deleted": len(self.deleted),
                "read_failures": len(self.read_failures),
                "chunks_written": self.chunks_written,
                "chunks_deleted": self.chunks_deleted,
            },
            "data_generation": self.data_generation,
            "generation_raised": self.generation_raised,
            "cost_cny": round(self.cost_cny, 6),
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "read_failures": self.read_failures,
            "kept_old_version": self.kept_old_version,
            "no_pipeline": self.no_pipeline,
            "skipped_by_cleaning": self.skipped_by_cleaning,
            "new_index_fields": self.new_index_fields,
            "docstore": self.docstore,
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"ingest {self.run_id}: kb {self.kb}, index {self.index}, "
            f"embeddings {self.embed_model}",
            f"documents: sources {self.sources}, added {len(self.added)}, updated "
            f"{len(self.updated)}, unchanged {len(self.unchanged)}, deleted {len(self.deleted)}",
            f"chunks: written {self.chunks_written}, deleted {self.chunks_deleted}",
            f"read failures: {len(self.read_failures)} (kept the old version: "
            f"{len(self.kept_old_version)}); skipped by cleaning: {len(self.skipped_by_cleaning)}; "
            f"no pipeline: {len(self.no_pipeline)}",
        ]
        for doc_id, error in list(self.read_failures.items())[:5]:
            lines.append(f"  read failure {doc_id}: {error}")
        raised = "raised" if self.generation_raised else "unchanged, the index did not change"
        lines += [
            f"data generation: {self.data_generation} ({raised})",
            f"cost: {self.cost_cny:.6f} CNY; duration {self.duration_s:.1f} s",
        ]
        if self.new_index_fields:
            lines.append(f"fields added to the index mapping: {', '.join(self.new_index_fields)}")
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        return lines


@dataclass
class _Plan:
    sources: int
    documents: list[Document]
    fields: dict[str, WriteFields]
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    deleted: list[str]
    read_failures: dict[str, str]
    kept_old_version: list[str]
    no_pipeline: list[str]
    skipped_by_cleaning: dict[str, str]


class IngestJob:
    """One ingestion run of one knowledge base; `run()` does everything under the ingest lock.

    The phases `prepare()`, `write()` and `commit()` are public for tests that look at the index
    between them (`run()` is the only caller that holds the lock). Pass `index` to write a given
    version instead of the alias's (the rebuild of s7-5). With a `RunCostLedger`, the run is
    checked against the budget before any embedding call; without one, no budget check (fakes).
    """

    def __init__(
        self,
        cfg: KnowledgeBaseConfig,
        settings: ServiceSettings | None = None,
        *,
        es: Any,
        redis: Any,
        embed_model: BaseEmbedding,
        connector: Connector,
        processor: KbProcessor | None = None,
        names: KbIndexNames | None = None,
        index: str | None = None,
        allow_mass_delete: bool = False,
        lock_provider: LockProvider | None = None,
        ledger: RunCostLedger | None = None,
        run_id: str | None = None,
    ) -> None:
        if ledger is not None and run_id is not None and ledger.run_id != run_id:
            raise ValueError(f"run_id {run_id!r} differs from the ledger's {ledger.run_id!r}")
        self.cfg = cfg
        self.settings = settings if settings is not None else ServiceSettings()
        self.es = es
        self.redis = redis
        self.embed_model = embed_model
        self.embed_id = embed_model_id(embed_model)
        self.connector = connector
        self.processor = processor or KbProcessor(cfg, settings=self.settings)
        self.names = names or KbIndexNames(cfg.kb)
        if self.names.kb != cfg.kb:
            raise ValueError(f"index names are for kb {self.names.kb!r}, not {cfg.kb!r}")
        self.allow_mass_delete = allow_mass_delete
        self.lock_provider = lock_provider
        self.ledger = ledger
        self.run_id = ledger.run_id if ledger is not None else (run_id or new_run_id())
        self.generation_prefix = f"{DEFAULT_KEY_PREFIX}{self.names.prefix}"
        self.index: str | None = index
        self._started = 0.0
        self._started_at = ""
        self._new_fields: list[str] = []
        self._docstore: JsonFileDocumentStore | None = None
        self._store: EsVectorStore | None = None
        self._plan: _Plan | None = None

    # ---- the whole run ----------------------------------------------------------------------

    def run(self) -> IngestReport:
        locks = self.lock_provider or make_lock_provider(self.settings.lock_url)
        name = f"ingest:{self.names.alias}"
        with ExitStack() as stack:
            try:
                stack.enter_context(locks.lock(name, timeout=self.settings.ingest_lock_timeout_s))
            except LockTimeoutError as exc:
                raise IngestError(
                    f"another ingest of {self.names.alias} is running ({exc})"
                ) from exc
            self.prepare()
            self.write()
            return self.commit()

    # ---- phase 1: read, clean, compare, check -----------------------------------------------

    def prepare(self) -> _Plan:
        self._started = time.perf_counter()
        self._started_at = datetime.now(UTC).isoformat(timespec="seconds")
        body = index_body_for_kb(self.cfg, dims=self.settings.embed_dim)
        self.index = self._resolve_index(body)
        self._new_fields = ensure_index_fields(self.es, self.index, body)
        recorded = index_embed_model(self.es, self.index)
        if recorded is not None and recorded != self.embed_id:
            raise IndexModelMismatchError(
                f"{self.index} holds vectors of {recorded}, this run embeds with {self.embed_id}; "
                "vectors of two models cannot be compared. Build a new index version (s7-5) or "
                "use another --index-prefix"
            )
        docstore = make_docstore(self.settings.docstore_url, self.index)
        assert isinstance(docstore, JsonFileDocumentStore)
        self._docstore = docstore

        sources = list(self.connector.documents())
        read = self.processor.read(sources)
        no_pipeline = sorted(d for d, why in read.skipped.items() if why == "no_pipeline")
        eligible = len(sources) - len(no_pipeline)
        ratio = self.settings.ingest_max_read_failure_ratio
        if eligible and len(read.errors) / eligible > ratio:
            first = next(iter(read.errors.items()))
            raise ReadFailureError(
                f"{len(read.errors)} of {eligible} documents failed to read (more than "
                f"{ratio:.0%}); nothing was changed. First: {first[0]}: {first[1]}"
            )
        documents = self.processor.clean(read.documents)
        skipped_by_cleaning: dict[str, str] = {}
        for _rule, stats in self.processor.rule_stats():
            skipped_by_cleaning.update(stats.skipped)
        kept_old: list[str] = []
        for doc_id in read.errors:
            stored = docstore.get_document(doc_id, raise_error=False)
            if isinstance(stored, Document):
                documents.append(stored)
                kept_old.append(doc_id)

        ids = [d.id_ for d in documents]
        if len(set(ids)) != len(ids):
            raise IngestError("the connector delivered a document id twice")
        stored_ids = set(docstore.get_all_document_hashes().values())
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        for doc in documents:
            stored_hash = docstore.get_document_hash(doc.id_)
            if stored_hash is None:
                added.append(doc.id_)
            elif stored_hash != doc.hash:
                updated.append(doc.id_)
            else:
                unchanged.append(doc.id_)
        deleted = sorted(stored_ids - set(ids))

        min_ratio = self.settings.ingest_min_input_ratio
        if (
            stored_ids
            and len(documents) < min_ratio * len(stored_ids)
            and not self.allow_mass_delete
        ):
            raise MassDeleteError(
                f"this run would pass {len(documents)} documents, fewer than {min_ratio:.0%} of "
                f"the {len(stored_ids)} in the docstore: a possible mass delete (did the source "
                "fail to load?). Nothing was changed; rerun with --allow-mass-delete if the "
                "source really shrank"
            )
        if self.ledger is not None:
            changed = set(added) | set(updated)
            # Characters bound the tokens from above (as in the embedding client); each chunk
            # also embeds its title path. Cache hits make the real cost lower.
            tokens = sum(len(d.text) + 200 for d in documents if d.id_ in changed)
            self.ledger.check_budget(self.ledger.estimate(self.settings.embed_model, tokens))

        versions = {s.doc_id: s.source_version for s in sources}
        fields = {
            d.id_: WriteFields(
                source_version=versions.get(d.id_, ""), content_hash=content_hash(d.text)
            )
            for d in documents
        }
        self._plan = _Plan(
            sources=len(sources),
            documents=documents,
            fields=fields,
            added=added,
            updated=updated,
            unchanged=unchanged,
            deleted=deleted,
            read_failures=dict(read.errors),
            kept_old_version=kept_old,
            no_pipeline=no_pipeline,
            skipped_by_cleaning=skipped_by_cleaning,
        )
        return self._plan

    def _resolve_index(self, body: dict[str, Any]) -> str:
        if self.index is not None:
            if self.names.version_of(self.index) is None:
                raise ValueError(f"{self.index!r} is not a version of {self.names.alias}")
            if not self.es.indices.exists(index=self.index):
                raise IngestError(f"index {self.index} does not exist")
            return self.index
        current = aliased_indices(self.es, self.names)
        if len(current) > 1:
            raise IngestError(f"{self.names.alias} points to several indices: {current}")
        if current:
            return current[0]
        index, _created = init_kb(self.es, self.names, body)
        return index

    # ---- phase 2: write new chunks ----------------------------------------------------------

    def write(self) -> None:
        plan, docstore = self._plan, self._docstore
        if plan is None or docstore is None or self.index is None:
            raise RuntimeError("call prepare() before write()")
        if index_embed_model(self.es, self.index) is None:
            set_index_embed_model(self.es, self.index, self.embed_id)
        store = EsVectorStore(
            self.es, self.index, bulk_batch_size=self.settings.ingest_bulk_batch_size
        )
        store.begin_run(self.run_id, plan.fields)
        self._store = store
        pipeline = IngestionPipeline(
            transformations=[KbChunker(self.processor), self.embed_model],
            docstore=docstore,
            vector_store=store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            disable_cache=True,
        )
        # store_doc_text stays True (the default): read failures are replaced by the stored
        # text, and LlamaIndex's delete_document fails without it.
        pipeline.run(documents=plan.documents)
        store.refresh()
        expected = set(plan.updated) | set(plan.deleted)
        if set(store.pending_deletes) != expected:
            raise IngestError(
                "LlamaIndex asked to delete other documents than planned "
                f"({sorted(set(store.pending_deletes) ^ expected)[:5]}); nothing was deleted and "
                "the docstore was not saved"
            )

    # ---- phase 3: delete old chunks, raise the generation, save -----------------------------

    def commit(self) -> IngestReport:
        plan, docstore, store = self._plan, self._docstore, self._store
        if plan is None or docstore is None or store is None or self.index is None:
            raise RuntimeError("call prepare() and write() before commit()")
        finalized = store.finalize()
        changed = bool(plan.added or plan.updated or plan.deleted)
        if changed:
            generation = incr_data_generation(
                self.redis, self.cfg.kb, key_prefix=self.generation_prefix
            )
        else:
            generation = get_data_generation(
                self.redis, self.cfg.kb, key_prefix=self.generation_prefix
            )
        docstore.persist()
        report = IngestReport(
            run_id=self.run_id,
            kb=self.cfg.kb,
            index=self.index,
            embed_model=self.embed_id,
            started_at=self._started_at,
            duration_s=time.perf_counter() - self._started,
            sources=plan.sources,
            added=plan.added,
            updated=plan.updated,
            unchanged=plan.unchanged,
            deleted=plan.deleted,
            chunks_written=sum(store.written_docs.values()),
            chunks_deleted=finalized.deleted_chunks,
            data_generation=generation,
            generation_raised=changed,
            cost_cny=self.ledger.run_cny if self.ledger is not None else 0.0,
            docstore=docstore.path.as_posix(),
            read_failures=plan.read_failures,
            kept_old_version=plan.kept_old_version,
            no_pipeline=plan.no_pipeline,
            skipped_by_cleaning=plan.skipped_by_cleaning,
            new_index_fields=self._new_fields,
        )
        path = Path(self.settings.ingest_report_dir) / self.cfg.kb / f"ingest-{self.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), "utf-8")
        report.report_path = path.as_posix()
        return report


# ---- command line ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest.job",
        description="Ingest a knowledge base into Elasticsearch (incremental, write then delete).",
    )
    parser.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N documents (default: the YAML's limit)")
    size.add_argument("--all", action="store_true", help="every document (no limit)")
    parser.add_argument(
        "--fake-embed",
        action="store_true",
        help=f"FakeEmbedding instead of the gateway (no cost); index prefix {FAKE_INDEX_PREFIX!r} "
        "unless --index-prefix is given",
    )
    parser.add_argument(
        "--index-prefix",
        default=None,
        help="prefix of index, alias, docstore and data generation names (default: none, or "
        f"{FAKE_INDEX_PREFIX!r} with fake embeddings)",
    )
    parser.add_argument(
        "--allow-mass-delete",
        action="store_true",
        help="run although fewer than half of the docstore's documents were read",
    )
    return parser


def _close(embed: BaseEmbedding | None) -> None:
    inner = getattr(embed, "inner", embed)
    close = getattr(inner, "close", None)
    if callable(close):
        close()


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: ServiceSettings | None = None,
    es: Any = None,
    redis: Any = None,
    connector: Connector | None = None,
) -> int:
    """CLI entry; `settings`, `es`, `redis` and `connector` are for tests."""
    args = _parser().parse_args(argv)
    settings = settings if settings is not None else ServiceSettings()
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be a positive integer", file=sys.stderr)
        return 2
    try:
        configs = load_kb_configs(settings.kb_dir)
    except Exception as exc:  # a broken YAML is a usage error, reported in one line
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.kb not in configs:
        print(
            f"unknown knowledge base {args.kb!r}; known in {settings.kb_dir}/: {sorted(configs)}",
            file=sys.stderr,
        )
        return 2
    cfg = configs[args.kb]
    if args.fake_embed:
        settings = settings.model_copy(update={"embed_backend": "fake"})
    fake = settings.embed_backend == "fake"
    prefix = args.index_prefix
    if prefix is None:
        prefix = FAKE_INDEX_PREFIX if fake else ""
    try:
        names = KbIndexNames(cfg.kb, prefix=prefix)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_id = new_run_id()
    ledger = None if fake else RunCostLedger.from_settings(settings, run_id=run_id)
    own_es = es is None
    embed: BaseEmbedding | None = None
    try:
        if connector is None:
            extra = dict(cfg.source.model_extra or {})
            if args.all:
                extra["limit"] = None
            elif args.limit is not None:
                extra["limit"] = args.limit
            connector = connectors.create(
                cfg.source.connector, kb=cfg.kb, settings=settings, **extra
            )
        embed = make_embedding("ingest", settings, ledger=ledger)
        if es is None:
            es = make_es_client(settings.es_url, request_timeout=120.0)
        if redis is None:
            redis = make_redis_client(settings.redis_url)
        job = IngestJob(
            cfg,
            settings,
            es=es,
            redis=redis,
            embed_model=embed,
            connector=connector,
            names=names,
            allow_mass_delete=args.allow_mass_delete,
            ledger=ledger,
            run_id=run_id,
        )
        print(
            f"ingest {run_id}: kb {cfg.kb}, alias {names.alias}, embeddings {job.embed_id}, "
            f"Elasticsearch {mask_url(settings.es_url)}"
        )
        report = job.run()
    except (IngestError, BudgetExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # one line instead of a traceback; nothing was saved
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        _close(embed)
        if own_es and es is not None:
            es.close()
    for line in report.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
