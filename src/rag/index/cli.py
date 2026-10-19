"""Knowledge-base indices from the command line.

    uv run python -m rag.index.cli init   --kb hf_docs              # v1 + alias, once
    uv run python -m rag.index.cli status --kb hf_docs              # alias, versions, generation
    uv run python -m rag.index.cli switch --kb hf_docs --version 2  # move alias, generation + 1
    uv run python -m rag.index.cli rebuild --kb hf_docs             # v(N+1), gate, switch (s7-5)
    uv run python -m rag.index.cli rollback --kb hf_docs            # alias back to the previous

`rebuild` writes the whole knowledge base into the next version, syncs once more, force-merges,
runs the gate (by default only a chunk-count sanity floor; the retrieval eval is P2-03) and
moves the alias only if the gate passes; any failure before that leaves the alias alone.
`--fake-embed` works on the `fake_` prefixed names, like the ingest job. `rollback` moves the
alias back to the index recorded in the live index's `_meta` and deletes nothing.

Connection strings come from ServiceSettings (env / .env); the analyzer from the knowledge
base's YAML (`es.analyzer`), the vector size from `EMBED_DIM`. Exit codes: 0 ok, 1 failed,
2 usage error (unknown knowledge base, bad arguments).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from elastic_transport import TransportError
from elasticsearch import ApiError
from redis.exceptions import RedisError

from rag.config import ServiceSettings, load_kb_configs
from rag.config.settings import mask_url
from rag.embed import make_embedding
from rag.factories import make_es_client, make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation, incr_data_generation
from rag.index.manage import (
    AliasConflictError,
    IndexStateError,
    aliased_indices,
    index_meta,
    init_kb,
    list_versions,
    switch_alias,
)
from rag.index.mapping import index_body_for_kb
from rag.index.names import KbIndexNames
from rag.index.rebuild import (
    PREVIOUS_INDEX_KEY,
    GateContext,
    GateResult,
    GenerationNotRaisedError,
    RebuildError,
    RollbackError,
    chunk_count_gate,
    rebuild_kb,
    rollback_kb,
)
from rag.ingest.job import FAKE_INDEX_PREFIX, Connector, RunCostLedger
from rag.registry import connectors

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.index.cli", description="Manage knowledge-base indices."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    common.add_argument(
        "--index-prefix",
        default=None,
        help="prefix for index, alias and data generation key names (default: none, or "
        f"{FAKE_INDEX_PREFIX!r} for `rebuild --fake-embed`)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "init", parents=[common], help="create <alias>_v1 and point the alias at it, once"
    )
    sub.add_parser(
        "status", parents=[common], help="show alias, versions, document counts, data generation"
    )
    switch = sub.add_parser(
        "switch",
        parents=[common],
        help="point the alias at another version, then add 1 to the data generation",
    )
    switch.add_argument("--version", type=int, required=True, help="target version N of _vN")
    rebuild = sub.add_parser(
        "rebuild",
        parents=[common],
        help="write the next version, sync, force-merge, gate, then switch the alias",
    )
    size = rebuild.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N documents (default: the YAML's limit)")
    size.add_argument("--all", action="store_true", help="every document (no limit)")
    rebuild.add_argument(
        "--fake-embed", action="store_true", help="FakeEmbedding instead of the gateway (no cost)"
    )
    rebuild.add_argument(
        "--version",
        type=int,
        help="resume this unaliased version (or create it) instead of the next free one",
    )
    gate = rebuild.add_mutually_exclusive_group()
    gate.add_argument(
        "--min-chunk-ratio",
        type=float,
        help="gate: the new index needs at least this share of the old index's chunks "
        "(default: REBUILD_MIN_CHUNK_RATIO)",
    )
    gate.add_argument("--no-gate", action="store_true", help="switch without any gate")
    rebuild.add_argument(
        "--allow-mass-delete", action="store_true", help="see `python -m rag.ingest.job --help`"
    )
    rebuild.add_argument(
        "--allow-unreadable-docs",
        action="store_true",
        help="switch although pages of the old index cannot be read now (they are dropped)",
    )
    rollback = sub.add_parser(
        "rollback",
        parents=[common],
        help="point the alias back at the previous index, then add 1 to the data generation",
    )
    rollback.add_argument(
        "--to-version", type=int, help="go to this version instead of the recorded previous one"
    )
    return parser


def _status(
    es: Elasticsearch, names: KbIndexNames, settings: ServiceSettings, key_prefix: str, redis: Any
) -> None:
    current = aliased_indices(es, names)
    print(f"{names.alias} -> {', '.join(current) if current else '(no alias)'}")
    if len(current) == 1:
        previous = index_meta(es, current[0]).get(PREVIOUS_INDEX_KEY)
        if previous:
            print(f"rollback target: {previous} (python -m rag.index.cli rollback)")
    versions = list_versions(es, names)
    if not versions:
        print("no versions")
    for version, index in versions.items():
        docs = es.count(index=index)["count"]
        mark = " (aliased)" if index in current else ""
        print(f"v{version} {index} docs={docs}{mark}")
    try:
        redis = redis or make_redis_client(settings.redis_url)
        generation = get_data_generation(redis, names.kb, key_prefix=key_prefix)
        print(f"data generation: {generation}")
    except RedisError as exc:
        print(f"data generation: unavailable ({exc})")


def _rebuild(
    args: argparse.Namespace,
    settings: ServiceSettings,
    cfg: Any,
    names: KbIndexNames,
    es: Elasticsearch,
    redis: Any,
    connector: Connector | None,
) -> int:
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be a positive integer", file=sys.stderr)
        return 2
    ratio = args.min_chunk_ratio
    if ratio is not None and not 0 <= ratio <= 1:
        print("error: --min-chunk-ratio must be between 0 and 1", file=sys.stderr)
        return 2
    if args.fake_embed:
        settings = settings.model_copy(update={"embed_backend": "fake"})
    fake = settings.embed_backend == "fake"
    if connector is None:
        extra = dict(cfg.source.model_extra or {})
        if args.all:
            extra["limit"] = None
        elif args.limit is not None:
            extra["limit"] = args.limit
        connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)

    def embed_factory(run_id: str) -> tuple[Any, RunCostLedger | None]:
        ledger = None if fake else RunCostLedger.from_settings(settings, run_id=run_id)
        return make_embedding("ingest", settings, ledger=ledger), ledger

    if args.no_gate:

        def gate(_ctx: GateContext) -> GateResult:
            return GateResult(True, "no gate (--no-gate)")

    else:
        gate = chunk_count_gate(ratio if ratio is not None else settings.rebuild_min_chunk_ratio)
    try:
        report = rebuild_kb(
            cfg,
            settings,
            es=es,
            redis=redis,
            connector=connector,
            names=names,
            embed_factory=embed_factory,
            gate=gate,
            version=args.version,
            allow_mass_delete=args.allow_mass_delete,
            allow_unreadable_docs=args.allow_unreadable_docs,
        )
    except RebuildError as exc:
        if exc.report is not None:
            for line in exc.report.summary_lines():
                print(line)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except GenerationNotRaisedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in report.summary_lines():
        print(line)
    return 0


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
    configs = load_kb_configs(settings.kb_dir)
    if args.kb not in configs:
        print(
            f"unknown knowledge base {args.kb!r}; known in {settings.kb_dir}/: {sorted(configs)}",
            file=sys.stderr,
        )
        return 2
    prefix = args.index_prefix
    if prefix is None:
        fake = args.command == "rebuild" and (args.fake_embed or settings.embed_backend == "fake")
        prefix = FAKE_INDEX_PREFIX if fake else ""
    try:
        names = KbIndexNames(args.kb, prefix=prefix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    key_prefix = f"{DEFAULT_KEY_PREFIX}{prefix}"
    print(f"Elasticsearch {mask_url(settings.es_url)}, Redis {mask_url(settings.redis_url)}")
    own_es = es is None
    if es is None:
        # A rebuild's bulk writes and delete_by_query calls are as long as an ingest's.
        timeout = 120.0 if args.command == "rebuild" else 30.0
        es = make_es_client(settings.es_url, request_timeout=timeout)
    try:
        if args.command == "init":
            body = index_body_for_kb(configs[args.kb], dims=settings.embed_dim)
            index, created = init_kb(es, names, body)
            if created:
                print(f"created {index}; {names.alias} -> {index}")
            else:
                print(f"already initialized: {names.alias} -> {index}")
        elif args.command == "status":
            _status(es, names, settings, key_prefix, redis)
        elif args.command == "switch":
            target = names.index(args.version)
            previous = switch_alias(es, names, target)
            print(f"{names.alias}: {previous or '(no alias)'} -> {target}")
            try:
                client = redis or make_redis_client(settings.redis_url)
                generation = incr_data_generation(client, names.kb, key_prefix=key_prefix)
            except RedisError as exc:
                print(
                    f"error: the alias was switched but the data generation was not increased "
                    f"({exc}); caches may serve old results. Run the same switch again once "
                    "Redis is up.",
                    file=sys.stderr,
                )
                return 1
            print(f"data generation: {generation}")
        elif args.command == "rebuild":
            return _rebuild(
                args,
                settings,
                configs[args.kb],
                names,
                es,
                redis or make_redis_client(settings.redis_url),
                connector,
            )
        elif args.command == "rollback":
            result = rollback_kb(
                settings,
                es=es,
                redis=redis or make_redis_client(settings.redis_url),
                names=names,
                to_version=args.to_version,
            )
            print(
                f"{result.alias}: {result.from_index} -> {result.to_index} (rolled back; "
                f"{result.from_index} is kept)"
            )
            print(f"data generation: {result.generation}")
            if not result.recorded:
                print(
                    "warning: the roll back could not be written to the indices' _meta; "
                    "`status` will not show it",
                    file=sys.stderr,
                )
    except (ValueError, AliasConflictError, IndexStateError, RollbackError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except GenerationNotRaisedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (TransportError, ApiError, RedisError) as exc:
        print(
            f"error: {type(exc).__name__}: {exc}. Are the services up? scripts/dev_up.sh",
            file=sys.stderr,
        )
        return 1
    finally:
        if own_es:
            es.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
