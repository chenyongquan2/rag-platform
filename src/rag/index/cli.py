"""Knowledge-base indices from the command line.

    uv run python -m rag.index.cli init   --kb hf_docs              # v1 + alias, once
    uv run python -m rag.index.cli status --kb hf_docs              # alias, versions, generation
    uv run python -m rag.index.cli switch --kb hf_docs --version 2  # move alias, generation + 1

Connection strings come from ServiceSettings (env / .env); the analyzer from the knowledge
base's YAML (`es.analyzer`), the vector size from `EMBED_DIM`. Exit codes: 0 ok, 1 failed,
2 usage error (unknown knowledge base, bad arguments).
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from elastic_transport import TransportError
from elasticsearch import ApiError
from redis.exceptions import RedisError

from rag.config import ServiceSettings, load_kb_configs
from rag.config.settings import mask_url
from rag.factories import make_es_client, make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation, incr_data_generation
from rag.index.manage import (
    AliasConflictError,
    IndexStateError,
    aliased_indices,
    init_kb,
    list_versions,
    switch_alias,
)
from rag.index.mapping import index_body_for_kb
from rag.index.names import KbIndexNames

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
        default="",
        help="prefix for index, alias and data generation key names (used by tests)",
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
    return parser


def _status(
    es: Elasticsearch, names: KbIndexNames, settings: ServiceSettings, key_prefix: str
) -> None:
    current = aliased_indices(es, names)
    print(f"{names.alias} -> {', '.join(current) if current else '(no alias)'}")
    versions = list_versions(es, names)
    if not versions:
        print("no versions")
    for version, index in versions.items():
        docs = es.count(index=index)["count"]
        mark = " (aliased)" if index in current else ""
        print(f"v{version} {index} docs={docs}{mark}")
    try:
        redis = make_redis_client(settings.redis_url)
        generation = get_data_generation(redis, names.kb, key_prefix=key_prefix)
        print(f"data generation: {generation}")
    except RedisError as exc:
        print(f"data generation: unavailable ({exc})")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = ServiceSettings()
    configs = load_kb_configs(settings.kb_dir)
    if args.kb not in configs:
        print(
            f"unknown knowledge base {args.kb!r}; known in {settings.kb_dir}/: {sorted(configs)}",
            file=sys.stderr,
        )
        return 2
    try:
        names = KbIndexNames(args.kb, prefix=args.index_prefix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    key_prefix = f"{DEFAULT_KEY_PREFIX}{args.index_prefix}"
    print(f"Elasticsearch {mask_url(settings.es_url)}, Redis {mask_url(settings.redis_url)}")
    es = make_es_client(settings.es_url)
    try:
        if args.command == "init":
            body = index_body_for_kb(configs[args.kb], dims=settings.embed_dim)
            index, created = init_kb(es, names, body)
            if created:
                print(f"created {index}; {names.alias} -> {index}")
            else:
                print(f"already initialized: {names.alias} -> {index}")
        elif args.command == "status":
            _status(es, names, settings, key_prefix)
        elif args.command == "switch":
            target = names.index(args.version)
            previous = switch_alias(es, names, target)
            print(f"{names.alias}: {previous or '(no alias)'} -> {target}")
            try:
                redis = make_redis_client(settings.redis_url)
                generation = incr_data_generation(redis, names.kb, key_prefix=key_prefix)
            except RedisError as exc:
                print(
                    f"error: the alias was switched but the data generation was not increased "
                    f"({exc}); caches may serve old results. Run the same switch again once "
                    "Redis is up.",
                    file=sys.stderr,
                )
                return 1
            print(f"data generation: {generation}")
    except (ValueError, AliasConflictError, IndexStateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (TransportError, ApiError, RedisError) as exc:
        print(
            f"error: {type(exc).__name__}: {exc}. Are the services up? scripts/dev_up.sh",
            file=sys.stderr,
        )
        return 1
    finally:
        es.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
