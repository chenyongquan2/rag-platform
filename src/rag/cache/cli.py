"""Command line of the result cache (spec P5-01).

    uv run python -m rag.cache drop --kb hf_docs        # remove one knowledge base's L1 entries
    uv run python -m rag.cache shadow-report            # L2 shadow records -> would-hit, wrong-hit
    uv run python -m rag.cache pairs build|eval         # the counter-example test set

`drop` is for an operator who wants a knowledge base's cache gone at once (a wrong entry, a
retracted document). The normal invalidation needs no command: a new data generation or config
version changes every key.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from rag.cache.exact import ExactCache
from rag.cache.factory import cache_config, make_cache_backend
from rag.cache.layer import ResultCache
from rag.cache.store import CacheBackend, CacheUnavailable
from rag.config import ServiceSettings

_KB = re.compile(r"[a-z0-9][a-z0-9_-]*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.cache", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    drop = sub.add_parser("drop", help="remove every L1 entry of one knowledge base")
    drop.add_argument("--kb", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: ServiceSettings | None = None,
    backend: CacheBackend | None = None,
) -> int:
    settings = settings if settings is not None else ServiceSettings()
    args = build_parser().parse_args(argv)
    if args.command == "drop":
        if not _KB.fullmatch(args.kb):
            print(f"invalid knowledge base name: {args.kb!r}", file=sys.stderr)
            return 2
        layer = ResultCache(
            ExactCache(backend or make_cache_backend(settings), cache_config(settings)), settings
        )
        try:
            count = layer.drop_kb(args.kb)
        except CacheUnavailable as exc:
            print(f"cache store unavailable: {exc.reason}", file=sys.stderr)
            return 1
        finally:
            layer.close()
        print(f"dropped {count} keys of knowledge base {args.kb}")
        return 0
    return 2
