"""Command line of the result cache (spec P5-01).

    uv run python -m rag.cache drop --kb hf_docs        # remove one knowledge base's L1 entries
    uv run python -m rag.cache shadow-report            # L2 shadow records -> would-hit, wrong-hit
    uv run python -m rag.cache pairs build|eval         # the counter-example test set
    uv run python -m rag.cache bench                    # L1 workload replay (needs the services)

`drop` is for an operator who wants a knowledge base's cache gone at once (a wrong entry, a
retracted document). The normal invalidation needs no command: a new data generation or config
version changes every key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.cache.exact import ExactCache
from rag.cache.factory import cache_config, make_cache_backend
from rag.cache.layer import ResultCache
from rag.cache.store import CacheBackend, CacheUnavailable
from rag.config import ServiceSettings, repo_path

_KB = re.compile(r"[a-z0-9][a-z0-9_-]*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.cache", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    drop = sub.add_parser("drop", help="remove every L1 entry of one knowledge base")
    drop.add_argument("--kb", required=True)
    report = sub.add_parser("shadow-report", help="would-hit and wrong-hit rates of L2 shadow mode")
    report.add_argument(
        "--records", action="append", help="shadow JSONL file (default: the settings' file)"
    )
    report.add_argument(
        "--gold", action="append", default=[], help="question set with gold pages (repeatable)"
    )
    report.add_argument("--json", help="also write the report as JSON to this file")
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
        return _drop(args.kb, settings, backend)
    if args.command == "shadow-report":
        return _shadow_report(args, settings)
    return 2


def _drop(kb: str, settings: ServiceSettings, backend: CacheBackend | None) -> int:
    if not _KB.fullmatch(kb):
        print(f"invalid knowledge base name: {kb!r}", file=sys.stderr)
        return 2
    layer = ResultCache(
        ExactCache(backend or make_cache_backend(settings), cache_config(settings)), settings
    )
    try:
        count = layer.drop_kb(kb)
    except CacheUnavailable as exc:
        print(f"cache store unavailable: {exc.reason}", file=sys.stderr)
        return 1
    finally:
        layer.close()
    print(f"dropped {count} keys of knowledge base {kb}")
    return 0


def _shadow_report(args: argparse.Namespace, settings: ServiceSettings) -> int:
    from rag.cache.report import load_gold, load_records, render_report, shadow_report

    paths = args.records or [repo_path(settings.cache_shadow_dir) / "shadow.jsonl"]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"no shadow records at {missing[0]}", file=sys.stderr)
        return 1
    report = shadow_report(load_records(paths), load_gold(args.gold))
    print(render_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    return 0
