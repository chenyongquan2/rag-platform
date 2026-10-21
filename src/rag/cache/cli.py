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
from rag.cache.pair_eval import Embedder
from rag.cache.store import CacheBackend, CacheUnavailable
from rag.config import ServiceSettings, repo_path

_KB = re.compile(r"[a-z0-9][a-z0-9_-]*")

PAIR_FILE = "evals/datasets/hf_docs/cache_pairs.jsonl"
PAIR_SOURCES: dict[str, list[str]] = {
    "dev": ["huggingface_doc_qa_eval.jsonl", "synthetic_dev.jsonl"],
    "holdout": ["synthetic_holdout.jsonl"],
}
"""Question sets the pairs are made from (`evals/datasets/hf_docs/`): the public 65 and the
synthetic dev set are the dev part, where the threshold is chosen; the synthetic holdout set is
read for the final report only. The MT5 questions are local-only and not used."""


def build_pair_sources() -> dict[str, list[tuple[str, str]]]:
    """`{split: [(qid, question)]}` read from the committed question sets."""
    root = repo_path("evals/datasets/hf_docs")
    sources: dict[str, list[tuple[str, str]]] = {}
    for split, names in PAIR_SOURCES.items():
        items: list[tuple[str, str]] = []
        for name in names:
            with (root / name).open(encoding="utf-8", newline="") as fh:
                for line in fh:
                    if line.strip():
                        data = json.loads(line)
                        items.append((data["qid"], data["question"]))
        sources[split] = items
    return sources


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
    report.add_argument(
        "--max-distance",
        type=float,
        help="read the records as if the threshold had been this (at most the recording one)",
    )
    bench = sub.add_parser("bench", help="measurements against the local services (P5-01)")
    bench.add_argument(
        "--step", required=True, choices=["overhead", "semantic", "replay", "ask", "down"]
    )
    bench.add_argument("--out", default="data/reports/p5-01")
    bench.add_argument("--requests", type=int, default=400, help="replay: number of requests")
    bench.add_argument("--questions", type=int, default=5, help="ask: number of questions")
    pairs = sub.add_parser("pairs", help="the counter-example test set")
    pairs_sub = pairs.add_subparsers(dest="pairs_command", required=True)
    build = pairs_sub.add_parser("build", help="write the pairs file from the question sets")
    build.add_argument("--out", default=PAIR_FILE)
    ev = pairs_sub.add_parser(
        "evaluate", help="L1 false hits, and the guard and the threshold of the semantic cache"
    )
    ev.add_argument("--pairs", default=PAIR_FILE)
    ev.add_argument("--out", default="data/reports/p5-01", help="directory of the JSON results")
    ev.add_argument("--estimate-only", action="store_true", help="print the embedding cost only")
    ev.add_argument(
        "--final",
        action="store_true",
        help="also evaluate the holdout part, once, at the threshold chosen on the dev part",
    )
    ev.add_argument(
        "--force", action="store_true", help="with --final: overwrite a final file that exists"
    )
    ev.add_argument("--slice", default="v1", choices=["v1", "v2"], help="which pair classes")
    ev.add_argument("--tag", default="", help="added to the name of the result file")
    ev.add_argument(
        "--threshold",
        type=float,
        help="use this distance threshold instead of choosing one on the dev part",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: ServiceSettings | None = None,
    backend: CacheBackend | None = None,
    embedder: Embedder | None = None,
) -> int:
    settings = settings if settings is not None else ServiceSettings()
    args = build_parser().parse_args(argv)
    if args.command == "drop":
        return _drop(args.kb, settings, backend)
    if args.command == "shadow-report":
        return _shadow_report(args, settings)
    if args.command == "bench":
        from rag.cache.measure import run

        result = run(args.step, repo_path(args.out), n=args.requests, questions=args.questions)
        print(json.dumps(result, indent=2)[:6000])
        return 0
    if args.command == "pairs" and args.pairs_command == "build":
        from rag.cache.pairs import build_pairs, write_pairs

        count = write_pairs(build_pairs(build_pair_sources()), repo_path(args.out))
        print(f"wrote {count} pairs to {args.out}")
        return 0
    if args.command == "pairs" and args.pairs_command == "evaluate":
        return _pairs_evaluate(args, settings, embedder)
    return 2


def _pairs_evaluate(
    args: argparse.Namespace, settings: ServiceSettings, embedder: Embedder | None
) -> int:
    from rag.cache.pair_embed import embed_pair_texts
    from rag.cache.pair_eval import (
        THRESHOLDS,
        choose_threshold,
        collect_texts,
        estimate_cost,
        evaluate_split,
        paraphrase_pairs,
        summary,
        sweep,
    )
    from rag.cache.pairs import evaluate_l1, read_pairs
    from rag.cost.ledger import BudgetExceeded, check_budget

    pairs = [p for p in read_pairs(repo_path(args.pairs)) if p.slice == args.slice]
    stem = "pairs_eval" + ("" if args.slice == "v1" else f"_{args.slice}")
    stem += f"_{args.tag}" if args.tag else ""
    out = repo_path(args.out)
    final_path = out / f"{stem}_final.json"
    if args.final and final_path.exists() and not args.force:
        print(
            f"{final_path} exists: the holdout part is evaluated once; --force overwrites it",
            file=sys.stderr,
        )
        return 4
    dev = [p for p in pairs if p.split == "dev"]
    holdout = [p for p in pairs if p.split == "holdout"]
    paraphrases = paraphrase_pairs(repo_path("evals/datasets/hf_docs"))
    texts = collect_texts(pairs, extra=paraphrases)
    estimate = estimate_cost(texts)
    print(f"{summary(pairs)}")
    print(f"embedding estimate: {estimate}")
    if args.estimate_only:
        return 0
    if embedder is None:
        try:
            check_budget(estimate["cny"] * 2, settings)
        except BudgetExceeded as exc:
            print(f"budget: {exc}", file=sys.stderr)
            return 3
        vectors, counts = embed_pair_texts(
            texts, settings, cache_path=repo_path("data/cache/cache_pairs.sqlite3")
        )
    else:
        vectors, counts = embedder(texts), {"fake": len(texts)}
    print(f"vectors from: {counts}")
    rows = sweep(dev, vectors, paraphrases, THRESHOLDS)
    threshold = args.threshold if args.threshold is not None else choose_threshold(rows)
    result: dict[str, object] = {
        "summary": summary(pairs),
        "l1": evaluate_l1(pairs),
        "embedding": {"estimate": estimate, "sources": counts},
        "dev_sweep": rows,
        "threshold": threshold,
        "dev": evaluate_split(dev, vectors, threshold),
    }
    target = out / f"{stem}.json"
    if args.final:
        result["holdout"] = evaluate_split(holdout, vectors, threshold)
        target = final_path
        if args.force and target.exists():
            print(f"overwriting {target} (--force)", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=list), encoding="utf-8")
    how = "given" if args.threshold is not None else "chosen on dev"
    print(f"threshold {how}: {threshold}; wrote {target}")
    return 0


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
    report = shadow_report(
        load_records(paths), load_gold(args.gold), max_distance=args.max_distance
    )
    print(render_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    return 0
