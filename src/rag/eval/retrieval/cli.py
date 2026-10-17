"""Command line of the retrieval eval (spec P0-10).

uv run python -m rag.eval.retrieval compare <run_a> <run_b> [--list final] [--level page]
    [--metric mrr@10] [--iterations 10000] [--seed 0] [--alpha 0.05] [--out FILE]
uv run python -m rag.eval.retrieval export-hf [--manifest P] [--out P] [--no-download]

A run is a run id (looked up under evals/reports/public and private), its directory, or its
results.json.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag.config import repo_path
from rag.data.hf import load_eval_questions, load_manifest
from rag.eval.questions import labels_from_hf_eval, save_question_set
from rag.eval.retrieval.compare import compare_runs, render_comparison
from rag.eval.retrieval.record import FINAL
from rag.eval.retrieval.runner import REPORTS_ROOT, load_run

DEFAULT_MANIFEST = "data/manifests/hf_docs.json"


def _compare(args: argparse.Namespace) -> int:
    a = load_run(args.run_a, reports_root=args.reports_dir)
    b = load_run(args.run_b, reports_root=args.reports_dir)
    comparison = compare_runs(
        a,
        b,
        list_name=args.list,
        level=args.level,
        metric=args.metric,
        iterations=args.iterations,
        seed=args.seed,
        alpha=args.alpha,
    )
    text = render_comparison(comparison)
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _export_hf(args: argparse.Namespace) -> int:
    manifest = load_manifest(repo_path(args.manifest))
    if manifest.eval is None:
        raise SystemExit(f"{args.manifest}: manifest has no eval dataset")
    out = args.out or (
        f"evals/datasets/{manifest.kb}/{manifest.eval.repo_id.split('/', 1)[1]}.jsonl"
    )
    questions = labels_from_hf_eval(load_eval_questions(manifest, download=not args.no_download))
    sha = save_question_set(questions, out)
    print(f"wrote {len(questions)} questions to {out} (sha256 {sha})")
    print("page labels are the source documents; `python -m rag.data.hf verify` checks they exist")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.retrieval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="McNemar and permutation tests between two runs")
    p_compare.add_argument("run_a")
    p_compare.add_argument("run_b")
    p_compare.add_argument("--list", default=FINAL, help="final (default) or a route: bm25, vector")
    p_compare.add_argument("--level", default="page", choices=["page", "chunk"])
    p_compare.add_argument("--metric", help="lists questions changed on it (default: the MRR)")
    p_compare.add_argument("--iterations", type=int, default=10_000)
    p_compare.add_argument("--seed", type=int, default=0)
    p_compare.add_argument("--alpha", type=float, default=0.05)
    p_compare.add_argument("--reports-dir", default=REPORTS_ROOT)
    p_compare.add_argument("--out", help="also write the Markdown to this file")

    p_export = sub.add_parser("export-hf", help="write the HF eval set as a labelled question set")
    p_export.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_export.add_argument("--out", help="default: evals/datasets/<kb>/<eval dataset name>.jsonl")
    p_export.add_argument(
        "--no-download", action="store_true", help="use the cached copy only (fail if missing)"
    )

    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(args)
    if args.command == "export-hf":
        return _export_hf(args)
    raise AssertionError(args.command)
