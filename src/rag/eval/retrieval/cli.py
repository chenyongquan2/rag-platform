"""Command line of the retrieval eval (spec P0-10).

uv run python -m rag.eval.retrieval export-hf [--manifest P] [--out P] [--no-download]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rag.config import repo_path
from rag.data.hf import load_eval_questions, load_manifest
from rag.eval.questions import labels_from_hf_eval, save_question_set

DEFAULT_MANIFEST = "data/manifests/hf_docs.json"


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
    parser = argparse.ArgumentParser(prog="python -m rag.eval.retrieval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export-hf", help="write the HF eval set as a labelled question set")
    p_export.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_export.add_argument("--out", help="default: evals/datasets/<kb>/<eval dataset name>.jsonl")
    p_export.add_argument(
        "--no-download", action="store_true", help="use the cached copy only (fail if missing)"
    )

    args = parser.parse_args(argv)
    if args.command == "export-hf":
        return _export_hf(args)
    raise AssertionError(args.command)
