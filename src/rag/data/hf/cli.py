"""`python -m rag.data.hf download|verify`: fetch the HF data and check it against the manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.config import ServiceSettings
from rag.data.hf.connector import corpus_documents
from rag.data.hf.download import ensure_dataset, read_dataset
from rag.data.hf.evalset import eval_questions, missing_sources
from rag.data.hf.manifest import DatasetEntry, HfDataError, Manifest, load_manifest, save_manifest

DEFAULT_MANIFEST = "data/manifests/hf_docs.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.data.hf",
        description="Download the Hugging Face corpus and eval set and verify them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="fetch missing files, verify, print counts")
    verify = sub.add_parser("verify", help="verify cached files (no network), print counts")
    for p in (download, verify):
        p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="data manifest JSON")
        p.add_argument("--data-dir", help="cache directory (default: HF_DATA_DIR setting)")
    download.add_argument("--endpoint", help="HF endpoint (default: HF_ENDPOINT setting)")
    download.add_argument(
        "--pin",
        action="store_true",
        help="record size, sha256 and rows for manifest files that have none yet",
    )
    return parser


def _entries(manifest: Manifest) -> list[tuple[str, DatasetEntry]]:
    pairs = [("corpus", manifest.corpus)]
    if manifest.eval is not None:
        pairs.append(("eval", manifest.eval))
    return pairs


def _download(args: argparse.Namespace, settings: ServiceSettings, data_dir: str) -> int:
    manifest = load_manifest(args.manifest)
    endpoint = args.endpoint or settings.hf_endpoint
    print(f"endpoint: {endpoint}")
    print(f"data dir: {Path(data_dir).resolve()}")
    for role, entry in _entries(manifest):
        was_pinned = all(f.pinned for f in entry.files)
        for cached in ensure_dataset(
            entry,
            endpoint=endpoint,
            data_dir=data_dir,
            timeout_s=settings.hf_timeout_s,
            pin=args.pin,
        ):
            state = "downloaded" if cached.downloaded else "cached"
            print(
                f"{role} {entry.label} {cached.entry.path}: {state}, "
                f"{cached.entry.size} bytes, {cached.entry.rows} rows"
            )
        if args.pin and not was_pinned:
            save_manifest(manifest, args.manifest)
            print(f"pinned {entry.repo_id} in {args.manifest}")
    docs = list(corpus_documents(read_dataset(manifest.corpus, data_dir), manifest))
    print(f"docs: {len(docs)}")
    if manifest.eval is not None:
        print(f"questions: {len(eval_questions(read_dataset(manifest.eval, data_dir), manifest))}")
    return 0


def _verify(args: argparse.Namespace, data_dir: str) -> int:
    manifest = load_manifest(args.manifest)
    print(f"manifest: {args.manifest} (kb {manifest.kb})")
    print(f"data dir: {Path(data_dir).resolve()}")
    for role, entry in _entries(manifest):
        print(f"{role} {entry.repo_id} revision {entry.revision} license {entry.license}")
    docs = list(corpus_documents(read_dataset(manifest.corpus, data_dir), manifest))
    print(f"docs: {len(docs)} (manifest {manifest.corpus.rows})")
    if manifest.eval is None:
        return 0
    questions = eval_questions(read_dataset(manifest.eval, data_dir), manifest)
    print(f"questions: {len(questions)} (manifest {manifest.eval.rows})")
    missing = missing_sources(questions, (d.doc_id for d in docs))
    print(f"missing eval sources: {len(missing)}")
    for q in missing:
        print(f"  {q.qid} {q.source_doc}")
    return 1 if missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = ServiceSettings()
    data_dir = args.data_dir or settings.hf_data_dir
    try:
        if args.command == "download":
            return _download(args, settings, data_dir)
        return _verify(args, data_dir)
    except HfDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
