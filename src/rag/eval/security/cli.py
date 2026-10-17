"""Command line of the security eval (spec P1-03).

uv run python -m rag.eval.security run --answers FILE [--set DIR] [--out FILE]
uv run python -m rag.eval.security dry-run [--set DIR] [--tokenizer model|whitespace]
    [--out-dir DIR]
uv run python -m rag.eval.security scan --kb hf_docs [--limit N | --all] [--samples 20] [--seed 0]
    [--out-dir DIR]
uv run python -m rag.eval.security scan --pages DIR ...     # every *.md / *.mdx under DIR

`run` scores answers produced elsewhere (JSONL lines `{"qid", "answer": <Answer JSON>}` or
`{"qid", "text": "..."}`). Reports go under the git-ignored `data/reports/` by default.
Exit codes: 0 ok, 1 failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag.config import REPO_ROOT, ServiceSettings, load_kb_config, load_kb_configs, repo_path
from rag.data import SourceDocument, make_doc_id
from rag.eval.security.dataset import DEFAULT_SET_DIR, SecuritySetError, load_security_set
from rag.eval.security.harness import dry_run
from rag.eval.security.scan import render_scan, scan_sources
from rag.eval.security.score import (
    SecurityReport,
    read_answer_records,
    render_report,
    score_answers,
    write_answer_records,
)
from rag.registry import connectors

SECURITY_REPORTS = "data/reports/security"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_report(report: SecurityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=1), encoding="utf-8", newline="\n")
    path.with_suffix(".md").write_text(render_report(report), encoding="utf-8", newline="\n")


def _run(args: argparse.Namespace) -> int:
    sset = load_security_set(args.set)
    try:
        report = score_answers(sset, read_answer_records(args.answers))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else repo_path(f"{SECURITY_REPORTS}/run-{_stamp()}.json")
    _write_report(report, out)
    for line in report.summary_lines():
        print(line)
    print(f"report: {out}")
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    sset = load_security_set(args.set)
    kb_config = load_kb_config(
        REPO_ROOT / "kb" / "public" / "hf_docs.yaml", kb_root=REPO_ROOT / "kb"
    )
    tokenizer = str.split if args.tokenizer == "whitespace" else None
    compliant, followed = dry_run(sset, kb_config, tokenizer=tokenizer)
    out = Path(args.out_dir) if args.out_dir else repo_path(f"{SECURITY_REPORTS}/dry-{_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("compliant", compliant), ("followed", followed)):
        write_answer_records(records, out / f"answers-{name}.jsonl")
        report = score_answers(sset, records)
        path = out / f"report-{name}.json"
        path.write_text(report.model_dump_json(indent=1), encoding="utf-8", newline="\n")
        print(f"== {name} replies")
        for line in report.summary_lines():
            print(line)
    print(f"reports: {out}")
    return 0


def _page_sources(directory: Path) -> Iterator[SourceDocument]:
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in (".md", ".mdx") or not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        yield SourceDocument(
            doc_id=make_doc_id("pages", rel),
            kb="pages",
            source_rel=rel,
            source_url=None,
            ext=path.suffix.lower(),
            content=path.read_bytes(),
            source_version="local",
        )


def _scan(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.pages:
        sources: Iterator[SourceDocument] = _page_sources(Path(args.pages))
        name = "pages"
        default_dir = SECURITY_REPORTS
    else:
        configs = load_kb_configs(settings.kb_dir)
        cfg = configs.get(args.kb)
        if cfg is None:
            print(f"error: unknown knowledge base {args.kb!r}", file=sys.stderr)
            return 2
        extra = dict(cfg.source.model_extra or {})
        if args.all:
            extra["limit"] = None
        elif args.limit is not None:
            extra["limit"] = args.limit
        connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)
        sources = connector.documents()
        name = cfg.kb
        default_dir = f"data/reports/{cfg.kb}"
    report = scan_sources(sources, samples=args.samples, seed=args.seed)
    out = Path(args.out_dir) if args.out_dir else repo_path(default_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"injection-scan-{_stamp()}"
    base.with_suffix(".json").write_text(
        report.model_dump_json(indent=1), encoding="utf-8", newline="\n"
    )
    base.with_suffix(".md").write_text(render_scan(report, title=name), "utf-8", newline="\n")
    for line in report.summary_lines():
        print(line)
    print(f"report: {base}.json")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.eval.security", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score an answers file")
    run.add_argument("--answers", required=True, help="JSONL answers file")
    run.add_argument("--set", default=str(DEFAULT_SET_DIR), help="security set directory")
    run.add_argument("--out", help=f"JSON report (default {SECURITY_REPORTS}/run-<time>.json)")

    dry = sub.add_parser("dry-run", help="score scripted compliant and violating replies")
    dry.add_argument("--set", default=str(DEFAULT_SET_DIR), help="security set directory")
    dry.add_argument(
        "--tokenizer",
        choices=("model", "whitespace"),
        default="model",
        help="chunk tokenizer: the reranker's (local HF cache) or whitespace split",
    )
    dry.add_argument("--out-dir", help=f"output directory (default {SECURITY_REPORTS}/dry-<time>)")

    scan = sub.add_parser("scan", help="run the injection detector over a corpus")
    where = scan.add_mutually_exclusive_group(required=True)
    where.add_argument("--kb", help="knowledge base name (`kb` in its YAML)")
    where.add_argument("--pages", help="directory of Markdown pages")
    size = scan.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N documents (default: the YAML's limit)")
    size.add_argument("--all", action="store_true", help="every document")
    scan.add_argument("--samples", type=int, default=20, help="hits to sample for labelling")
    scan.add_argument("--seed", type=int, default=0, help="sampling seed")
    scan.add_argument("--out-dir", help="report directory (default data/reports/<kb>/)")
    return parser


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "dry-run":
            return _dry_run(args)
        if args.limit is not None and args.limit < 1:
            print("error: --limit must be a positive integer", file=sys.stderr)
            return 2
        return _scan(args, settings if settings is not None else ServiceSettings())
    except SecuritySetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # one line, not a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
