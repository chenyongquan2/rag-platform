"""Command line of the retrieval eval (specs P0-10, P0-12).

uv run python -m rag.eval.retrieval run --kb hf_docs [--rerank-n N | --no-rerank]
    [--routes bm25,vector] [--deadline-ms MS] [--top-k K] [--questions P] [--index-prefix P]
    [--label L] [--reports-dir D]
uv run python -m rag.eval.retrieval rerank-curve --kb hf_docs [--rerank-ns 10,20,30]
    [--warmup 1] [--out DIR]
uv run python -m rag.eval.retrieval compare <run_a> <run_b> [--list final] [--level page]
    [--metric mrr@10] [--iterations 10000] [--seed 0] [--alpha 0.05] [--out FILE]
uv run python -m rag.eval.retrieval export-hf [--manifest P] [--out P] [--no-download]

`run` builds the chain search_docs uses (`make_search_pipeline`; with `--no-rerank` the
fusion-only `make_hybrid_search`) from the KB config and the service settings, asks every
question of the KB's question set and writes the run record. `rerank-curve` measures the
reranker for each rerank_n on the fusion candidates. A run is a run id (looked up under
evals/reports/public and private), its directory, or its results.json.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rag.config import KnowledgeBaseConfig, ServiceSettings, repo_path
from rag.config.kb import is_public_kb_file, load_kb_config
from rag.data.hf import load_eval_questions, load_manifest
from rag.eval.questions import labels_from_hf_eval, load_question_set, save_question_set
from rag.eval.retrieval.compare import compare_runs, render_comparison
from rag.eval.retrieval.protocol import Retriever
from rag.eval.retrieval.record import FINAL, RunRecord
from rag.eval.retrieval.rerank_curve import (
    render_rerank_curve,
    rerank_latency_curve,
    write_rerank_curve,
)
from rag.eval.retrieval.runner import (
    REPORTS_ROOT,
    collect_provenance,
    load_run,
    make_run_id,
    run_dir,
    run_retrieval_eval,
    write_run,
)
from rag.retrieve.hybrid import ROUTES
from rag.types import Chunk

DEFAULT_MANIFEST = "data/manifests/hf_docs.json"

ChunkReorder = Callable[[str, Sequence[Chunk]], Sequence[Chunk]]


def _echo(text: str) -> None:
    """Print `text` without failing on a console that cannot show every character (a GBK
    console on Windows and the emoji in HF questions): those characters are replaced."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding), end="")


class UsageError(Exception):
    """Bad arguments or configuration: exit code 2, one line on stderr."""


# ---- run / rerank-curve: the real chain (these seams are replaced in unit tests) ----------------


def build_retriever(
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    *,
    rerank: bool,
    routes: Sequence[str],
    index_prefix: str,
) -> Retriever:
    """The chain search_docs uses (`make_search_pipeline`: dedupe, rerank, threshold), or with
    `rerank=False` the fusion-only chain (`make_hybrid_search`: dedupe only). Real ES, query
    embedding (L0 cache first, then the gateway), Redis for the data generation, the caller
    groups of the service (`service_acl_groups`)."""
    from rag.factories import make_redis_client
    from rag.retrieve.pipeline import make_search_pipeline
    from rag.retrieve.search import make_hybrid_search

    redis = make_redis_client(settings.redis_url)
    groups = settings.service_acl_groups
    if rerank:
        return make_search_pipeline(
            cfg, settings, acl_groups=groups, index_prefix=index_prefix, redis=redis, routes=routes
        )
    return make_hybrid_search(
        cfg, settings, acl_groups=groups, index_prefix=index_prefix, redis=redis, routes=routes
    )


def build_reranker(settings: ServiceSettings) -> ChunkReorder:
    """The process's cross-encoder without a deadline (`ChunkReranker`), for the curve."""
    from rag.retrieve.rerank import ChunkReranker, get_rerank_runner

    return ChunkReranker(get_rerank_runner(settings))


def reranker_revision(model: str) -> str | None:
    """The snapshot revision of `model` in the local HF cache (no network), or None."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, local_files_only=True)).name
    except Exception:  # provenance only: a missing cache must not stop the run
        return None


def _models(settings: ServiceSettings) -> dict[str, str]:
    models = {
        "embed": settings.embed_model,
        "embed_dim": str(settings.embed_dim),
        "reranker": settings.reranker_model,
    }
    revision = reranker_revision(settings.reranker_model)
    if revision:
        models["reranker_revision"] = revision
    return models


def _kb_config(settings: ServiceSettings, kb: str) -> tuple[KnowledgeBaseConfig, bool]:
    """The KB's config and whether it lives under `kb/public/` (decides the report directory)."""
    kb_dir = Path(settings.kb_dir)
    if not kb_dir.is_absolute():
        kb_dir = repo_path(kb_dir)
    known: list[str] = []
    for path in sorted([*kb_dir.rglob("*.yaml"), *kb_dir.rglob("*.yml")]):
        try:
            cfg = load_kb_config(path, kb_root=kb_dir)
        except Exception as exc:
            raise UsageError(f"{path}: {type(exc).__name__}: {exc}") from exc
        if cfg.kb == kb:
            return cfg, is_public_kb_file(path, kb_dir)
        known.append(cfg.kb)
    raise UsageError(f"unknown knowledge base {kb!r}; known in {kb_dir}: {sorted(known)}")


def _question_file(cfg: KnowledgeBaseConfig, given: str | None) -> Path:
    if given:
        return Path(given)
    eval_dir = repo_path(cfg.eval.dir or f"evals/datasets/{cfg.kb}")
    if cfg.eval.questions:
        path = eval_dir / cfg.eval.questions
        if not path.is_file():
            raise UsageError(f"{path}: eval.questions file not found")
        return path
    files = sorted(eval_dir.glob("*.jsonl"))
    if len(files) != 1:
        raise UsageError(f"{eval_dir}: expected one *.jsonl question set, found {len(files)}")
    return files[0]


def _routes(text: str | None) -> tuple[str, ...]:
    if not text:
        return ROUTES
    names = tuple(dict.fromkeys(n.strip() for n in text.split(",") if n.strip()))
    unknown = [n for n in names if n not in ROUTES]
    if unknown or not names:
        raise UsageError(f"--routes: unknown route(s) {unknown}; choose from {list(ROUTES)}")
    return tuple(r for r in ROUTES if r in names)


def _run_label(args: argparse.Namespace, cfg: KnowledgeBaseConfig, routes: tuple[str, ...]) -> str:
    if args.label:
        return str(args.label)
    parts = [] if routes == ROUTES else ["-".join(routes)]
    parts.append("norerank" if args.no_rerank else f"rerank{cfg.retrieval.rerank_n}")
    return "_".join(parts)


def _setup(
    args: argparse.Namespace, settings: ServiceSettings
) -> tuple[KnowledgeBaseConfig, bool, ServiceSettings, tuple[str, ...]]:
    """KB config (with `--rerank-n`), visibility, settings (with `--deadline-ms`), routes."""
    cfg, public = _kb_config(settings, args.kb)
    rerank_n = getattr(args, "rerank_n", None)
    if rerank_n is not None:
        if rerank_n < 1:
            raise UsageError("--rerank-n must be a positive integer")
        retrieval = cfg.retrieval.model_copy(update={"rerank_n": rerank_n})
        cfg = cfg.model_copy(update={"retrieval": retrieval})
    if args.deadline_ms is not None:
        if args.deadline_ms < 1:
            raise UsageError("--deadline-ms must be a positive integer")
        update: dict[str, Any] = {"search_deadline_ms": args.deadline_ms}
        if args.deadline_ms > settings.search_max_deadline_ms:
            update["search_max_deadline_ms"] = args.deadline_ms
        settings = settings.model_copy(update=update)
    return cfg, public, settings, _routes(args.routes)


def _close(retriever: object) -> None:
    close = getattr(retriever, "close", None)
    if callable(close):
        close()


def summary_table(record: RunRecord) -> str:
    """Page-level means per list, latency and counts (printed after a run)."""
    k = max(record.run.ks)
    cols = [*(f"hit@{x}" for x in record.run.ks), f"mrr@{k}", f"ndcg@{k}", f"recall@{k}"]
    lines = ["| list | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name in record.run.lists:
        page = record.summary.get(name, {}).get("page")
        if page is None:
            continue
        cells = [f"{page.means.get(c, 0.0):.3f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lat = record.latency
    lines.append(
        f"latency ms: P50 {lat.p50_ms:.0f}, P95 {lat.p95_ms:.0f}, max {lat.max_ms:.0f}; "
        f"counts: {record.counts}"
    )
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    cfg, public, settings, routes = _setup(args, settings)
    if args.top_k is not None and args.top_k < 1:
        raise UsageError("--top-k must be a positive integer")
    questions = load_question_set(_question_file(cfg, args.questions), kb=cfg.kb)
    label = _run_label(args, cfg, routes)
    retriever = build_retriever(
        cfg, settings, rerank=not args.no_rerank, routes=routes, index_prefix=args.index_prefix
    )
    try:
        record = run_retrieval_eval(
            retriever,
            questions,
            kb_config=cfg,
            top_k=args.top_k,
            label=label,
            provenance=collect_provenance(settings=settings, models=_models(settings)),
        )
    finally:
        _close(retriever)
    out = write_run(record, public=public, reports_root=args.reports_dir)
    print(f"wrote {out}")
    _echo(summary_table(record))
    return 0


def _rerank_curve(args: argparse.Namespace, settings: ServiceSettings) -> int:
    cfg, public, settings, routes = _setup(args, settings)
    try:
        ns = [int(n) for n in args.rerank_ns.split(",") if n.strip()]
    except ValueError as exc:
        raise UsageError(f"--rerank-ns: {exc}") from exc
    if not ns or min(ns) < 1:
        raise UsageError("--rerank-ns must list positive integers")
    questions = load_question_set(_question_file(cfg, args.questions), kb=cfg.kb)
    retriever = build_retriever(
        cfg, settings, rerank=False, routes=routes, index_prefix=args.index_prefix
    )
    try:
        curve = rerank_latency_curve(
            retriever,
            build_reranker(settings),
            questions,
            kb=cfg.kb,
            rerank_ns=ns,
            final_k=cfg.retrieval.final_k,
            warmup=args.warmup,
            provenance=collect_provenance(settings=settings, models=_models(settings)),
        )
    finally:
        _close(retriever)
    out_dir = args.out or run_dir(
        make_run_id(cfg.kb, "rerank-curve"), public=public, reports_root=args.reports_dir
    )
    out = write_rerank_curve(curve, out_dir)
    print(f"wrote {out}")
    _echo(render_rerank_curve(curve))
    return 0


# ---- compare / export-hf (spec P0-10) ----------------------------------------------------------


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
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    _echo(text)
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


def _add_chain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    p.add_argument("--questions", help="question set (default: the one *.jsonl in eval.dir)")
    p.add_argument("--routes", help=f"comma-separated routes to send (default: {','.join(ROUTES)})")
    p.add_argument("--deadline-ms", type=int, help="search deadline (default: the settings')")
    p.add_argument("--index-prefix", default="", help="e.g. fake_ for the fake-vector index")
    p.add_argument("--reports-dir", default=REPORTS_ROOT)


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """CLI entry; `settings` is for tests. Exit codes: 0 ok, 2 usage error."""
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.retrieval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the question set through the retrieval chain")
    _add_chain_args(p_run)
    rerank = p_run.add_mutually_exclusive_group()
    rerank.add_argument("--rerank-n", type=int, help="override retrieval.rerank_n")
    rerank.add_argument("--no-rerank", action="store_true", help="fusion and dedupe only")
    p_run.add_argument("--top-k", type=int, help="final list length (default: retrieval.final_k)")
    p_run.add_argument("--label", help="run id suffix (default: from the options)")

    p_curve = sub.add_parser("rerank-curve", help="rerank latency and nDCG per rerank_n")
    _add_chain_args(p_curve)
    p_curve.add_argument("--rerank-ns", default="10,20,30")
    p_curve.add_argument("--warmup", type=int, default=1)
    p_curve.add_argument("--out", help="output directory (default: a new run directory)")

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
    if args.command in ("run", "rerank-curve"):
        settings = settings if settings is not None else ServiceSettings()
        try:
            return (_run if args.command == "run" else _rerank_curve)(args, settings)
        except UsageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "compare":
        return _compare(args)
    if args.command == "export-hf":
        return _export_hf(args)
    raise AssertionError(args.command)
