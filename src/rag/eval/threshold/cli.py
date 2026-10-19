"""Command line of the rerank-threshold script (spec P2-02; design s10-2, s14-4).

uv run python -m rag.eval.threshold collect --kb hf_docs --questions P --kind answerable|no_answer
    --out FILE [--rerank-n N] [--deadline-ms MS] [--index-prefix P]
uv run python -m rag.eval.threshold sweep --dev FILE [--check NAME=FILE]...
    [--no-answer NAME=FILE]... [--thresholds 0.3,0.5 | 0.05:0.95:0.05] [--max-false-refusal 0.05]
    [--bound observed|wilson_upper] [--baseline T] [--top-k 8] [--out FILE]

`collect` asks every question of a set through the real search chain (the one search_docs uses)
with the threshold forced to 0, so every candidate the reranker scored is kept, and writes the
scores. Nothing is sent to a chat model; the query vectors come from the L0 cache or the gateway.
`sweep` reads score files (no chain, no network) and prints, per candidate threshold, the
false-refusal rate on the answerable `--dev` set (the only set that chooses the threshold), on the
`--check` sets (external, reported only) and the wrong-answer-accepted rate on the `--no-answer`
sets, each with a Wilson 95% interval.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag.config import ServiceSettings
from rag.eval.questions import LabeledQuestion, load_question_set
from rag.eval.retrieval.cli import UsageError, _echo, _setup, build_retriever, reranker_revision
from rag.eval.retrieval.record import QuestionSetInfo
from rag.eval.retrieval.runner import collect_provenance
from rag.eval.threshold.records import (
    Candidate,
    Kind,
    ScoreFile,
    ScoreRecord,
    load_scores,
    write_scores,
)
from rag.eval.threshold.sweep import (
    DEFAULT_TOP_K,
    parse_thresholds,
    recommend,
    render_paired,
    render_sweep,
    sweep,
)

DEFAULT_THRESHOLDS = "0.05:0.95:0.05"
DEV = "dev"
POOLED = "all no-answer"


class CollectError(Exception):
    """A question could not be scored cleanly (rerank timeout, one route down): exit code 1."""


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


def _check_labels(questions: Sequence[LabeledQuestion], kind: Kind) -> None:
    for q in questions:
        if kind == "no_answer" and q.relevant_pages:
            raise UsageError(f"question {q.qid!r} has page labels but --kind is no_answer")
        if kind == "answerable" and not q.relevant_pages:
            raise UsageError(f"question {q.qid!r} has no page labels but --kind is answerable")


def _collect(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.deadline_ms is not None and args.deadline_ms < 1:
        raise UsageError("--deadline-ms must be a positive integer")
    cfg, _public, settings, routes = _setup(args, settings)
    cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"threshold": 0.0})})
    try:
        question_set = load_question_set(args.questions, kb=cfg.kb)
    except (OSError, ValueError) as exc:
        raise UsageError(str(exc).splitlines()[0]) from exc
    _check_labels(question_set.questions, args.kind)
    retriever = build_retriever(
        cfg, settings, rerank=True, routes=routes, index_prefix=args.index_prefix
    )
    records: list[ScoreRecord] = []
    try:
        for i, q in enumerate(question_set.questions, start=1):
            result = retriever.retrieve(q.question, kb=cfg.kb, top_k=cfg.retrieval.final_k)
            if result.error_categories:
                flagged = ", ".join(str(c.value) for c in result.error_categories)
                raise CollectError(
                    f"question {q.qid!r}: {flagged}; its scores are incomplete, "
                    "raise --deadline-ms and run again"
                )
            if any(h.score is None for h in result.rerank):
                raise CollectError(f"question {q.qid!r}: a reranked candidate has no score")
            records.append(
                ScoreRecord(
                    qid=q.qid,
                    question=q.question,
                    kind=args.kind,
                    relevant_pages=q.relevant_pages,
                    candidates=[
                        Candidate(chunk_id=h.chunk_id, doc_id=h.doc_id, score=float(h.score or 0))
                        for h in result.rerank
                    ],
                )
            )
            if i % 25 == 0 or i == len(question_set.questions):
                print(f"{i}/{len(question_set.questions)} questions scored", flush=True)
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    scores = ScoreFile(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        kb=cfg.kb,
        rerank_n=cfg.retrieval.rerank_n,
        question_set=QuestionSetInfo(
            path=question_set.path, sha256=question_set.sha256, count=len(question_set.questions)
        ),
        provenance=collect_provenance(settings=settings, models=_models(settings)),
        records=records,
    )
    print(f"wrote {write_scores(scores, args.out)}")
    return 0


def _named_file(text: str, option: str) -> tuple[str, Path]:
    name, sep, file = text.partition("=")
    if not sep or not name.strip() or not file.strip():
        raise UsageError(f"{option} takes NAME=FILE, got {text!r}")
    return name.strip(), Path(file.strip())


def _load(path: Path, kind: Kind, option: str) -> ScoreFile:
    if not path.is_file():
        raise UsageError(f"{option}: {path} not found")
    try:
        scores = load_scores(path)
        found = scores.kind
    except ValueError as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise UsageError(f"{option}: {path}: {first}") from exc
    if found != kind:
        raise UsageError(f"{option}: {path} holds {found} questions, expected {kind}")
    return scores


def _sweep(args: argparse.Namespace) -> int:
    if not 0.0 <= args.max_false_refusal <= 1.0:
        raise UsageError("--max-false-refusal must be between 0 and 1")
    if args.baseline is not None and not 0.0 <= args.baseline <= 1.0:
        raise UsageError("--baseline must be between 0 and 1")
    if args.top_k < 1:
        raise UsageError("--top-k must be a positive integer")
    try:
        thresholds = parse_thresholds(args.thresholds)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    files: dict[str, ScoreFile] = {DEV: _load(Path(args.dev), "answerable", "--dev")}
    checks: list[str] = []
    no_answers: list[str] = []
    for option, kind, bucket in (
        ("--check", "answerable", checks),
        ("--no-answer", "no_answer", no_answers),
    ):
        for item in getattr(args, option[2:].replace("-", "_")) or []:
            name, path = _named_file(item, option)
            if name in files or name == POOLED:
                raise UsageError(f"{option}: the name {name!r} is already used")
            files[name] = _load(path, kind, option)  # type: ignore[arg-type]
            bucket.append(name)
    if len({f.rerank_n for f in files.values()}) > 1:
        raise UsageError("the score files were collected with different rerank_n")
    groups = {name: scores.records for name, scores in files.items()}
    if len(no_answers) > 1:
        groups[POOLED] = [r for name in no_answers for r in groups[name]]
    names = [*no_answers, POOLED] if len(no_answers) > 1 else no_answers
    rows = sweep(groups, thresholds, top_k=args.top_k)
    recommended = recommend(
        rows, group=DEV, max_false_refusal=args.max_false_refusal, bound=args.bound
    )
    sizes = ", ".join(f"{name} {len(groups[name])} questions" for name in groups)
    text = f"Sets: {sizes}; rerank_n {next(iter(files.values())).rerank_n}.\n\n" + render_sweep(
        rows,
        dev=DEV,
        checks=checks,
        no_answers=names,
        recommended=recommended,
        max_false_refusal=args.max_false_refusal,
        bound=args.bound,
    )
    if args.baseline is not None:
        text += "\n" + render_paired(thresholds, groups, baseline=args.baseline)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    _echo(text)
    return 0


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """CLI entry; `settings` is for tests. Exit codes: 0 ok, 1 a question could not be scored
    cleanly, 2 usage error."""
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.threshold",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="score a question set with threshold 0")
    p_collect.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    p_collect.add_argument("--questions", required=True, help="question set (JSONL)")
    p_collect.add_argument("--kind", required=True, choices=["answerable", "no_answer"])
    p_collect.add_argument("--out", required=True, help="score file to write (JSON)")
    p_collect.add_argument("--rerank-n", type=int, help="override retrieval.rerank_n")
    p_collect.add_argument("--deadline-ms", type=int, help="search deadline (default: settings')")
    p_collect.add_argument("--index-prefix", default="", help="e.g. fake_ for the fake index")
    p_collect.set_defaults(routes=None)

    p_sweep = sub.add_parser("sweep", help="false-refusal and wrong-accept rates per threshold")
    p_sweep.add_argument("--dev", required=True, help="score file of the answerable selection set")
    p_sweep.add_argument(
        "--check", action="append", metavar="NAME=FILE", help="answerable set, reported only"
    )
    p_sweep.add_argument(
        "--no-answer", action="append", metavar="NAME=FILE", help="no-answer set (repeatable)"
    )
    p_sweep.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    p_sweep.add_argument("--max-false-refusal", type=float, default=0.05)
    p_sweep.add_argument("--bound", choices=["observed", "wilson_upper"], default="observed")
    p_sweep.add_argument(
        "--baseline", type=float, help="also list questions that flip against this threshold"
    )
    p_sweep.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_sweep.add_argument("--out", help="also write the Markdown to this file")

    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            return _collect(args, settings if settings is not None else ServiceSettings())
        return _sweep(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CollectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
