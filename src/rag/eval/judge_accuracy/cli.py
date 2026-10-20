"""Command line of the judge-accuracy script (spec P3-03; design s13-4, s14-2).

uv run python -m rag.eval.judge_accuracy run --kb hf_docs --set NAME=FILE [--set ...] --out DIR
    [--limit N] [--max-cny 2.0] [--rescue] [--threshold T] [--index-prefix P] [--deadline-ms MS]
uv run python -m rag.eval.judge_accuracy report RECORDS.json [--out FILE]

`run` asks every question of the sets through the real search chain (the one search_docs uses),
asks the sufficiency judge with the real `agent_judge_profile` model, labels each question from
its gold page (a question is "enough on round one" when the page is among the returned chunks; a
no-answer question never is) and writes `records.json` and `report.md` under `--out`. Each set
file holds either answerable questions (all with a gold page) or no-answer questions (none).
`--rescue` also writes rounds two and three for the answerable questions that were not enough,
with the real helper model, to see what another round is worth. The run stops before a question
once it has spent `--max-cny` (default 2.0) according to the cost ledger. `report` recomputes the
report from a records file: no model, no search.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings
from rag.eval.judge_accuracy import metrics
from rag.eval.judge_accuracy.records import (
    JudgeFile,
    JudgeRecord,
    SetInfo,
    load_records,
    write_records,
)
from rag.eval.judge_accuracy.runner import QuestionSpec, collect_records
from rag.eval.questions import load_question_set
from rag.eval.retrieval.cli import UsageError, _echo, _setup, build_retriever, reranker_revision
from rag.eval.retrieval.runner import collect_provenance

DEFAULT_MAX_CNY = 2.0


def build_models(settings: ServiceSettings, rescue: bool) -> tuple[Any, Any]:
    """The judge's chat model and, for `--rescue`, the helper model that writes round two and
    three (the real ones, through the cost ledger). A seam: unit tests replace it."""
    from rag.llm import make_chat_model

    judge = make_chat_model(settings.agent_judge_profile, settings)
    helper = make_chat_model(settings.agent_rewrite_profile, settings) if rescue else None
    return judge, helper


def spent_cny(settings: ServiceSettings) -> float:
    """CNY spent so far according to the cost ledger. A seam: unit tests replace it."""
    from rag.cost import total_spent

    return total_spent(settings)


def _named_file(text: str) -> tuple[str, Path]:
    name, sep, file = text.partition("=")
    if not sep or not name.strip() or not file.strip():
        raise UsageError(f"--set takes NAME=FILE, got {text!r}")
    return name.strip(), Path(file.strip())


def _load_specs(
    sets: Sequence[str], kb: str, limit: int | None
) -> tuple[list[QuestionSpec], list[SetInfo]]:
    specs: list[QuestionSpec] = []
    infos: list[SetInfo] = []
    names: set[str] = set()
    for item in sets:
        name, path = _named_file(item)
        if name in names:
            raise UsageError(f"--set: the name {name!r} is used twice")
        names.add(name)
        try:
            loaded = load_question_set(path, kb=kb)
        except (OSError, ValueError) as exc:
            raise UsageError(str(exc).splitlines()[0]) from exc
        with_pages = [bool(q.relevant_pages) for q in loaded.questions]
        if any(with_pages) and not all(with_pages):
            raise UsageError(
                f"--set {name}: {path} mixes questions with and without gold pages; "
                "split it into an answerable and a no-answer file"
            )
        kind = "answerable" if all(with_pages) else "no_answer"
        questions = loaded.questions[:limit] if limit else loaded.questions
        for q in questions:
            default = "single_hop" if kind == "answerable" else "no_answer"
            specs.append(QuestionSpec(name, q, kind, str(q.meta.get("type") or default)))
        infos.append(
            SetInfo(name=name, path=loaded.path, sha256=loaded.sha256, count=len(questions))
        )
    return specs, infos


def _models_provenance(settings: ServiceSettings) -> dict[str, str]:
    models = {
        "embed": settings.embed_model,
        "reranker": settings.reranker_model,
        "judge_profile": settings.agent_judge_profile,
    }
    revision = reranker_revision(settings.reranker_model)
    if revision:
        models["reranker_revision"] = revision
    return models


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.max_cny <= 0:
        raise UsageError("--max-cny must be positive")
    if args.limit is not None and args.limit < 1:
        raise UsageError("--limit must be a positive integer")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise UsageError("--threshold must be between 0 and 1")
    cfg, _public, settings, routes = _setup(args, settings)
    if args.threshold is not None:
        retrieval = cfg.retrieval.model_copy(update={"threshold": args.threshold})
        cfg = cfg.model_copy(update={"retrieval": retrieval})
    specs, infos = _load_specs(args.set, cfg.kb, args.limit)
    spent = spent_cny(settings)
    if spent + args.max_cny > settings.budget_stop_cny:
        raise UsageError(
            f"spent {spent:.4f} CNY + the limit {args.max_cny} CNY would pass the stop "
            f"{settings.budget_stop_cny} CNY; lower --max-cny or leave the run to the owner"
        )
    judge_model, helper_model = build_models(settings, args.rescue)
    retriever = build_retriever(
        cfg, settings, rerank=True, routes=routes, index_prefix=args.index_prefix
    )

    def progress(index: int, total: int, record: JudgeRecord) -> None:
        if index % 10 == 0 or index == total:
            print(f"{index}/{total} questions judged", flush=True)

    try:
        file = collect_records(
            specs,
            retriever=retriever,
            cfg=cfg,
            settings=settings,
            judge_model=judge_model,
            helper_model=helper_model,
            rescue=args.rescue,
            max_cny=args.max_cny,
            spent_cny=lambda: spent_cny(settings),
            progress=progress,
        )
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    file = file.model_copy(
        update={
            "question_sets": infos,
            "provenance": collect_provenance(
                settings=settings, models=_models_provenance(settings)
            ),
        }
    )
    out = Path(args.out)
    print(f"wrote {write_records(file, out / 'records.json')}")
    text = metrics.render_report(file)
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    _echo(text)
    return 0


def _report(args: argparse.Namespace) -> int:
    path = Path(args.records)
    if not path.is_file():
        raise UsageError(f"{path} not found")
    try:
        file: JudgeFile = load_records(path)
    except ValueError as exc:
        raise UsageError(f"{path}: {str(exc).splitlines()[0]}") from exc
    text = metrics.render_report(file)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    _echo(text)
    return 0


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """CLI entry; `settings` is for tests. Exit codes: 0 ok, 2 usage error."""
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.judge_accuracy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="judge every question and write the records")
    p_run.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    p_run.add_argument(
        "--set", action="append", required=True, metavar="NAME=FILE", help="question set (JSONL)"
    )
    p_run.add_argument("--out", required=True, help="directory for records.json and report.md")
    p_run.add_argument("--limit", type=int, help="only the first N questions of each set")
    p_run.add_argument("--max-cny", type=float, default=DEFAULT_MAX_CNY, help="spend limit")
    p_run.add_argument("--rescue", action="store_true", help="also try rounds two and three")
    p_run.add_argument("--threshold", type=float, help="override retrieval.threshold")
    p_run.add_argument("--index-prefix", default="", help="e.g. fake_ for the fake index")
    p_run.add_argument("--deadline-ms", type=int, help="search deadline (default: settings')")
    p_run.set_defaults(routes=None)

    p_report = sub.add_parser("report", help="recompute the report from a records file")
    p_report.add_argument("records", help="records.json written by run")
    p_report.add_argument("--out", help="also write the Markdown to this file")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args, settings if settings is not None else ServiceSettings())
        return _report(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
