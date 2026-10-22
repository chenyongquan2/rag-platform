# ruff: noqa: E501 (long lines are Markdown table rows and console messages)
"""Command line of the Agentic RAG comparison (spec P3-05, design s13-5).

uv run python -m rag.eval.agentic run --kb hf_docs --out DIR [--system standard|agentic|both]
    [--types T,...] [--take TYPE=N ...] [--qids ID,...] [--deadline-ms MS] [--retry-errors]
    [--max-cny 5.0] [--index-prefix P] [--kb-override SECTION.KEY=VALUE ...]
uv run python -m rag.eval.agentic judge --out DIR [--runs 2] [--model M] [--concurrency 4]
uv run python -m rag.eval.agentic report --out DIR [--md FILE]
uv run python -m rag.eval.agentic human-sheet --out DIR [--n 30] [--seed 0]
uv run python -m rag.eval.agentic project --out DIR [--judge-cny-per-call 0.0014]
uv run python -m rag.eval.agentic compare --before DIR --after DIR [--md FILE]
uv run python -m rag.eval.agentic compare-runs --before DIR --after DIR [--system S] [--md FILE]

`run` answers the question sets (public single-hop, two-hop, colloquial, no-answer) by standard
RAG and/or the real `ask_docs` graph, one question at a time, appending each result to
`DIR/<system>/records.jsonl`; questions already there are skipped, so the same command resumes
after a crash or continues a pilot (`--take`) into the full run. The run stops before a question
once this spec's spend (both systems' records and the judge's lines in DIR) has reached
`--max-cny`, and refuses to start when the cost ledger is already past the budget stop line.
`judge` scores the answerable records with the P1-04 judge (1-3 scale), `--runs` passes.
`report`, `project` and `compare` call no model. `compare` (spec P3-06) sets a new run of the
graph (`--after`) beside the P3-05 records of standard RAG and of the graph (`--before`), on the
questions the new run has, by group (what happened to the question in P3-05). `compare-runs`
(spec P3-07) pairs one system's records in `--after` with the same system's in `--before`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rag.agent import AgentHooks, AgentRuntime
from rag.config import KnowledgeBaseConfig, ServiceSettings, repo_path
from rag.cost import CostLedger
from rag.eval.agentic.analysis import judge_scores
from rag.eval.agentic.human import export_human_sheet
from rag.eval.agentic.judging import judge_records
from rag.eval.agentic.records import (
    SYSTEMS,
    SystemRecord,
    judge_file,
    read_judge_lines,
    read_records,
    system_file,
)
from rag.eval.agentic.report import project_cost, render_report
from rag.eval.agentic.runner import REPORTS_ROOT, Tracer, run_agentic, run_standard
from rag.eval.agentic.runs import compare_runs, render_compare_runs
from rag.eval.agentic.sets import QTYPES, EvalItem, load_items
from rag.eval.agentic.subset import compare_subset
from rag.eval.answer.cli import build_answer_model, build_judge_model
from rag.eval.answer.judge import JudgeConfig
from rag.eval.answer.runner import RetrieverContexts
from rag.eval.retrieval.cli import UsageError, _echo, _kb_config
from rag.eval.retrieval.protocol import Retriever

DEFAULT_MAX_CNY = 5.0

# ---- seams (replaced in unit tests) -------------------------------------------------------------


def build_retriever(
    cfg: KnowledgeBaseConfig, settings: ServiceSettings, *, index_prefix: str
) -> Retriever:
    """The chain search_docs and the graph use: fusion, rerank, threshold."""
    from rag.eval.answer.cli import build_retriever as build

    return build(cfg, settings, index_prefix=index_prefix)


def build_agent_models(settings: ServiceSettings) -> tuple[Any, Any, Any]:
    """(answer model, helper model, judge model) of the graph: `None` lets the runtime make
    them from the profiles in the settings (`online` for the answer, `cheap` for the rest)."""
    return None, None, None


def build_runtime(settings: ServiceSettings, hooks: AgentHooks) -> AgentRuntime:
    """One runtime for the whole run; checkpoints, history and locks in process memory (the
    eval keeps no session, and every question is its own thread)."""
    memory = settings.model_copy(
        update={
            "checkpoint_url": "memory://",
            "session_store_url": "memory://",
            "session_lock_url": "memory://",
        }
    )
    return AgentRuntime.from_settings(memory, hooks=hooks)


# ---- helpers ------------------------------------------------------------------------------------


def _spent_in(out: Path, *, skip: str | None = None) -> float:
    """CNY already spent by this spec's records and judge lines in `out`."""
    total = 0.0
    for system in SYSTEMS:
        if system != skip:
            total += sum(r.usage.cny for r in read_records(system_file(out, system)))
    total += sum(x.usage.cny for x in read_judge_lines(judge_file(out)))
    return total


def _ledger_check(settings: ServiceSettings) -> None:
    spent = CostLedger.from_settings(settings).total_spent()
    print(f"ledger: spent {spent:.4f} CNY, stop at {settings.budget_stop_cny:.2f} CNY")
    if spent >= settings.budget_stop_cny:
        raise UsageError(
            f"the cost ledger ({spent:.4f} CNY) is already at the stop line "
            f"({settings.budget_stop_cny} CNY); leave this run to the owner"
        )


def _take(values: Sequence[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    take: dict[str, int] = {}
    for text in values:
        name, sep, number = text.partition("=")
        if not sep or not number.isdigit() or int(number) < 1:
            raise UsageError(f"--take takes TYPE=N (N >= 1), got {text!r}")
        take[name.strip()] = int(number)
    return take


def _items(args: argparse.Namespace, kb: str) -> list[EvalItem]:
    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    try:
        items = load_items(kb, types=types, take=_take(args.take))
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in {i.qid for i in items}]
        if unknown:
            raise UsageError(f"--qids: unknown question ids {unknown}")
        items = [i for i in items if i.qid in set(wanted)]
    return items


def apply_kb_overrides(
    cfg: KnowledgeBaseConfig, overrides: Sequence[str] | None
) -> KnowledgeBaseConfig:
    """A copy of `cfg` with `section.key=value` set (value read as YAML), validated again; for
    eval runs only (spec P3-07: `agent.round2_on_not_enough=false`). The kb file is unchanged."""
    if not overrides:
        return cfg
    data = cfg.model_dump()
    for text in overrides:
        key, sep, raw = text.partition("=")
        parts = key.strip().split(".")
        if not sep or len(parts) < 2:
            raise UsageError(f"--kb-override takes SECTION.KEY=VALUE, got {text!r}")
        node = data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise UsageError(f"--kb-override: unknown section {key!r}")
            node = node[part]
        if parts[-1] not in node:
            raise UsageError(f"--kb-override: unknown key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    try:
        return KnowledgeBaseConfig.model_validate(data)
    except ValueError as exc:
        raise UsageError(f"--kb-override: {exc}") from exc


RUNS_NAME = "runs.jsonl"


def _note_run(out: Path, system: str, args: argparse.Namespace, questions: int) -> None:
    """One line per `run` and system in `out/runs.jsonl`: when, which settings were changed for
    this run only (`--kb-override`, `--deadline-ms`), how many questions were asked for."""
    line = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": system,
        "kb": args.kb,
        "kb_overrides": list(args.kb_override or []),
        "deadline_ms": args.deadline_ms,
        "questions": questions,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / RUNS_NAME).open("a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(line) + "\n")


def _out(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else repo_path(REPORTS_ROOT)


def _progress(system: str):  # type: ignore[no-untyped-def]
    def show(index: int, total: int, record: SystemRecord) -> None:
        if index % 5 == 0 or index == total:
            print(
                f"{system}: {index}/{total} questions, last {record.qid} "
                f"({record.status}, {record.total_ms / 1000:.1f} s, {record.usage.cny:.4f} CNY)",
                flush=True,
            )

    return show


# ---- commands -----------------------------------------------------------------------------------


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.max_cny <= 0:
        raise UsageError("--max-cny must be positive")
    cfg, _public = _kb_config(settings, args.kb)
    cfg = apply_kb_overrides(cfg, args.kb_override)
    items = _items(args, cfg.kb)
    out = _out(args)
    _ledger_check(settings)
    systems = list(SYSTEMS) if args.system == "both" else [args.system]
    if args.kb_override:
        print(f"kb overrides for this run: {args.kb_override}")
    retriever = build_retriever(cfg, settings, index_prefix=args.index_prefix)
    try:
        for system in systems:
            path = system_file(out, system)
            _note_run(out, system, args, len(items))
            before = _spent_in(out, skip=system)
            if system == "standard":
                stats = asyncio.run(
                    run_standard(
                        items,
                        records_path=path,
                        contexts=RetrieverContexts(
                            retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k
                        ),
                        kb_config=cfg,
                        model=build_answer_model(settings),
                        max_cny=args.max_cny,
                        spent_before=before,
                        progress=_progress(system),
                        retry_errors=args.retry_errors,
                    )
                )
            else:
                tracer = Tracer()
                model, helper, judge = build_agent_models(settings)
                stats = asyncio.run(
                    run_agentic(
                        items,
                        records_path=path,
                        runtime=build_runtime(settings, tracer.hooks()),
                        retriever=retriever,
                        kb_config=cfg,
                        tracer=tracer,
                        max_cny=args.max_cny,
                        spent_before=before,
                        deadline_ms=args.deadline_ms,
                        model=model,
                        helper_model=helper,
                        judge_model=judge,
                        progress=_progress(system),
                        retry_errors=args.retry_errors,
                    )
                )
            print(
                f"{system}: ran {stats.ran}, already done {stats.skipped}, pending {stats.pending}, "
                f"spend counted {stats.spent_cny:.4f} CNY of {args.max_cny}"
                + (f"; STOPPED by the {stats.stopped} limit" if stats.stopped else "")
            )
            if stats.stopped:
                break
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    return 0


def _all_records(out: Path) -> list[SystemRecord]:
    return [r for system in SYSTEMS for r in read_records(system_file(out, system))]


def _judge(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.runs < 1:
        raise UsageError("--runs must be a positive integer")
    out = _out(args)
    records = _all_records(out)
    if not records:
        raise UsageError(f"no records under {out}; run the systems first")
    _ledger_check(settings)
    update: dict[str, Any] = {"concurrency": args.concurrency}
    if args.model:
        update["model"] = args.model
    config = JudgeConfig().model_copy(update=update)
    for run in range(1, args.runs + 1):
        # A new model (and HTTP client) per pass: pooled connections belong to the event loop of
        # the `asyncio.run` that opened them and fail once that loop is closed.
        model = build_judge_model(settings, config)
        new = asyncio.run(
            judge_records(records, run=run, model=model, config=config, out_path=judge_file(out))
        )
        called = sum(1 for x in new if x.by_rule is None)
        print(f"judge pass {run}: {len(new)} new scores ({called} by the model)")
    print(
        f"judge spend so far {sum(x.usage.cny for x in read_judge_lines(judge_file(out))):.4f} CNY"
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    out = _out(args)
    text = render_report(_all_records(out), read_judge_lines(judge_file(out)))
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.md}")
    _echo(text)
    return 0


def _human_sheet(args: argparse.Namespace) -> int:
    out = _out(args)
    sheet, key = export_human_sheet(
        _all_records(out),
        n=args.n,
        seed=args.seed,
        out_dir=out / "human",
        lines=read_judge_lines(judge_file(out)),
    )
    print(f"wrote {sheet} (to fill, blind) and {key} (do not open before scoring)")
    return 0


def _compare(args: argparse.Namespace) -> int:
    before, after = Path(args.before), Path(args.after)

    def records(out: Path, system: str) -> dict[str, SystemRecord]:
        return {r.qid: r for r in read_records(system_file(out, system))}

    standard, old, new = (
        records(before, "standard"),
        records(before, "agentic"),
        records(after, "agentic"),
    )
    if not new:
        raise UsageError(f"no graph records under {after}")
    old_lines = read_judge_lines(judge_file(before))
    new_lines = read_judge_lines(judge_file(after))
    scores = {
        "A standard": judge_scores(old_lines, "standard"),
        "B before": judge_scores(old_lines, "agentic"),
        "B after": judge_scores(new_lines, "agentic"),
    }
    text = compare_subset(standard, old, new, scores)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.md}")
    _echo(text)
    return 0


def _compare_runs(args: argparse.Namespace) -> int:
    before, after = Path(args.before), Path(args.after)
    old = read_records(system_file(before, args.system))
    new = read_records(system_file(after, args.system))
    if not old or not new:
        raise UsageError(f"no {args.system} records under {before if not old else after}")
    report = compare_runs(
        old,
        new,
        read_judge_lines(judge_file(before)),
        read_judge_lines(judge_file(after)),
        system=args.system,
    )
    text = render_compare_runs(report)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.md}")
    _echo(text)
    return 0


def _project(args: argparse.Namespace) -> int:
    out = _out(args)
    pilot = _all_records(out)
    if not pilot:
        raise UsageError(f"no records under {out}")
    sizes = {t: len(load_items(args.kb, types=[t])) for t in QTYPES}
    p = project_cost(
        pilot, sizes, judge_cny_per_call=args.judge_cny_per_call, judge_passes=args.judge_passes
    )
    print("mean CNY per question from the records so far:")
    for system, by_type in p.mean_cny.items():
        print(f"  {system}: " + ", ".join(f"{t} {m:.4f}" for t, m in by_type.items()))
    for system in SYSTEMS:
        print(
            f"{system}: whole set {p.by_system[system]:.3f} CNY, still to run "
            f"{p.remaining_by_system[system]:.3f} CNY"
        )
    print(f"judge ({args.judge_passes} passes): {p.judge:.3f} CNY")
    print(f"total: {p.total:.3f} CNY (still to spend {p.remaining_total:.3f} CNY)")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.agentic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_out(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", help=f"records directory (default {REPORTS_ROOT})")

    p_run = sub.add_parser("run", help="run standard RAG and/or the ask_docs graph")
    p_run.add_argument("--kb", default="hf_docs")
    p_run.add_argument("--system", choices=["standard", "agentic", "both"], default="both")
    p_run.add_argument("--types", help=f"question types, comma separated ({', '.join(QTYPES)})")
    p_run.add_argument("--take", action="append", metavar="TYPE=N", help="first N of a type")
    p_run.add_argument("--qids", help="only these question ids, comma separated")
    p_run.add_argument(
        "--deadline-ms", type=int, help="total deadline of each graph run (default: the settings')"
    )
    p_run.add_argument(
        "--retry-errors",
        action="store_true",
        help="run again the questions whose record is an error",
    )
    p_run.add_argument("--max-cny", type=float, default=DEFAULT_MAX_CNY)
    p_run.add_argument(
        "--kb-override",
        action="append",
        metavar="SECTION.KEY=VALUE",
        help="set one kb config key for this run only (YAML value)",
    )
    p_run.add_argument("--index-prefix", default="")
    add_out(p_run)

    p_judge = sub.add_parser("judge", help="score the answers with the P1-04 judge")
    p_judge.add_argument("--runs", type=int, default=2)
    p_judge.add_argument("--model", help=f"judge model (default {JudgeConfig().model})")
    p_judge.add_argument("--concurrency", type=int, default=JudgeConfig().concurrency)
    add_out(p_judge)

    p_report = sub.add_parser("report", help="tables from the records and judge lines")
    p_report.add_argument("--md", help="also write the Markdown to this file")
    add_out(p_report)

    p_sheet = sub.add_parser("human-sheet", help="blind sheet for human scores")
    p_sheet.add_argument("--n", type=int, default=30)
    p_sheet.add_argument("--seed", type=int, default=0)
    add_out(p_sheet)

    p_cmp = sub.add_parser("compare", help="a new graph run beside the P3-05 records (P3-06)")
    p_cmp.add_argument("--before", required=True, help="P3-05 records directory")
    p_cmp.add_argument("--after", required=True, help="records directory of the new graph run")
    p_cmp.add_argument("--md", help="also write the Markdown to this file")

    p_runs = sub.add_parser(
        "compare-runs", help="one system's new run against an older run (P3-07)"
    )
    p_runs.add_argument("--before", required=True, help="records directory of the older run")
    p_runs.add_argument("--after", required=True, help="records directory of the new run")
    p_runs.add_argument("--system", choices=list(SYSTEMS), default="agentic")
    p_runs.add_argument("--md", help="also write the Markdown to this file")

    p_proj = sub.add_parser("project", help="project the cost of the full run from the records")
    p_proj.add_argument("--kb", default="hf_docs")
    p_proj.add_argument("--judge-cny-per-call", type=float, default=0.0014)
    p_proj.add_argument("--judge-passes", type=int, default=2)
    add_out(p_proj)
    return parser


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """Exit codes: 0 ok, 2 usage error."""
    args = _parser().parse_args(argv)
    try:
        if args.command in ("run", "judge"):
            settings = settings if settings is not None else ServiceSettings()
            return (_run if args.command == "run" else _judge)(args, settings)
        handler = {
            "report": _report,
            "human-sheet": _human_sheet,
            "project": _project,
            "compare": _compare,
            "compare-runs": _compare_runs,
        }
        return handler[args.command](args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
