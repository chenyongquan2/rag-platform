"""Command line of citation check level 2 over stored answers (spec P1-06, design s11-2).

uv run python -m rag.eval.support run --source p1-04|p3-05 --path P --out DIR [--run 1]
    [--limit K] [--max-cny 0.6] [--estimate-cny 0.0] [--retry-unknown] [--concurrency 4]
uv run python -m rag.eval.support plant --out DIR [--n 10] [--seed 0] [--max-cny 0.6]
uv run python -m rag.eval.support human-sheet --out DIR [--n 40] [--seed 0] [--extra FILE]
uv run python -m rag.eval.support report --out DIR [--md FILE]

`run` judges the stored answers of one source (P1-04 `results.json` or a P3-05 records
directory), appending one line per answer to `DIR/lines.jsonl`; answers already there are
skipped, so the same command resumes or continues a pilot (`--limit`). It refuses to start when
the cost ledger plus `--estimate-cny` is past the budget stop line, and stops before an answer
once the lines in DIR have cost `--max-cny`. No answer is generated. `plant` judges copies of
first-pass answers with one deliberately unsupported sentence each. `report` and `human-sheet`
call no model.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.answer.support import SupportJudgeConfig, build_judge_model
from rag.config import ServiceSettings
from rag.cost import CostLedger
from rag.eval.retrieval.cli import UsageError, _echo
from rag.eval.support.human import export_support_sheet
from rag.eval.support.items import load_p1_04, load_p3_05
from rag.eval.support.planted import plant
from rag.eval.support.records import SupportLine, latest_lines, lines_file, read_lines
from rag.eval.support.report import render_report
from rag.eval.support.runner import RunStats, run_support

DEFAULT_MAX_CNY = 0.6

# ---- seams (replaced in unit tests) -------------------------------------------------------------


def build_support_model(settings: ServiceSettings, config: SupportJudgeConfig) -> Any:
    """The judge model, built by `rag.answer.support.build_judge_model` (one builder for the eval
    and for online sampling)."""
    return build_judge_model(settings, config)


# ---- helpers ------------------------------------------------------------------------------------


def _ledger_check(settings: ServiceSettings, estimate: float) -> None:
    spent = CostLedger.from_settings(settings).total_spent()
    print(
        f"ledger: spent {spent:.4f} CNY, estimate {estimate:.4f} CNY, "
        f"stop at {settings.budget_stop_cny:.2f} CNY"
    )
    if spent + estimate > settings.budget_stop_cny:
        raise UsageError(
            f"the cost ledger ({spent:.4f} CNY) plus the estimate ({estimate:.4f} CNY) is past "
            f"the stop line ({settings.budget_stop_cny} CNY); leave this run to the owner"
        )


def _config(args: argparse.Namespace) -> SupportJudgeConfig:
    update: dict[str, Any] = {"concurrency": args.concurrency}
    if getattr(args, "model", None):
        update["model"] = args.model
    return SupportJudgeConfig().model_copy(update=update)


def _progress(index: int, total: int, line: SupportLine) -> None:
    if index % 5 == 0 or index == total:
        status = "skipped" if line.result is None else (line.result.error or "ok")
        print(
            f"{index}/{total}, last {line.system} {line.qid} ({status}, {line.usage.cny:.4f} CNY)",
            flush=True,
        )


def _print_stats(stats: RunStats, max_cny: float) -> None:
    print(
        f"ran {stats.ran}, already done {stats.done_before}, pending {stats.pending}, "
        f"spend in this directory {stats.spent_cny:.4f} CNY of {max_cny}"
        + (f"; STOPPED by the {stats.stopped} limit" if stats.stopped else "")
    )


# ---- commands -----------------------------------------------------------------------------------


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.max_cny <= 0:
        raise UsageError("--max-cny must be positive")
    if args.run < 1:
        raise UsageError("--run must be a positive integer")
    loader = load_p1_04 if args.source == "p1-04" else load_p3_05
    items = loader(args.path)
    judgeable = sum(i.skip_reason is None for i in items)
    print(f"{args.source}: {len(items)} items, {judgeable} with sentence citations")
    _ledger_check(settings, args.estimate_cny)
    config = _config(args)
    stats = asyncio.run(
        run_support(
            items,
            run=args.run,
            model=build_support_model(settings, config),
            config=config,
            out_path=lines_file(args.out),
            max_cny=args.max_cny,
            limit=args.limit,
            retry_unknown=args.retry_unknown,
            progress=_progress,
        )
    )
    _print_stats(stats, args.max_cny)
    return 0


def _plant(args: argparse.Namespace, settings: ServiceSettings) -> int:
    path = lines_file(args.out)
    first = [x for x in latest_lines(read_lines(path)) if x.run == 1 and x.item.planted is None]
    items = plant(first, n=args.n, seed=args.seed)
    print(f"planted {len(items)} copies: " + ", ".join(i.qid for i in items))
    if not items:
        raise UsageError("no first-pass sentence the judge called supported; run the judge first")
    _ledger_check(settings, args.estimate_cny)
    config = _config(args)
    stats = asyncio.run(
        run_support(
            items,
            run=1,
            model=build_support_model(settings, config),
            config=config,
            out_path=path,
            max_cny=args.max_cny,
            progress=_progress,
        )
    )
    _print_stats(stats, args.max_cny)
    return 0


def _human_sheet(args: argparse.Namespace) -> int:
    lines = latest_lines(read_lines(lines_file(args.out)))
    extra = latest_lines(read_lines(args.extra)) if args.extra else []
    sheet, key = export_support_sheet(
        lines, n=args.n, seed=args.seed, out_dir=Path(args.out) / "human", extra=extra
    )
    print(f"wrote {sheet} (to fill, blind) and {key} (do not open before labelling)")
    return 0


def _report(args: argparse.Namespace) -> int:
    text = render_report(read_lines(lines_file(args.out)))
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.md}")
    _echo(text)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.support",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_judge(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-cny", type=float, default=DEFAULT_MAX_CNY)
        p.add_argument(
            "--estimate-cny", type=float, default=0.0, help="expected spend of this call"
        )
        p.add_argument("--concurrency", type=int, default=SupportJudgeConfig().concurrency)
        p.add_argument("--model", help=f"judge model (default {SupportJudgeConfig().model})")

    p_run = sub.add_parser("run", help="judge stored answers of one source")
    p_run.add_argument("--source", choices=["p1-04", "p3-05"], required=True)
    p_run.add_argument("--path", required=True, help="P1-04 results.json or P3-05 directory")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--run", type=int, default=1)
    p_run.add_argument("--limit", type=int, help="judge at most this many new answers")
    p_run.add_argument("--retry-unknown", action="store_true")
    add_judge(p_run)

    p_plant = sub.add_parser("plant", help="judge copies with planted unsupported sentences")
    p_plant.add_argument("--out", required=True)
    p_plant.add_argument("--n", type=int, default=10)
    p_plant.add_argument("--seed", type=int, default=0)
    add_judge(p_plant)

    p_sheet = sub.add_parser("human-sheet", help="blind sheet for the owner's labels")
    p_sheet.add_argument("--out", required=True)
    p_sheet.add_argument("--n", type=int, default=40)
    p_sheet.add_argument("--seed", type=int, default=0)
    p_sheet.add_argument(
        "--extra", help="lines file of hand-made near misses (planted kind reviewer_hard)"
    )

    p_report = sub.add_parser("report", help="Markdown tables from the lines")
    p_report.add_argument("--out", required=True)
    p_report.add_argument("--md", help="also write the Markdown to this file")
    return parser


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """Exit codes: 0 ok, 2 usage error."""
    args = _parser().parse_args(argv)
    try:
        if args.command in ("run", "plant"):
            settings = settings if settings is not None else ServiceSettings()
            return (_run if args.command == "run" else _plant)(args, settings)
        return {"human-sheet": _human_sheet, "report": _report}[args.command](args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
