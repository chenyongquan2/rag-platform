# ruff: noqa: E501 (long lines are console messages)
"""Commands of the repeated runs (spec EVAL-3, plan step 3), part of `python -m rag.eval.agentic`.

    repeat-sample  [--kb hf_docs] [--seed N] [--date D] [--out FILE] [--check]
    repeat-run     --base DIR --pass N [--list FILE] [--pass1-old DIR]... [--pass1-new DIR]
                   [--estimate-only] [--no-judge] [--retry-errors] [--max-cny X]
                   [--allow-warmup-failures N] [--kb hf_docs] [--index-prefix P]
    repeat-analyze --base DIR [--passes 3] [--list FILE] [--pass1-old DIR]... [--pass1-new DIR]
                   [--report FILE | --no-report] [--stable-out FILE]

`repeat-run` runs one pass (2, 3, ...) into `BASE/passN`: it prints the estimated cost and the
ledger first, refuses when spent + estimate (with margin) would pass the stop line, warms up the
retrieval chain (no chat model), runs the two systems in the pass's order (even passes: standard
RAG first), then judges. The same command resumes a pass that stopped half way. `repeat-analyze`
reads pass 1 (the stored runs), `pass2`, `pass3`, ..., writes the Chinese section into the report
and the file of stable questions. No command here changes agent code or settings.

The commands that run questions go through the seams of `rag.eval.agentic.cli` (`build_retriever`,
`build_runtime`, `run_standard`, ...), so unit tests replace them with fakes the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings, repo_path
from rag.cost import CostLedger
from rag.eval.agentic.records import (
    SYSTEMS,
    SystemRecord,
    judge_file,
    read_judge_lines,
    read_records,
    retry_reason,
    system_file,
)
from rag.eval.agentic.repeat.analysis import analyze, stable_payload
from rag.eval.agentic.repeat.passes import (
    Estimate,
    RepeatManifest,
    check_budget,
    estimate_pass,
    load_manifest,
    load_pass,
    load_stored_pass,
    mean_costs,
    pass_dir,
    save_manifest,
    system_order,
    warm_up,
)
from rag.eval.agentic.repeat.report import ReportContext, render_section, write_section
from rag.eval.agentic.repeat.sample import (
    DEFAULT_DATE,
    DEFAULT_SEED,
    LIST_PATH,
    LIST_TYPES,
    QuestionList,
    build_question_list,
    read_question_list,
    write_question_list,
)
from rag.eval.agentic.sets import EvalItem, load_items
from rag.eval.answer.judge import JudgeConfig
from rag.eval.answer.runner import RetrieverContexts
from rag.eval.retrieval.cli import UsageError, _echo, _kb_config

REPORT_PATH = "docs/reports/eval-upgrade-report.md"
STABLE_NAME = "stable_questions.json"


def _base() -> Any:
    """`rag.eval.agentic.cli`, imported late: it imports this module to register the commands."""
    from rag.eval.agentic import cli

    return cli


# ---- repeat-sample -------------------------------------------------------------------------------


def _sample(args: argparse.Namespace) -> int:
    items = load_items(args.kb, types=list(LIST_TYPES))
    fresh = build_question_list(items, seed=args.seed, date=args.date)
    target = Path(args.out) if args.out else repo_path(LIST_PATH)
    if args.check:
        if not target.is_file():
            raise UsageError(f"no question list at {target}")
        if read_question_list(target) != fresh:
            raise UsageError(
                f"{target} is not what seed {args.seed} and date {args.date} draw from the question sets"
            )
        print(
            f"{target}: matches a fresh draw (seed {args.seed}, {sum(fresh.counts.values())} questions)"
        )
        return 0
    write_question_list(target, fresh)
    print(f"wrote {target}: {fresh.counts} (seed {args.seed}, {args.date})")
    return 0


# ---- shared helpers ------------------------------------------------------------------------------


def _sources(args: argparse.Namespace) -> RepeatManifest:
    """Question list and pass-1 directories: the options, else what `repeat-run` saved in BASE."""
    saved = load_manifest(args.base)
    return RepeatManifest(
        question_list=args.list or saved.question_list or LIST_PATH,
        pass1_old=list(args.pass1_old or saved.pass1_old),
        pass1_new=list(args.pass1_new or saved.pass1_new),
    )


def _question_list(manifest: RepeatManifest) -> QuestionList:
    path = Path(manifest.question_list)
    if not path.is_absolute() and not path.is_file():
        path = repo_path(manifest.question_list)
    if not path.is_file():
        raise UsageError(f"no question list at {manifest.question_list}; run repeat-sample")
    return read_question_list(path)


def _items(ql: QuestionList, kb: str) -> list[EvalItem]:
    by_qid = {i.qid: i for i in load_items(kb, types=list(LIST_TYPES))}
    missing = [q for q in ql.all_qids() if q not in by_qid]
    if missing:
        raise UsageError(f"questions of the list are not in the question sets: {missing[:5]}")
    return [by_qid[q] for q in ql.all_qids()]


def _reference_records(manifest: RepeatManifest, base: str, this_pass: int) -> list[SystemRecord]:
    """Earlier records the cost per question is taken from: the stored pass 1 runs (all their
    questions) and the other passes already in BASE."""
    records: list[SystemRecord] = []
    stored = [
        load_pass(manifest.pass1_old, None),
        load_pass(manifest.pass1_new, None),
    ]
    for data in stored:
        records += [c.record for cells in data.values() for c in cells.values()]
    for other in sorted(Path(base).glob("pass*")):
        if other.name != f"pass{this_pass}" and other.is_dir():
            data = load_pass([other], None)
            records += [c.record for cells in data.values() for c in cells.values()]
    return records


# ---- repeat-run ----------------------------------------------------------------------------------


def _todo(items: Sequence[EvalItem], directory: Path, *, retry: bool) -> dict[str, list[EvalItem]]:
    todo: dict[str, list[EvalItem]] = {}
    for system in SYSTEMS:
        path = system_file(directory, system)
        have = {r.qid: r for r in read_records(path)}
        todo[system] = [
            i
            for i in items
            if i.qid not in have or (retry and retry_reason(have[i.qid]) is not None)
        ]
    return todo


def _judge_calls(items: Sequence[EvalItem], directory: Path) -> int:
    judged = {(x.system, x.qid) for x in read_judge_lines(judge_file(directory)) if x.run == 1}
    return sum(1 for s in SYSTEMS for i in items if i.answerable and (s, i.qid) not in judged)


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.pass_no < 2:
        raise UsageError("--pass must be 2 or more: pass 1 is the stored run, it is not run again")
    manifest = _sources(args)
    if not manifest.pass1_old:
        raise UsageError(
            "pass 1 is not known: give --pass1-old DIR (the P3-07 run; repeat it for a retry directory)"
        )
    ql = _question_list(manifest)
    items = _items(ql, args.kb)
    save_manifest(args.base, manifest)
    directory = pass_dir(args.base, args.pass_no)
    base_cli = _base()

    todo = _todo(items, directory, retry=args.retry_errors)
    judge_calls = 0 if args.no_judge else _judge_calls(items, directory)
    means = mean_costs(_reference_records(manifest, args.base, args.pass_no))
    est: Estimate = estimate_pass(todo, means, judge_calls=judge_calls)
    for line in est.lines():
        print(line)
    spent = CostLedger.from_settings(settings).total_spent()
    print(f"ledger: spent {spent:.4f} CNY, stop at {settings.budget_stop_cny:.2f} CNY")
    check_budget(est, spent=spent, stop=settings.budget_stop_cny)
    order = system_order(args.pass_no)
    print(f"pass {args.pass_no}: {order[0]} first, then {order[1]}; records in {directory}")
    if args.estimate_only:
        return 0
    if not any(todo.values()) and judge_calls == 0:
        print(f"pass {args.pass_no} is complete: nothing to run")
        base_cli._classify(argparse.Namespace(out=str(directory)))
        return 0

    cfg, _public = _kb_config(settings, args.kb)
    if any(todo.values()):
        retriever = base_cli.build_retriever(cfg, settings, index_prefix=args.index_prefix)
        try:
            warm_up(
                RetrieverContexts(retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k),
                [i for i in items if any(i in todo[s] for s in SYSTEMS)],
                max_failed=args.allow_warmup_failures,
            )
        finally:
            close = getattr(retriever, "close", None)
            if callable(close):
                close()
    max_cny = args.max_cny if args.max_cny else base_cli._spent_in(directory) + est.total
    qids = ",".join(i.qid for i in items)
    for system in order:
        if not todo[system]:
            print(f"{system}: nothing to run in pass {args.pass_no}")
            continue
        status = base_cli._run(
            argparse.Namespace(
                kb=args.kb,
                system=system,
                types=",".join(LIST_TYPES),
                take=None,
                qids=qids,
                deadline_ms=None,
                retry_errors=args.retry_errors,
                max_cny=max_cny,
                kb_override=None,
                index_prefix=args.index_prefix,
                out=str(directory),
                pass_no=args.pass_no,
            ),
            settings,
        )
        if status:
            return int(status)
    left = {s: len(v) for s, v in _todo(items, directory, retry=False).items() if v}
    if left:
        print(
            f"pass {args.pass_no} is not complete (questions without a record: {left}); the spend "
            "limit stopped it. Run the same command again to finish it (raise --max-cny if needed)",
            file=sys.stderr,
        )
        return 1
    if not args.no_judge:
        base_cli._judge(
            argparse.Namespace(
                runs=1, model=None, concurrency=JudgeConfig().concurrency, out=str(directory)
            ),
            settings,
        )
    base_cli._classify(argparse.Namespace(out=str(directory)))
    print(
        f"pass {args.pass_no} done. Next: repeat-run --pass {args.pass_no + 1}, or repeat-analyze"
    )
    return 0


# ---- repeat-analyze ------------------------------------------------------------------------------


def _analyze(args: argparse.Namespace) -> int:
    if args.passes < 2:
        raise UsageError("--passes must be 2 or more")
    manifest = _sources(args)
    ql = _question_list(manifest)
    wanted = set(ql.all_qids())
    passes = [load_stored_pass(manifest.pass1_old, manifest.pass1_new, ql)]
    for number in range(2, args.passes + 1):
        directory = pass_dir(args.base, number)
        data = load_pass([directory], wanted)
        if not any(data.values()):
            raise UsageError(f"no records in {directory}; run repeat-run --pass {number} first")
        passes.append(data)
    result = analyze(passes, ql)
    stable_path = Path(args.stable_out) if args.stable_out else Path(args.base) / STABLE_NAME
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    stable_path.write_text(
        json.dumps(stable_payload(result, ql), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {stable_path}")
    section = render_section(
        result,
        ReportContext(
            qlist=ql,
            base=str(args.base),
            list_path=manifest.question_list,
            pass1_old=manifest.pass1_old,
            pass1_new=manifest.pass1_new,
            generated=datetime.now(UTC).date().isoformat(),
        ),
        stable_path=str(stable_path),
    )
    if args.no_report:
        _echo(section)
        return 0
    report = Path(args.report) if args.report else repo_path(REPORT_PATH)
    write_section(report, section)
    print(
        f"wrote the step-3 section into {report} "
        f"({result.questions} questions in the numbers, {len(result.incomplete)} incomplete, "
        f"{len(result.incidents)} cut or failed records)"
    )
    return 0


# ---- parser --------------------------------------------------------------------------------------


def add_parsers(sub: Any) -> None:
    p = sub.add_parser("repeat-sample", help="write the fixed-seed question list of step 3")
    p.add_argument("--kb", default="hf_docs")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--date", default=DEFAULT_DATE)
    p.add_argument("--out", help=f"default {LIST_PATH}")
    p.add_argument(
        "--check", action="store_true", help="only check that the file is what the seed draws"
    )

    def sources(q: argparse.ArgumentParser) -> None:
        q.add_argument("--base", required=True, help="directory of the passes (BASE/pass2, ...)")
        q.add_argument("--list", help=f"question list (default {LIST_PATH})")
        q.add_argument(
            "--pass1-old",
            action="append",
            metavar="DIR",
            help="P3-07 records directory, pass 1 of the old questions; repeat it for a retry "
            "directory (a later one wins). Saved in BASE/repeat.json",
        )
        q.add_argument(
            "--pass1-new",
            action="append",
            metavar="DIR",
            help="EVAL-2 step-2 run directory, pass 1 of the hard and ambiguous questions",
        )

    r = sub.add_parser("repeat-run", help="run one repeated pass (cost estimate first)")
    sources(r)
    r.add_argument("--pass", dest="pass_no", type=int, required=True, help="2, 3, ...")
    r.add_argument("--kb", default="hf_docs")
    r.add_argument("--index-prefix", default="")
    r.add_argument(
        "--estimate-only",
        action="store_true",
        help="print the estimate and the ledger, run nothing",
    )
    r.add_argument("--no-judge", action="store_true", help="run the systems, do not judge")
    r.add_argument(
        "--retry-errors",
        action="store_true",
        help="also run again the questions whose record in this pass is an error, a generation "
        "error or a deadline cut (off by default: such a result is part of what is measured)",
    )
    r.add_argument("--max-cny", type=float, help="spend limit of the pass (default: estimate)")
    r.add_argument(
        "--allow-warmup-failures",
        type=int,
        default=0,
        help="searches that may still fail after the warm-up (default 0)",
    )

    a = sub.add_parser("repeat-analyze", help="pass^k, noise and the report section")
    sources(a)
    a.add_argument("--passes", type=int, default=3)
    a.add_argument("--report", help=f"default {REPORT_PATH}")
    a.add_argument("--no-report", action="store_true", help="print the section, write nothing")
    a.add_argument("--stable-out", help=f"default BASE/{STABLE_NAME}")


def handle(command: str, args: argparse.Namespace, settings: ServiceSettings | None) -> int:
    if command == "repeat-sample":
        return _sample(args)
    if command == "repeat-analyze":
        return _analyze(args)
    return _run(args, settings if settings is not None else ServiceSettings())
