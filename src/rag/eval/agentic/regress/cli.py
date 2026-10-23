"""Commands of the regression set (spec EVAL-4, plan step 4), part of `python -m rag.eval.agentic`.

    regress-build --base DIR [--seed N] [--date D] [--max-single-hop N] [--out FILE]
                  [--baseline-out FILE] [--force] [--kb hf_docs]
    regress       [--set FILE] [--baseline FILE] [--out DIR] [--with-standard] [--estimate-only]
                  [--max-attempts 5] [--max-cny X] [--allow-warmup-failures N]
                  [--kb-override SECTION.KEY=VALUE]... [--index-prefix P] [--kb hf_docs]

`regress-build` writes the set and its baseline from the three passes of the repeated runs
(EVAL-3), once. `regress` is the command to run after a change to the agent, the answer step or
the retrieval configuration. It prints the estimated cost and the ledger, refuses when the first
attempt would pass the stop line, warms up retrieval (no chat model), runs the set on the current
code, judges it, and applies the comparison rule of `verdict.py`: a question that is not correct is
run again, and it is a regression only when fewer than 2 of 3 clean attempts are correct. Deadline
cuts and gateway errors are run problems, never regressions. Exit codes: 0 pass, 1 regression,
2 refused or usage error (nothing was decided), 3 inconclusive (run problems left).
`--kb-override` changes one configuration key for this run only (for checking the tool by making
something worse on purpose); the output says so.

The runs go through the seams of `rag.eval.agentic.cli` (`_run`, `_judge`, `build_retriever`),
so unit tests replace them with fakes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.config import ServiceSettings
from rag.config.paths import REPO_ROOT, repo_path, shared_data_dir
from rag.cost import CostLedger
from rag.eval.agentic.records import SYSTEMS, read_records, system_file
from rag.eval.agentic.regress.suite import (
    BASELINE_PATH,
    DEFAULT_DATE,
    DEFAULT_MAX_PER_TYPE,
    DEFAULT_SEED,
    SET_PATH,
    Baseline,
    RegressionSet,
    build_suite,
    check_items,
    read_baseline,
    read_suite,
    write_baseline,
    write_suite,
)
from rag.eval.agentic.regress.verdict import Attempt, classify_attempt, decide
from rag.eval.agentic.repeat import cli as repeat_cli
from rag.eval.agentic.repeat.passes import (
    MARGIN,
    PassData,
    load_manifest,
    load_pass,
    load_stored_pass,
    pass_dir,
    warm_up,
)
from rag.eval.agentic.repeat.sample import LIST_TYPES
from rag.eval.agentic.sets import EvalItem, load_items
from rag.eval.answer.judge import JudgeConfig
from rag.eval.answer.runner import RetrieverContexts
from rag.eval.retrieval.cli import UsageError, _kb_config

DEFAULT_MAX_ATTEMPTS = 5
BROKEN_STAGE_MIN = 3
"""A stage of at least this many questions that is all run problems means the environment is
broken: it is not tried again."""
GUARD_ORDER: tuple[str, ...] = ("agentic", "standard")


def _base() -> Any:
    """`rag.eval.agentic.cli`, imported late: it imports this module to register the commands."""
    from rag.eval.agentic import cli

    return cli


# ---- regress-build -------------------------------------------------------------------------------


def _build(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.base)
    ql = repeat_cli._question_list(manifest)
    items = load_items(args.kb, types=list(LIST_TYPES))
    wanted = set(ql.all_qids())
    passes: list[PassData] = [load_stored_pass(manifest.pass1_old, manifest.pass1_new, ql)]
    directories: list[str] = []
    for number in range(2, args.passes + 1):
        directory = pass_dir(args.base, number)
        data = load_pass([directory], wanted)
        if not any(data.values()):
            raise UsageError(f"no records in {directory}")
        passes.append(data)
        directories.append(directory.as_posix())
    cap = (
        DEFAULT_MAX_PER_TYPE if args.max_single_hop is None else {"single_hop": args.max_single_hop}
    )
    rs, baseline = build_suite(
        passes,
        qlist=ql,
        items=items,
        seed=args.seed,
        date=args.date,
        max_per_type=cap,
        sources={
            "question_list": manifest.question_list,
            "pass1_old": manifest.pass1_old,
            "pass1_new": manifest.pass1_new,
            "later_passes": directories,
            "stable_file": (Path(args.base) / repeat_cli.STABLE_NAME).as_posix(),
        },
    )
    target = repo_path(args.out or SET_PATH)
    base_target = repo_path(args.baseline_out or BASELINE_PATH)
    for path in (target, base_target):
        if path.is_file() and not args.force:
            raise UsageError(
                f"{path} exists; a set is not edited once used (a new version is a new file). "
                "Use --force only before the first regression run"
            )
    write_suite(target, rs)
    write_baseline(base_target, baseline)
    print(f"wrote {target} and {base_target}")
    for system, by_type in rs.stable_available.items():
        print(f"stable for {system}: {by_type}")
    print(f"in the set: {rs.counts} ({len(rs.questions)} questions)")
    both = len(rs.qids_for("standard"))
    print(f"also stable for standard RAG: {both} of {len(rs.questions)}")
    return 0


# ---- regress: estimate and budget ----------------------------------------------------------------


@dataclass
class Estimate:
    by_system: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    margin: float = MARGIN

    @property
    def subtotal(self) -> float:
        return sum(self.by_system.values())

    @property
    def total(self) -> float:
        """With the margin: the number checked against the stop line."""
        return self.subtotal * (1 + self.margin)

    @property
    def upper_bound(self) -> float:
        """Every question run twice more."""
        return 3 * self.total

    def lines(self) -> list[str]:
        out = ["estimated cost of this run (CNY):"]
        for system, cny in self.by_system.items():
            out.append(
                f"  {system}: {cny:.4f} ({self.counts[system]} questions, "
                "cost per question from the baseline passes)"
            )
        out.append(
            f"  with {self.margin:.0%} margin {self.total:.4f}; "
            f"upper bound if every question is run twice more {self.upper_bound:.4f}"
        )
        return out


def estimate(guard: Mapping[str, Sequence[str]], baseline: Baseline) -> Estimate:
    """The cost of running `guard` (qids per system) once, from what the baseline passes cost."""
    est = Estimate()
    for system, qids in guard.items():
        known = [
            m for q in baseline.entries if (m := baseline.mean_cny(system, q)) is not None and m > 0
        ]
        fallback = sum(known) / len(known) if known else 0.0
        total = 0.0
        for qid in qids:
            mean = baseline.mean_cny(system, qid)
            total += mean if mean is not None else fallback
        est.by_system[system] = total
        est.counts[system] = len(qids)
    return est


# ---- regress: result -----------------------------------------------------------------------------

STATUSES = ("pass", "recovered", "regression", "run_problem")


class QuestionResult(BaseModel):
    system: str
    qid: str
    qtype: str
    status: str
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    baseline_correct: int = 0
    baseline_passes: int = 0


class RegressResult(BaseModel):
    set: str
    set_date: str
    started: str
    code_revision: str
    code_dirty: bool
    overrides: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
    verdict: str
    counts: dict[str, int] = Field(default_factory=dict)
    spent_cny: float = 0.0
    estimated_cny: float = 0.0
    ledger_after_cny: float = 0.0
    notes: list[str] = Field(default_factory=list)
    questions: list[QuestionResult] = Field(default_factory=list)


def _revision() -> tuple[str, bool]:
    """Short commit of the checkout and whether `src` or `kb` has uncommitted changes."""

    def git(*argv: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *argv],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    head = git("rev-parse", "--short", "HEAD") or "unknown"
    changed = git("status", "--porcelain", "--", "src", "kb")
    return head, bool(changed)


def _exit_code(verdict: str) -> int:
    return {"PASS": 0, "REGRESSION": 1, "INCONCLUSIVE": 3}[verdict]


def render(result: RegressResult) -> str:
    out = [
        f"regression check: set {result.set} ({result.set_date}), systems: "
        f"{', '.join(result.systems)}",
        f"code revision {result.code_revision}"
        + (" with uncommitted changes in src or kb" if result.code_dirty else ""),
    ]
    if result.overrides:
        out.append(
            f"NOTE: this run used the overrides {result.overrides}; "
            "it is NOT the committed configuration"
        )
    counts = ", ".join(f"{s.replace('_', ' ')} {result.counts.get(s, 0)}" for s in STATUSES)
    out.append(f"questions: {counts}")
    out.append(
        f"cost of this run {result.spent_cny:.4f} CNY (estimate with margin "
        f"{result.estimated_cny:.4f}); ledger now {result.ledger_after_cny:.4f} CNY"
    )

    def listing(status: str, title: str) -> None:
        rows = [q for q in result.questions if q.status == status]
        if not rows:
            return
        out.append(f"{title}:")
        for q in rows:
            tries = ", ".join(
                a["outcome"] + (f" ({a['reason']})" if a.get("reason") else "") for a in q.attempts
            )
            out.append(
                f"  {q.system} {q.qid} [{q.qtype}] attempts: {tries}; "
                f"baseline {q.baseline_correct} of {q.baseline_passes} correct"
            )

    listing("regression", "REGRESSIONS (fewer than 2 of 3 clean attempts correct)")
    listing("run_problem", "Run problems (not regressions; the question could not be decided)")
    listing("recovered", "Recovered (not correct at first, then correct: flaky, not a regression)")
    for note in result.notes:
        out.append(f"note: {note}")
    out.append(f"RESULT: {result.verdict}")
    return "\n".join(out) + "\n"


# ---- regress -------------------------------------------------------------------------------------


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    set_path = repo_path(args.set or SET_PATH)
    if args.baseline:
        return set_path, repo_path(args.baseline)
    if args.set:
        return set_path, set_path.with_name(set_path.stem + ".baseline.json")
    return set_path, repo_path(BASELINE_PATH)


def _stage_namespace(
    args: argparse.Namespace,
    rs: RegressionSet,
    system: str,
    qids: Sequence[str],
    out: Path,
    cap: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        kb=args.kb,
        system=system,
        types=",".join(rs.counts),
        take=None,
        qids=",".join(qids),
        deadline_ms=None,
        retry_errors=False,
        max_cny=cap,
        kb_override=list(args.kb_override or []) or None,
        index_prefix=args.index_prefix,
        out=str(out),
        pass_no=None,
    )


def _regress(args: argparse.Namespace, settings: ServiceSettings) -> int:
    if args.max_attempts < 1:
        raise UsageError("--max-attempts must be at least 1")
    if args.max_cny is not None and args.max_cny <= 0:
        raise UsageError("--max-cny must be positive")
    base_cli = _base()
    set_path, baseline_path = _paths(args)
    if not set_path.is_file() or not baseline_path.is_file():
        raise UsageError(f"no set or baseline at {set_path} and {baseline_path}")
    rs, baseline = read_suite(set_path), read_baseline(baseline_path)
    items = [i for i in load_items(args.kb, types=list(rs.counts))]
    try:
        check_items(rs, items)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    cfg, _public = _kb_config(settings, args.kb)
    cfg = base_cli.apply_kb_overrides(cfg, args.kb_override)

    guard = {"agentic": rs.qids_for("agentic")}
    if args.with_standard:
        guard["standard"] = rs.qids_for("standard")
    by_qid = {i.qid: i for i in items}
    qtypes = rs.qtype_of()
    est = estimate(guard, baseline)
    for line in est.lines():
        print(line)
    ledger = CostLedger.from_settings(settings)
    stop = settings.budget_stop_cny
    spent = ledger.total_spent()
    print(f"ledger: spent {spent:.4f} CNY, stop at {stop:.2f} CNY")
    run_cap = args.max_cny if args.max_cny is not None else est.upper_bound
    if spent + est.total > stop + 1e-9:
        raise UsageError(
            f"spent {spent:.4f} + estimate with margin {est.total:.4f} CNY would pass the stop "
            f"line {stop:.2f} CNY; not started. Raise the line (owner) or guard fewer questions"
        )
    if est.total > run_cap + 1e-9:
        raise UsageError(
            f"the estimate with margin {est.total:.4f} CNY is above --max-cny {run_cap:.4f}"
        )
    if args.estimate_only:
        return 0
    if args.kb_override:
        print(f"kb overrides for this run: {args.kb_override} (NOT the committed configuration)")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else shared_data_dir() / "regress" / stamp
    started = datetime.now(UTC).isoformat(timespec="seconds")
    wanted_items: list[EvalItem] = [by_qid[q] for q in rs.qids()]

    retriever = base_cli.build_retriever(cfg, settings, index_prefix=args.index_prefix)
    try:
        try:
            warm_up(
                RetrieverContexts(retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k),
                wanted_items,
                max_failed=args.allow_warmup_failures,
            )
        except UsageError as exc:
            raise UsageError(f"run problem, not a regression (nothing was decided): {exc}") from exc
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()

    attempts: dict[tuple[str, str], list[Attempt]] = {
        (s, q): [] for s, qids in guard.items() for q in qids
    }
    notes: list[str] = []
    run_spent = 0.0
    judge_config = JudgeConfig()
    for number in range(1, args.max_attempts + 1):
        todo = {
            s: [
                q
                for q in qids
                if decide(attempts[(s, q)], max_attempts=args.max_attempts) == "pending"
            ]
            for s, qids in guard.items()
        }
        if not any(todo.values()):
            break
        directory = out / f"attempt{number}"
        if number > 1:
            stage = estimate(todo, baseline)
            now = ledger.total_spent()
            if now + stage.total > stop + 1e-9:
                notes.append(
                    f"attempt {number} was not run: ledger {now:.4f} + {stage.total:.4f} CNY "
                    f"would pass the stop line {stop:.2f}; the questions still undecided stay "
                    "run problems"
                )
                break
            if run_spent + stage.total > run_cap + 1e-9:
                notes.append(
                    f"attempt {number} was not run: the spend limit {run_cap:.4f} CNY of this run "
                    "would be passed; the questions still undecided stay run problems"
                )
                break
        print(
            f"attempt {number}: "
            + ", ".join(f"{s} {len(q)} questions" for s, q in todo.items() if q)
        )
        for system in GUARD_ORDER:
            qids = todo.get(system) or []
            if not qids:
                continue
            cap = base_cli._spent_in(directory) + max(run_cap - run_spent, 0.0)
            base_cli._run(_stage_namespace(args, rs, system, qids, directory, cap), settings)
        if any(read_records(system_file(directory, s)) for s in SYSTEMS):
            base_cli._judge(
                argparse.Namespace(
                    runs=1, model=None, concurrency=judge_config.concurrency, out=str(directory)
                ),
                settings,
            )
        run_spent += base_cli._spent_in(directory)
        data = load_pass([directory], set(rs.qids()))
        fresh: list[Attempt] = []
        for system, qids in todo.items():
            for qid in qids:
                attempt = classify_attempt(qtypes[qid], data[system].get(qid))
                attempts[(system, qid)].append(attempt)
                fresh.append(attempt)
        if len(fresh) >= BROKEN_STAGE_MIN and all(a.outcome == "problem" for a in fresh):
            notes.append(
                f"the environment looks broken: every one of the {len(fresh)} questions of attempt "
                f"{number} was a run problem; not trying again. Check Docker, Elasticsearch, the "
                "reranker and the gateway"
            )
            break

    results: list[QuestionResult] = []
    for (system, qid), tries in attempts.items():
        status = decide(tries, max_attempts=args.max_attempts)
        if status == "pending":
            status = "run_problem"
        rows = baseline.entries[qid][system]
        results.append(
            QuestionResult(
                system=system,
                qid=qid,
                qtype=qtypes[qid],
                status=status,
                attempts=[
                    {"attempt": n, "outcome": a.outcome, "reason": a.reason}
                    for n, a in enumerate(tries, start=1)
                ],
                baseline_correct=sum(r.correct for r in rows),
                baseline_passes=len(rows),
            )
        )
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}
    verdict = (
        "REGRESSION"
        if counts["regression"]
        else "INCONCLUSIVE"
        if counts["run_problem"]
        else "PASS"
    )
    revision, dirty = _revision()
    result = RegressResult(
        set=rs.name,
        set_date=rs.date,
        started=started,
        code_revision=revision,
        code_dirty=dirty,
        overrides=list(args.kb_override or []),
        systems=list(guard),
        verdict=verdict,
        counts=counts,
        spent_cny=round(run_spent, 6),
        estimated_cny=round(est.total, 6),
        ledger_after_cny=round(ledger.total_spent(), 6),
        notes=notes,
        questions=results,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    text = render(result)
    (out / "result.md").write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    print(f"details in {out}")
    return _exit_code(verdict)


# ---- parser --------------------------------------------------------------------------------------


def add_parsers(sub: Any) -> None:
    b = sub.add_parser("regress-build", help="write the regression set and baseline (once)")
    b.add_argument("--base", required=True, help="directory of the repeated passes (EVAL-3)")
    b.add_argument("--kb", default="hf_docs")
    b.add_argument("--passes", type=int, default=3)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--date", default=DEFAULT_DATE)
    b.add_argument(
        "--max-single-hop",
        type=int,
        help=f"cap of single-hop questions (default {DEFAULT_MAX_PER_TYPE['single_hop']})",
    )
    b.add_argument("--out", help=f"default {SET_PATH}")
    b.add_argument("--baseline-out", help=f"default {BASELINE_PATH}")
    b.add_argument(
        "--force", action="store_true", help="overwrite (before the first regression run)"
    )

    r = sub.add_parser("regress", help="run the regression set on the current code and compare")
    r.add_argument("--set", help=f"default {SET_PATH}")
    r.add_argument("--baseline", help="default: next to the set")
    r.add_argument("--out", help="records and result directory (default: shared data dir/regress)")
    r.add_argument("--kb", default="hf_docs")
    r.add_argument("--index-prefix", default="")
    r.add_argument(
        "--with-standard",
        action="store_true",
        help="also run standard RAG on the questions that are stable for it",
    )
    r.add_argument("--estimate-only", action="store_true", help="print the estimate, run nothing")
    r.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    r.add_argument(
        "--max-cny", type=float, help="spend limit of the whole run (default: upper bound)"
    )
    r.add_argument(
        "--allow-warmup-failures",
        type=int,
        default=0,
        help="searches that may still fail after the warm-up (default 0)",
    )
    r.add_argument(
        "--kb-override",
        action="append",
        metavar="SECTION.KEY=VALUE",
        help="set one kb config key for this run only; for checking the tool, not for real runs",
    )


def handle(command: str, args: argparse.Namespace, settings: ServiceSettings | None) -> int:
    if command == "regress-build":
        return _build(args)
    return _regress(args, settings if settings is not None else ServiceSettings())
