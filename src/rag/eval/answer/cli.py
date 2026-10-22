"""Command line of the answer eval (spec P1-04).

uv run python -m rag.eval.answer run --kb hf_docs [--questions P] [--limit N] [--label L]
    [--index-prefix P] [--contexts-from RUN [--retry-errors]] [--out-dir D]
uv run python -m rag.eval.answer security [--set DIR] [--tokenizer model|whitespace] [--label L]
uv run python -m rag.eval.answer judge --answers RUN [--scales 1-5,1-3] [--repeat 1]
    [--model M] [--concurrency 4] [--label L]
uv run python -m rag.eval.answer stability JUDGE_A JUDGE_B [--scale S] [--out FILE]
uv run python -m rag.eval.answer human-sheet --answers RUN [--n 50] [--seed 0] [--out-dir D]
uv run python -m rag.eval.answer human-agreement --sheet CSV --key CSV --judge RUN [--out FILE]
uv run python -m rag.eval.answer rejudge-sheet --sheet CSV --key CSV --judge RUN [--n 15]
    [--seed 0] [--out-dir D]
uv run python -m rag.eval.answer rejudge-agreement --sheet CSV --key CSV --earlier CSV
uv run python -m rag.eval.answer project --answers PILOT [--judge PILOT_JUDGE] [--questions 65]
    [--judge-calls-per-answer 4] [--security-questions 30]

`run` and `security` call the chat model (`online` profile) once or twice per question and
`judge` once per answer and scale: they cost money and every call goes to the cost ledger,
whose stop line refuses calls past the budget. Records go to the git-ignored
`data/reports/answer-eval/` (`answers/<run_id>/`, `judge/<run_id>/`, `human/<answer run>/`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag.config import KnowledgeBaseConfig, ServiceSettings, repo_path
from rag.cost import CostLedger
from rag.eval.answer.agreement import export_human_sheet, human_agreement
from rag.eval.answer.cost import project_cost
from rag.eval.answer.judge import (
    SCALES,
    JudgeConfig,
    judge_answers,
    judge_stability,
    load_judge_run,
    render_judge_summary,
    write_judge_run,
)
from rag.eval.answer.record import AnswerRunRecord
from rag.eval.answer.rejudge import export_rejudge_sheet, rejudge_agreement
from rag.eval.answer.runner import (
    REPORTS_ROOT,
    ContextSource,
    RetrieverContexts,
    StoredContexts,
    gateway_error_items,
    items_from_question_set,
    load_answer_run,
    merge_retried,
    render_answer_summary,
    run_answer_eval,
    write_answer_run,
)
from rag.eval.answer.security import PlantedContexts, security_items, security_report
from rag.eval.calibration import check_gate, render_calibration
from rag.eval.questions import load_question_set
from rag.eval.retrieval.cli import UsageError, _echo, _kb_config, _question_file, reranker_revision
from rag.eval.retrieval.protocol import Retriever
from rag.eval.retrieval.runner import collect_provenance
from rag.eval.security import DEFAULT_SET_DIR, load_security_set, render_report

# ---- seams (replaced in unit tests) ------------------------------------------------------------


def build_retriever(
    cfg: KnowledgeBaseConfig, settings: ServiceSettings, *, index_prefix: str
) -> Retriever:
    """The chain search_docs uses: `make_search_pipeline` (fusion, rerank, threshold)."""
    from rag.eval.retrieval.cli import build_retriever as build
    from rag.retrieve.hybrid import ROUTES

    return build(cfg, settings, rerank=True, routes=ROUTES, index_prefix=index_prefix)


def build_answer_model(settings: ServiceSettings) -> BaseChatModel:
    from rag.llm import make_chat_model

    return make_chat_model("online", settings)


def judge_settings(settings: ServiceSettings, config: JudgeConfig) -> ServiceSettings:
    """`settings` with the judge's model, output cap and temperature in its profile."""
    profile = getattr(settings, f"chat_{config.profile}").model_copy(
        update={
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
    )
    return settings.model_copy(update={f"chat_{config.profile}": profile})


def build_judge_model(settings: ServiceSettings, config: JudgeConfig) -> BaseChatModel:
    from rag.llm import make_chat_model

    return make_chat_model(config.profile, judge_settings(settings, config))


# ---- commands ----------------------------------------------------------------------------------


def _out_root(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else repo_path(REPORTS_ROOT)


def _models(settings: ServiceSettings, chat: str) -> dict[str, str]:
    models = {"chat": chat, "embed": settings.embed_model, "reranker": settings.reranker_model}
    revision = reranker_revision(settings.reranker_model)
    if revision:
        models["reranker_revision"] = revision
    return models


def _spent(settings: ServiceSettings) -> str:
    ledger = CostLedger.from_settings(settings)
    return (
        f"ledger: spent {ledger.total_spent():.4f} CNY, stop at {settings.budget_stop_cny:.2f} CNY"
    )


def _print_run(run: AnswerRunRecord, out: Path) -> None:
    print(f"wrote {out}")
    _echo(render_answer_summary(run))


def _run(args: argparse.Namespace, settings: ServiceSettings) -> int:
    cfg, _public = _kb_config(settings, args.kb)
    question_set = load_question_set(_question_file(cfg, args.questions), kb=cfg.kb)
    items = items_from_question_set(question_set)
    if args.limit is not None:
        if args.limit < 1:
            raise UsageError("--limit must be a positive integer")
        items = items[: args.limit]
    chat = settings.chat_online.model or settings.chat_model
    print(_spent(settings))
    retriever: Retriever | None = None
    contexts: ContextSource
    stored = load_answer_run(args.contexts_from) if args.contexts_from else None
    if args.retry_errors:
        if stored is None:
            raise UsageError("--retry-errors needs --contexts-from RUN (the run to repair)")
        items = gateway_error_items(stored)
        print(f"retrying {len(items)} gateway errors of {stored.run_id}")
    if stored is not None:
        contexts = StoredContexts(stored)
    else:
        retriever = build_retriever(cfg, settings, index_prefix=args.index_prefix)
        contexts = RetrieverContexts(retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k)
    try:
        run = asyncio.run(
            run_answer_eval(
                items,
                contexts,
                kb_config=cfg,
                model=build_answer_model(settings),
                chat_model=chat,
                label=args.label or "",
                kind="hf",
                question_set={
                    "path": question_set.path,
                    "sha256": question_set.sha256,
                    "count": len(items),
                    "of": len(question_set.questions),
                },
                provenance=collect_provenance(settings=settings, models=_models(settings, chat)),
            )
        )
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    if args.retry_errors and stored is not None:
        run = merge_retried(stored, run.questions, label=args.label or f"{stored.label}-retried")
    out = write_answer_run(run, _out_root(args))
    _print_run(run, out)
    print(_spent(settings))
    return 0


def _security(args: argparse.Namespace, settings: ServiceSettings) -> int:
    sset = load_security_set(args.set)
    cfg, _public = _kb_config(settings, args.kb)
    items = security_items(sset, kb=cfg.kb)
    if args.limit is not None:
        items = items[: args.limit]
    tokenizer = str.split if args.tokenizer == "whitespace" else None
    chat = settings.chat_online.model or settings.chat_model
    print(_spent(settings))
    run = asyncio.run(
        run_answer_eval(
            items,
            PlantedContexts(sset, cfg, tokenizer=tokenizer),
            kb_config=cfg,
            model=build_answer_model(settings),
            chat_model=chat,
            label=args.label or "security",
            kind="security",
            question_set={"path": sset.path, "sha256": sset.sha256, "count": len(items)},
            provenance=collect_provenance(settings=settings, models=_models(settings, chat)),
        )
    )
    out = write_answer_run(run, _out_root(args))
    report = security_report(sset, run)
    (out / "security.json").write_text(
        report.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "security.md").write_text(render_report(report), encoding="utf-8", newline="\n")
    _print_run(run, out)
    for line in report.summary_lines():
        print(line)
    print(_spent(settings))
    return 0


def _scales(text: str) -> list[str]:
    scales = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in scales if s not in SCALES]
    if unknown or not scales:
        raise UsageError(f"--scales: unknown {unknown}; choose from {list(SCALES)}")
    return scales


def _judge(args: argparse.Namespace, settings: ServiceSettings) -> int:
    run = load_answer_run(args.answers)
    scales = _scales(args.scales)
    if args.repeat < 1:
        raise UsageError("--repeat must be a positive integer")
    update: dict[str, Any] = {"concurrency": args.concurrency}
    if args.model:
        update["model"] = args.model
    config = JudgeConfig().model_copy(update=update)
    print(_spent(settings))
    for i in range(1, args.repeat + 1):
        # A new model (and HTTP client) per run: pooled connections belong to the event loop of
        # the `asyncio.run` that opened them and fail once that loop is closed.
        model = build_judge_model(settings, config)
        label = args.label or ""
        if args.repeat > 1:
            label = f"{label}-r{i}" if label else f"r{i}"
        judged = asyncio.run(
            judge_answers(run, model=model, config=config, scales=scales, label=label)
        )
        out = write_judge_run(judged, _out_root(args))
        print(f"wrote {out}")
        _echo(render_judge_summary(judged))
    print(_spent(settings))
    return 0


def _stability(args: argparse.Namespace) -> int:
    a, b = load_judge_run(args.judge_a), load_judge_run(args.judge_b)
    if a.answer_run_id != b.answer_run_id:
        raise UsageError(f"different answer runs: {a.answer_run_id} and {b.answer_run_id}")
    scales = [args.scale] if args.scale else [s for s in a.scales if s in b.scales]
    results = []
    for scale in scales:
        s = judge_stability(a, b, scale)
        results.append(s.model_dump(mode="json"))
        kappa = "n/a" if s.weighted_kappa is None else f"{s.weighted_kappa:.3f}"
        mad = "n/a" if s.mean_abs_diff is None else f"{s.mean_abs_diff:.3f}"
        print(
            f"scale {scale}: judged in both {s.n}, rule-scored left out {s.n_rule_excluded}, "
            f"missing {s.n_missing}; flips: {s.flips.text()}; mean |diff| {mad}; "
            f"max |diff| {s.max_abs_diff}; weighted kappa {kappa}"
        )
        for change in s.changed:
            print(f"  {change['qid']}: {change['a']} -> {change['b']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


def _human_sheet(args: argparse.Namespace) -> int:
    run = load_answer_run(args.answers)
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(REPORTS_ROOT) / "human" / run.run_id
    sheet, key = export_human_sheet(run, n=args.n, seed=args.seed, out_dir=out_dir)
    print(f"wrote {sheet} (to fill, blind) and {key} (do not open before scoring)")
    return 0


def _human_agreement(args: argparse.Namespace) -> int:
    report = human_agreement(args.sheet, args.key, load_judge_run(args.judge))
    print(f"rows {report.rows}, blank {report.blank}, judge {report.judge_model}")
    for s in report.scales.values():
        kappa = "n/a" if s.weighted_kappa is None else f"{s.weighted_kappa:.3f}"
        rho = "n/a" if s.spearman is None else f"{s.spearman:.3f}"
        print(
            f"scale {s.scale}: n {s.n}, weighted kappa {kappa}, Spearman {rho}, "
            f"exact {s.exact.text()}, within one {s.within_one.text()}"
        )
    if report.binary is not None:
        print(render_calibration(report.binary, check_gate(report.binary), title="1-3 scale"))
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return 0


def _rejudge_sheet(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sheet).parent / "rejudge"
    sheet, key = export_rejudge_sheet(
        args.sheet, args.key, load_judge_run(args.judge), n=args.n, seed=args.seed, out_dir=out_dir
    )
    print(f"wrote {sheet} (to fill, blind) and {key} (do not open before scoring)")
    return 0


def _rejudge_agreement(args: argparse.Namespace) -> int:
    report = rejudge_agreement(args.sheet, args.key, args.earlier)
    print(f"rows {report.rows}, blank {report.blank}")
    for s in report.scales.values():
        print(f"scale {s.scale}: n {s.n}, same {s.same}, changed {len(s.changed)}")
        for c in s.changed:
            print(f"  {c.earlier_sample_id}: {c.earlier} -> {c.now}")
    return 0


def _project(args: argparse.Namespace) -> int:
    pilot = load_answer_run(args.answers)
    per_call = args.judge_cny_per_call
    if args.judge:
        judged = load_judge_run(args.judge)
        if judged.usage.calls:
            per_call = judged.usage.cny / judged.usage.calls
    p = project_cost(
        pilot,
        questions=args.questions,
        judge_cny_per_call=per_call,
        judge_calls_per_answer=args.judge_calls_per_answer,
        security_questions=args.security_questions,
    )
    print(
        f"pilot: {p.pilot_questions} questions; per answer prompt {p.mean_prompt_tokens:.0f}, "
        f"completion {p.mean_completion_tokens:.0f} (reasoning {p.mean_reasoning_tokens:.0f}) "
        f"tokens, {p.per_answer_cny:.5f} CNY; judge {p.judge_cny_per_call:.5f} CNY per call"
    )
    print(f"answers ({p.questions}): {p.answers_cny:.4f} CNY")
    print(f"judge ({p.questions} x {p.judge_calls_per_answer} calls): {p.judge_cny:.4f} CNY")
    print(f"security ({p.security_questions}): {p.security_cny:.4f} CNY")
    print(f"total: {p.total_cny:.4f} CNY")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.answer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="answer the KB's question set by the standard RAG path")
    p_run.add_argument("--kb", required=True)
    p_run.add_argument("--questions", help="question set (default: the KB's eval.questions)")
    p_run.add_argument("--limit", type=int, help="first N questions only (a pilot)")
    p_run.add_argument("--label", help="run id suffix")
    p_run.add_argument("--index-prefix", default="")
    p_run.add_argument(
        "--contexts-from",
        metavar="RUN",
        help="answer from the chunks of this earlier answer run instead of retrieving (P7-01)",
    )
    p_run.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --contexts-from: answer again only its gateway errors, write the merged run",
    )
    p_run.add_argument("--out-dir", help=f"default {REPORTS_ROOT}")

    p_sec = sub.add_parser("security", help="answer the security set from its planted pages")
    p_sec.add_argument("--kb", default="hf_docs", help="KB whose pipeline and prompts are used")
    p_sec.add_argument("--set", default=str(DEFAULT_SET_DIR))
    p_sec.add_argument("--tokenizer", choices=("model", "whitespace"), default="model")
    p_sec.add_argument("--limit", type=int)
    p_sec.add_argument("--label")
    p_sec.add_argument("--out-dir")

    p_judge = sub.add_parser("judge", help="score an answer run with the LLM judge")
    p_judge.add_argument("--answers", required=True, help="answer run directory or results.json")
    p_judge.add_argument("--scales", default="1-5,1-3")
    p_judge.add_argument("--repeat", type=int, default=1, help="judge runs on the same answers")
    p_judge.add_argument("--model", help=f"judge model (default {JudgeConfig().model})")
    p_judge.add_argument("--concurrency", type=int, default=JudgeConfig().concurrency)
    p_judge.add_argument("--label")
    p_judge.add_argument("--out-dir")

    p_stab = sub.add_parser("stability", help="score changes between two judge runs")
    p_stab.add_argument("judge_a")
    p_stab.add_argument("judge_b")
    p_stab.add_argument("--scale", choices=list(SCALES))
    p_stab.add_argument("--out", help="also write the result as JSON")

    p_sheet = sub.add_parser("human-sheet", help="blind CSV sheet for human scores")
    p_sheet.add_argument("--answers", required=True)
    p_sheet.add_argument("--n", type=int, default=50)
    p_sheet.add_argument("--seed", type=int, default=0)
    p_sheet.add_argument("--out-dir")

    p_agree = sub.add_parser("human-agreement", help="judge vs filled human sheet")
    p_agree.add_argument("--sheet", required=True)
    p_agree.add_argument("--key", required=True)
    p_agree.add_argument("--judge", required=True, help="judge run directory or results.json")
    p_agree.add_argument("--out", help="also write the report as JSON")

    p_rs = sub.add_parser("rejudge-sheet", help="blind re-score sheet from a filled sheet")
    p_rs.add_argument("--sheet", required=True, help="the filled sheet")
    p_rs.add_argument("--key", required=True)
    p_rs.add_argument("--judge", required=True, help="judge run directory or results.json")
    p_rs.add_argument("--n", type=int, default=15)
    p_rs.add_argument("--seed", type=int, default=0)
    p_rs.add_argument("--out-dir")

    p_ra = sub.add_parser("rejudge-agreement", help="a re-scored sheet against the earlier scores")
    p_ra.add_argument("--sheet", required=True, help="the re-scored sheet")
    p_ra.add_argument("--key", required=True, help="rejudge_key.csv")
    p_ra.add_argument("--earlier", required=True, help="the earlier filled sheet")

    p_proj = sub.add_parser("project", help="project the full cost from a pilot")
    p_proj.add_argument("--answers", required=True, help="pilot answer run")
    p_proj.add_argument("--judge", help="pilot judge run (its CNY per call)")
    p_proj.add_argument("--judge-cny-per-call", type=float, default=0.004)
    p_proj.add_argument("--questions", type=int, default=65)
    p_proj.add_argument("--judge-calls-per-answer", type=int, default=4)
    p_proj.add_argument("--security-questions", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    """Exit codes: 0 ok, 2 usage error."""
    args = _parser().parse_args(argv)
    try:
        if args.command in ("run", "security", "judge"):
            settings = settings if settings is not None else ServiceSettings()
            command = {"run": _run, "security": _security, "judge": _judge}[args.command]
            return command(args, settings)
        command_no_settings = {
            "stability": _stability,
            "human-sheet": _human_sheet,
            "human-agreement": _human_agreement,
            "rejudge-sheet": _rejudge_sheet,
            "rejudge-agreement": _rejudge_agreement,
            "project": _project,
        }[args.command]
        return command_no_settings(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
