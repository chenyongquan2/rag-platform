"""Run the two systems over the questions and record every question (spec P3-05, design s13-5).

    stats = await run_standard(items, records_path=..., contexts=RetrieverContexts(...), ...)
    stats = await run_agentic(items, records_path=..., runtime=AgentRuntime.from_settings(
        settings, hooks=tracer.hooks()), retriever=pipeline, tracer=tracer, ...)

Standard RAG is the P1-04 path: the search chain, `build_context`, `generate_answer`, no agent.
The agentic system is the real `ask_docs` graph (`AgentRuntime.ask`), with the same retriever;
a `Tracer` wraps the route, rewrite and judge hooks so each round's decisions are recorded.
Questions run one after the other (latencies are those of one request at a time; every ledger
line written meanwhile is that question's), each record is appended when the question is done,
questions already in the file are skipped, and the run stops before a question once the spend
(this file's records plus `spent_before`) has reached `max_cny`.

A graph run that stops with `need_clarification` is recorded as a clarification and not
resumed: nothing was answered, so it counts as a refusal (spec P3-05).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from rag.agent import AgentHooks, AgentRuntime
from rag.agent.state import RetrievalRound
from rag.answer import build_context, generate_answer
from rag.config import KnowledgeBaseConfig
from rag.eval.agentic.records import (
    HookEvent,
    RoundInfo,
    SystemName,
    SystemRecord,
    append_line,
    read_records,
    repair_tail,
    usage_by_profile,
)
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.record import AnswerItem, Usage, usage_by_kind
from rag.eval.answer.runner import ContextSource
from rag.eval.answer.usage import collect_ledger_rows
from rag.types import Answer, AnswerStatus, AskResult, Chunk

REPORTS_ROOT = "data/reports/p3-05"
HYDE_KEEP_CHARS = 300


@dataclasses.dataclass
class RunStats:
    ran: int = 0
    skipped: int = 0
    pending: int = 0
    """Questions not started because the spend limit was reached."""
    spent_cny: float = 0.0
    """Spend counted against the limit when the run ended."""
    stopped: str | None = None
    """`budget` when the spend limit ended the run."""


def _pages(chunks: Sequence[Chunk]) -> list[str]:
    return list(dict.fromkeys(c.doc_id for c in chunks))


def _base(item: EvalItem, system: SystemName) -> dict[str, Any]:
    return {
        "system": system,
        "qid": item.qid,
        "qtype": item.qtype,
        "set": item.set,
        "question": item.question,
        "reference": item.reference,
        "relevant_pages": list(item.relevant_pages),
        "requires": item.requires,
        "answerable": item.answerable,
    }


def _usage_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "usage": Usage.from_rows(rows),
        "usage_by_kind": usage_by_kind(rows),
        "usage_by_profile": usage_by_profile(rows),
    }


def _error_record(
    item: EvalItem, system: SystemName, exc: BaseException, elapsed_ms: float, rows: list[Any]
) -> SystemRecord:
    return SystemRecord(
        **_base(item, system),
        status="error",
        error=f"{type(exc).__name__}: {exc}",
        total_ms=round(elapsed_ms, 3),
        **_usage_fields(rows),
    )


def _cited_pages(answer_sentences: Sequence[Any], chunks: Sequence[Chunk]) -> list[str]:
    page_of = {c.chunk_id: c.doc_id for c in chunks}
    cited: list[str] = []
    for sentence in answer_sentences:
        for citation in sentence.citations:
            page = page_of.get(citation.chunk_id, citation.chunk_id.split("#", 1)[0])
            if page not in cited:
                cited.append(page)
    return cited


# ---- standard RAG -------------------------------------------------------------------------------


async def _standard_one(
    item: EvalItem,
    contexts: ContextSource,
    *,
    kb_config: KnowledgeBaseConfig,
    model: Any,
    clock: Callable[[], float],
) -> SystemRecord:
    with collect_ledger_rows() as rows:
        start = clock()
        try:
            ctx = await asyncio.to_thread(
                contexts, AnswerItem(qid=item.qid, question=item.question)
            )
        except Exception as exc:  # recorded per question; one failure must not end the run
            return _error_record(item, "standard", exc, (clock() - start) * 1000.0, rows)
        retrieved = clock()
        context = build_context(
            [ctx.chunks], kb_config.answer.context_tokens, metadata_keys=kb_config.metadata.llm
        )
        answer: Answer = await generate_answer(
            item.question,
            context,
            kb=kb_config.kb,
            config=kb_config.answer,
            model=model,
            request_id=f"p3-05-std-{item.qid}",
        )
        done = clock()
    text = answer.text.strip() if answer.found else ""
    return SystemRecord(
        **_base(item, "standard"),
        status=str(answer.status),
        found=answer.found,
        answered=answer.found and bool(text),
        text=text,
        sentences=len(answer.sentences) if answer.found else 0,
        unsupported_sentences=answer.citation_check.unsupported_sentences,
        answer_parse_failed=answer.answer_parse_failed,
        generation_error=str(answer.generation_error) if answer.generation_error else None,
        retrieved_pages=_pages(ctx.chunks),
        context_pages=_pages(answer.chunks),
        cited_pages=_cited_pages(answer.sentences, answer.chunks),
        retrieval_not_found=ctx.not_found,
        degraded=ctx.degraded,
        rerank_timeout=ctx.rerank_timeout,
        total_ms=round((done - start) * 1000.0, 3),
        steps_ms={
            "retrieval": round((retrieved - start) * 1000.0, 3),
            "generation": round((done - retrieved) * 1000.0, 3),
        },
        **_usage_fields(rows),
    )


# ---- the ask_docs graph -------------------------------------------------------------------------


class Tracer:
    """Records what the route, rewrite and judge hooks decide in each round, per request id.

    `hooks()` returns the graph's hooks with those three wrapped; the wrapped hook is called
    exactly as before and its result passed on unchanged.
    """

    def __init__(self, base: AgentHooks | None = None) -> None:
        self._base = base or AgentHooks()
        self._events: dict[str, list[HookEvent]] = {}

    def pop(self, request_id: str) -> list[HookEvent]:
        return self._events.pop(request_id, [])

    def _add(self, request_id: str, step: str, round_no: int, data: dict[str, Any]) -> None:
        self._events.setdefault(request_id, []).append(
            HookEvent(step=step, round=round_no, data=data)
        )

    def hooks(self) -> AgentHooks:
        base = self._base

        async def route_fallback(state: Any, ctx: Any, decision: Any, wants: Any) -> Any:
            out = await base.route_fallback(state, ctx, decision, wants)
            self._add(
                ctx.request_id,
                "route_model",
                0,
                {
                    "wants": sorted(wants),
                    "clarity": out.clarity,
                    "multi_hop": out.multi_hop,
                    "two_hop": out.two_hop,
                    "needs_clarification": out.needs_clarification,
                    "fallback": out.fallback,
                },
            )
            return out

        async def rewrite(state: Any, ctx: Any) -> Any:
            out = base.rewrite(state, ctx)
            if inspect.isawaitable(out):
                out = await out
            queries = [
                f"hyde passage: {q.text[:HYDE_KEEP_CHARS]}" if q.kind == "hyde" else q.text
                for q in out.queries
            ] or list(out.sub_questions)
            self._add(
                ctx.request_id,
                "rewrite",
                int(state.get("round", 0)) + 1,
                {
                    "standalone": out.standalone_question,
                    "queries": queries,
                    "kinds": [q.kind for q in out.queries],
                    "technique": out.technique,
                    "fallback": out.fallback,
                    "used_model": out.used_model,
                    "exhausted": out.exhausted,
                },
            )
            return out

        async def judge(state: Any, ctx: Any) -> Any:
            out = base.judge(state, ctx)
            if inspect.isawaitable(out):
                out = await out
            verdict = getattr(out, "verdict", out)
            self._add(
                ctx.request_id,
                "judge",
                int(state.get("round", 0)),
                {
                    "verdict": verdict,
                    "missing": list(getattr(out, "missing", ())),
                    "fallback": getattr(out, "fallback", None),
                    "used_model": bool(getattr(out, "used_model", False)),
                },
            )
            return out

        return dataclasses.replace(
            base, route_fallback=route_fallback, rewrite=rewrite, judge=judge
        )


def _round_infos(rounds: Sequence[RetrievalRound]) -> list[RoundInfo]:
    return [
        RoundInfo(
            round=r.round,
            sub_question=r.sub_question,
            query=r.query,
            kind=r.kind,
            pages=_pages(r.chunks),
            n_chunks=len(r.chunks),
            top_score=r.top_rerank_score,
            not_found=r.not_found,
        )
        for r in rounds
    ]


def _agentic_record(
    item: EvalItem,
    result: AskResult,
    values: dict[str, Any],
    events: list[HookEvent],
    rows: list[Any],
    elapsed_ms: float,
) -> SystemRecord:
    clarification = result.status == AnswerStatus.NEED_CLARIFICATION
    rounds: list[RetrievalRound] = list(values.get("rounds") or [])
    text = "" if clarification else result.answer.strip()
    steps = {k: v for k, v in result.timings_ms.items() if k != "total"}
    return SystemRecord(
        **_base(item, "agentic"),
        status=str(result.status),
        found=result.found and not clarification,
        clarification=clarification,
        answered=result.found and not clarification and bool(text),
        text=text,
        sentences=len(result.sentences) if result.found else 0,
        unsupported_sentences=result.citation_check.unsupported_sentences,
        answer_parse_failed=result.answer_parse_failed,
        generation_error=str(result.generation_error) if result.generation_error else None,
        retrieved_pages=_pages([c for r in rounds for c in r.chunks]),
        context_pages=_pages(result.sources) if not clarification else [],
        cited_pages=_cited_pages(result.sentences, result.sources),
        retrieval_not_found=all(r.not_found for r in rounds) if rounds else False,
        total_ms=round(elapsed_ms, 3),
        steps_ms=dict(steps),
        rounds=result.rounds,
        stop_reason=result.stop_reason,
        coverage_unconfirmed=result.coverage_unconfirmed,
        degraded=result.degraded,
        rerank_timeout=result.rerank_timeout,
        fallbacks=list(result.fallbacks),
        error_categories=[str(c) for c in result.error_categories],
        sufficiency=result.sufficiency,
        missing=list(result.missing),
        standalone_question=result.standalone_question,
        search_queries=list(result.search_queries),
        clarify_question=result.answer if clarification else None,
        two_hop=bool(values.get("two_hop")),
        tokens_used=result.tokens_used,
        round_info=_round_infos(rounds),
        events=events,
        **_usage_fields(rows),
    )


async def _agentic_one(
    item: EvalItem,
    *,
    runtime: AgentRuntime,
    retriever: Any,
    kb_config: KnowledgeBaseConfig,
    tracer: Tracer,
    deadline_ms: int | None,
    models: dict[str, Any],
    clock: Callable[[], float],
) -> SystemRecord:
    request_id = f"p3-05-{item.qid}"
    with collect_ledger_rows() as rows:
        start = clock()
        try:
            result = await runtime.ask(
                item.question,
                cfg=kb_config,
                retriever=retriever,
                request_id=request_id,
                deadline_ms=deadline_ms,
                **models,
            )
        except Exception as exc:  # recorded per question; one failure must not end the run
            tracer.pop(request_id)
            return _error_record(item, "agentic", exc, (clock() - start) * 1000.0, rows)
        elapsed = (clock() - start) * 1000.0
        try:  # the checkpoint holds every round's chunks (the result has the answer's only)
            values = await runtime.state_values(result.thread_id)
        except Exception:
            values = {}
    return _agentic_record(item, result, values, tracer.pop(request_id), rows, elapsed)


# ---- the loop shared by both systems ------------------------------------------------------------


async def _run(
    items: Sequence[EvalItem],
    records_path: Path,
    one: Callable[[EvalItem], Awaitable[SystemRecord]],
    *,
    max_cny: float | None,
    spent_before: float,
    progress: Callable[[int, int, SystemRecord], None] | None,
    retry_errors: bool = False,
) -> RunStats:
    repair_tail(records_path)
    existing = read_records(records_path)
    if retry_errors and any(r.status == "error" for r in existing):
        # An errored question counts as done unless asked: drop those records and run them again
        # (what they cost stays in the ledger; the file no longer counts it).
        existing = [r for r in existing if r.status != "error"]
        records_path.write_text(
            "".join(r.model_dump_json() + "\n" for r in existing), encoding="utf-8", newline=""
        )
    done = {r.qid for r in existing}
    spent = spent_before + sum(r.usage.cny for r in existing)
    stats = RunStats(skipped=sum(1 for i in items if i.qid in done))
    todo = [i for i in items if i.qid not in done]
    for index, item in enumerate(todo, start=1):
        if max_cny is not None and spent >= max_cny:
            stats.pending = len(todo) - stats.ran
            stats.stopped = "budget"
            break
        record = await one(item)
        append_line(records_path, record)
        spent += record.usage.cny
        stats.ran += 1
        if progress is not None:
            progress(index, len(todo), record)
    stats.spent_cny = round(spent, 8)
    return stats


async def run_standard(
    items: Sequence[EvalItem],
    *,
    records_path: Path,
    contexts: ContextSource,
    kb_config: KnowledgeBaseConfig,
    model: Any,
    max_cny: float | None = None,
    spent_before: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[int, int, SystemRecord], None] | None = None,
    retry_errors: bool = False,
) -> RunStats:
    """Standard RAG over `items`: search chain, context, one answer call."""

    async def one(item: EvalItem) -> SystemRecord:
        return await _standard_one(item, contexts, kb_config=kb_config, model=model, clock=clock)

    return await _run(
        items,
        records_path,
        one,
        max_cny=max_cny,
        spent_before=spent_before,
        progress=progress,
        retry_errors=retry_errors,
    )


async def run_agentic(
    items: Sequence[EvalItem],
    *,
    records_path: Path,
    runtime: AgentRuntime,
    retriever: Any,
    kb_config: KnowledgeBaseConfig,
    tracer: Tracer,
    max_cny: float | None = None,
    spent_before: float = 0.0,
    deadline_ms: int | None = None,
    model: Any = None,
    helper_model: Any = None,
    judge_model: Any = None,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[int, int, SystemRecord], None] | None = None,
    retry_errors: bool = False,
) -> RunStats:
    """The ask_docs graph over `items` (`runtime` built with `tracer.hooks()`); models default
    to the runtime's: `online` for the answer, `cheap` for route, rewrite and judge."""
    models = {"model": model, "helper_model": helper_model, "judge_model": judge_model}

    async def one(item: EvalItem) -> SystemRecord:
        return await _agentic_one(
            item,
            runtime=runtime,
            retriever=retriever,
            kb_config=kb_config,
            tracer=tracer,
            deadline_ms=deadline_ms,
            models=models,
            clock=clock,
        )

    return await _run(
        items,
        records_path,
        one,
        max_cny=max_cny,
        spent_before=spent_before,
        progress=progress,
        retry_errors=retry_errors,
    )
