"""Run the judge over question sets and record verdict and label (spec P3-03).

    file = collect_records(specs, retriever=..., cfg=..., settings=..., judge_model=...)

For every question: one search through `retriever` (the chain `search_docs` uses, threshold and
all), the label (`metrics.label_enough` on the pages it returned), then the judge as the graph
runs it (`rag.agent.judge.judge_rounds` in a real `AgentContext`, so the prompt, the token cap,
the timeout and the fallbacks are the production ones). With `rescue`, a question that was not
enough on round one (and has a gold page) also gets round two and, if that did not find the page,
round three, written by the production correction code (`corrective_queries`) and searched the
same way: it measures what another round is worth. The run stops before a question when the
ledger says this run has spent `max_cny` (the file is then `partial`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rag.agent.budget import TokenBudget
from rag.agent.context import AgentContext
from rag.agent.correction import corrective_queries
from rag.agent.hooks import AgentHooks
from rag.agent.judge import judge_rounds
from rag.agent.state import RetrievalRound
from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.eval.judge_accuracy import metrics
from rag.eval.judge_accuracy.records import JudgeFile, JudgeRecord, Kind, RescueRecord
from rag.eval.questions import LabeledQuestion
from rag.retrieve.deadline import Deadline
from rag.types import RetrievalResult

_TOTAL_MS = 600_000
"""Time limit of the context one question's calls run in: generous, the judge's own timeout
is what binds."""


@dataclass(frozen=True)
class QuestionSpec:
    set: str
    question: LabeledQuestion
    kind: Kind
    qtype: str


def _context(
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    retriever: Any,
    judge_model: Any,
    helper_model: Any,
    request_id: str,
) -> AgentContext:
    return AgentContext(
        settings=settings,
        cfg=cfg,
        retriever=retriever,
        deadline=Deadline.after_ms(_TOTAL_MS),
        total_ms=_TOTAL_MS,
        budget=TokenBudget(settings.agent_token_cap, 0),
        model=None,
        helper_model=helper_model,
        judge_model=judge_model,
        hooks=AgentHooks(),
        request_id=request_id,
        session_id=None,
        thread_id=f"-:{request_id}",
    )


def _pages(rounds: Sequence[RetrievalRound]) -> list[str]:
    return [c.doc_id for r in rounds for c in r.chunks]


def _top_score(result: RetrievalResult) -> float | None:
    scores = [h.score for h in result.rerank if h.score is not None]
    return max(scores) if scores else None


async def _rescue(
    spec: QuestionSpec,
    rounds: list[RetrievalRound],
    *,
    retriever: Any,
    cfg: KnowledgeBaseConfig,
    ctx: AgentContext,
    spent_cny: Callable[[], float] | None,
) -> RescueRecord:
    """Round two, and round three when round two did not find the gold page, as the graph
    would write and search them."""
    question = spec.question
    began = spent_cny() if spent_cny else 0.0
    state: dict[str, Any] = {
        "question": question.question,
        "rounds": list(rounds),
        "round": 1,
        "missing": [question.question],
    }
    queries: list[str] = []
    fallback: str | None = None
    found: list[bool] = []
    for round_no in (2, 3):
        outcome = await corrective_queries(state, ctx, missing=[question.question])
        fallback = fallback or outcome.fallback
        if not outcome.queries:
            found.append(False)
            break
        for query in outcome.queries:
            result = retriever.retrieve(query.text, kb=cfg.kb, top_k=cfg.retrieval.final_k)
            state["rounds"].append(
                RetrievalRound.from_result(
                    round_no, query.owner, result, query=query.text, kind=query.kind
                )
            )
            queries.append(query.text)
        state["round"] = round_no
        found.append(metrics.label_enough(question, _pages(state["rounds"])))
        if found[-1]:
            break
    round2 = bool(found and found[0])
    round3 = found[1] if len(found) > 1 and not round2 else None
    return RescueRecord(
        round2_found=round2,
        round3_found=round3,
        queries=queries,
        fallback=fallback,
        cny=round((spent_cny() if spent_cny else 0.0) - began, 8),
    )


async def _one(
    spec: QuestionSpec,
    *,
    retriever: Any,
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    judge_model: Any,
    helper_model: Any,
    rescue: bool,
    spent_cny: Callable[[], float] | None,
) -> JudgeRecord:
    question = spec.question
    k = cfg.retrieval.final_k
    result = retriever.retrieve(question.question, kb=cfg.kb, top_k=k)
    rounds = [RetrievalRound.from_result(1, question.question, result)]
    ctx = _context(cfg, settings, retriever, judge_model, helper_model, f"judge-{question.qid}")
    began = time.perf_counter()
    outcome = await judge_rounds(ctx, rounds)
    latency = (time.perf_counter() - began) * 1000.0
    returned = [c.doc_id for c in result.chunks]
    enough = spec.kind == "answerable" and metrics.label_enough(question, returned)
    reranked = [h.doc_id for h in result.rerank[:k]]
    record = JudgeRecord(
        set=spec.set,
        qid=question.qid,
        question=question.question,
        qtype=spec.qtype,
        kind=spec.kind,
        relevant_pages=list(question.relevant_pages),
        requires="all" if question.meta.get("requires") == "all" else "any",
        label_enough=enough,
        label_in_rerank_top_k=spec.kind == "answerable"
        and metrics.label_enough(question, reranked),
        n_chunks=len(result.chunks),
        top_score=_top_score(result),
        returned_pages=returned,
        verdict=outcome.verdict,
        missing=list(outcome.missing),
        fallback=outcome.fallback,
        used_model=outcome.used_model,
        tokens=ctx.budget.spent,
        latency_ms=round(latency, 1),
    )
    if rescue and spec.kind == "answerable" and not enough:
        rescued = await _rescue(
            spec, rounds, retriever=retriever, cfg=cfg, ctx=ctx, spent_cny=spent_cny
        )
        record = record.model_copy(update={"rescue": rescued})
    return record


async def _collect(
    specs: Sequence[QuestionSpec],
    *,
    retriever: Any,
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    judge_model: Any,
    helper_model: Any,
    rescue: bool,
    max_cny: float | None,
    spent_cny: Callable[[], float] | None,
    progress: Callable[[int, int, JudgeRecord], None] | None,
) -> JudgeFile:
    start = spent_cny() if spent_cny else 0.0
    records: list[JudgeRecord] = []
    partial = False
    for index, spec in enumerate(specs, start=1):
        before = spent_cny() if spent_cny else 0.0
        if max_cny is not None and spent_cny is not None and before - start >= max_cny:
            partial = True
            break
        record = await _one(
            spec,
            retriever=retriever,
            cfg=cfg,
            settings=settings,
            judge_model=judge_model,
            helper_model=helper_model,
            rescue=rescue,
            spent_cny=spent_cny,
        )
        if spent_cny is not None:
            record = record.model_copy(update={"cny": round(spent_cny() - before, 8)})
        records.append(record)
        if progress is not None:
            progress(index, len(specs), record)
    end = spent_cny() if spent_cny else 0.0
    return JudgeFile(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        kb=cfg.kb,
        threshold=cfg.retrieval.threshold,
        k=cfg.retrieval.final_k,
        max_rounds=cfg.agent.max_rounds,
        judge_profile=settings.agent_judge_profile,
        partial=partial,
        spent_cny=round(end - start, 8),
        records=records,
    )


def collect_records(
    specs: Sequence[QuestionSpec],
    *,
    retriever: Any,
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    judge_model: Any,
    helper_model: Any = None,
    rescue: bool = False,
    max_cny: float | None = None,
    spent_cny: Callable[[], float] | None = None,
    progress: Callable[[int, int, JudgeRecord], None] | None = None,
) -> JudgeFile:
    """Judge every question of `specs` (see the module doc). `spent_cny` reads the cost ledger's
    total (None: costs are not tracked and there is no spend limit)."""
    return asyncio.run(
        _collect(
            specs,
            retriever=retriever,
            cfg=cfg,
            settings=settings,
            judge_model=judge_model,
            helper_model=helper_model,
            rescue=rescue,
            max_cny=max_cny,
            spent_cny=spent_cny,
            progress=progress,
        )
    )
