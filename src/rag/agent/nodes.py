"""The nodes of the ask_docs graph (design s13-3, s4-2; spec P3-01).

    route     rules: retrieve or not, which KB; clears the retrieval fields of the thread
    clarify   hook (P3-02): a question to ask the caller instead of searching
    rewrite   hook (P3-02, P3-03): standalone question, sub-questions, another wording
    retrieve  the retrieval layer, once per sub-question (a blocking call, run in a thread)
    judge     hook (P3-03): enough found? plus the guards that end the loop
    answer    the answer layer (`rag.answer`) over the chunks of every round
    verify    builds the `AskResult`: retrieval flags, answer flags, stop reason

Every node runs in its own step span and writes its time into `timings_ms`. The three hooks are
stubs in P3-01 (`rag.agent.hooks`); the guards around them live here, so a hook that calls a
model only has to say "insufficient" and the loop still ends at `max_rounds`, at the deadline
(no round starts when less than the answer's time is left) and at the token cap.
"""

import asyncio
import inspect
import time
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from rag.agent.budget import RequestTokenCapExceeded
from rag.agent.context import AgentContext
from rag.agent.result import compose_result, group_by_sub_question, smalltalk_answer
from rag.agent.state import AgentState, RetrievalRound
from rag.answer import build_context, generate_answer
from rag.observe import attributes as attr
from rag.observe import layer_span
from rag.retrieve.observe import observe_retrieval

TOOL = "ask_docs"


def _ms(began: float) -> float:
    return round((time.perf_counter() - began) * 1000.0, 3)


def _sub_questions(state: AgentState) -> list[str]:
    planned = [q for q in state.get("sub_questions") or [] if q.strip()]
    return planned or [state.get("standalone_question") or state["question"]]


async def route_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Rules only, no model. Overwrites everything a former run on this thread left behind."""
    ctx = runtime.context
    began = time.perf_counter()
    with layer_span("route") as span:
        decision = ctx.hooks.decide_route(
            state["question"], kb=state["kb"], history=state.get("history") or []
        )
        if inspect.isawaitable(decision):  # P3-02: the rules may fall back to a small model
            decision = await decision
        span.set_attribute(attr.ROUTE_RETRIEVE, decision.retrieve)
        span.set_attribute(attr.ROUTE_REASON, decision.reason)
    return {
        "rounds": Overwrite([]),
        "timings_ms": Overwrite({"route": _ms(began)}),
        "round": 0,
        "loop": False,
        "sub_questions": [],
        "standalone_question": None,
        "clarification": None,
        "stop_reason": None,
        "answer": None,
        "result": None,
        "kb": decision.kb or state["kb"],
        "needs_retrieval": decision.retrieve,
        "needs_clarification": decision.needs_clarification,
    }


async def clarify_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    began = time.perf_counter()
    with layer_span("clarify"):
        question = await ctx.hooks.clarify(state, ctx)
    return {"clarification": question or None, "timings_ms": {"clarify": _ms(began)}}


async def rewrite_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    began = time.perf_counter()
    update: dict[str, Any] = {}
    with layer_span("rewrite") as span:
        try:
            outcome = await ctx.hooks.rewrite(state, ctx)
        except RequestTokenCapExceeded:  # no tokens left for the rewrite: answer from what we have
            span.set_attribute(attr.AGENT_STOP_REASON, "token_cap")
            update["stop_reason"] = "token_cap"
        else:
            if outcome.standalone_question:
                update["standalone_question"] = outcome.standalone_question
            if outcome.sub_questions:
                update["sub_questions"] = list(outcome.sub_questions)
    update["timings_ms"] = {"rewrite": _ms(began)}
    return update


async def retrieve_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """One search per sub-question (the retrieval layer is blocking: a worker thread; the
    request context and span follow it). A failure is the request's failure (P3-03 decides what
    a later round may survive)."""
    ctx = runtime.context
    began = time.perf_counter()
    round_no = state.get("round", 0) + 1
    retrieval = ctx.cfg.retrieval
    top_k = retrieval.final_k
    found: list[RetrievalRound] = []
    for sub_question in _sub_questions(state):
        with layer_span(
            "retrieve",
            attributes={
                attr.TOP_K: top_k,
                attr.RERANK_N: retrieval.rerank_n,
                attr.FINAL_K: retrieval.final_k,
                attr.THRESHOLD: retrieval.threshold,
                attr.AGENT_ROUND: round_no,
            },
        ) as span:
            result = await asyncio.to_thread(
                ctx.retriever.retrieve,
                sub_question,
                ctx.cfg.kb,
                top_k,
                filters=dict(ctx.filters or {}),
                deadline_ms=ctx.retrieval_deadline_ms(),
                request_id=ctx.request_id,
                acl_groups=ctx.acl_groups,
            )
            span.set_attribute(attr.HITS, len(result.chunks))
        observe_retrieval(result, ctx.cfg.kb, TOOL)
        found.append(RetrievalRound.from_result(round_no, sub_question, result))
    return {"rounds": found, "round": round_no, "timings_ms": {"retrieve": _ms(began)}}


def plan_next(state: AgentState, verdict: str, ctx: AgentContext) -> tuple[bool, str | None]:
    """Whether another round runs, and why not when the judge wanted one: the KB's `max_rounds`,
    the deadline (less than the answer's time is left), the token cap (no tokens left for
    another rewrite and judge after keeping the answer's)."""
    if verdict == "sufficient":
        return False, None
    if state.get("round", 0) >= ctx.cfg.agent.max_rounds:
        return False, "max_rounds"
    if ctx.deadline.remaining_ms() <= ctx.reserve_ms:
        return False, "deadline"
    if ctx.budget.exhausted(reserve=ctx.settings.agent_answer_reserve_tokens):
        return False, "token_cap"
    return True, None


async def judge_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    began = time.perf_counter()
    forced: str | None = None
    with layer_span("judge") as span:
        try:
            verdict = await ctx.hooks.judge(state, ctx)
        except RequestTokenCapExceeded:  # could not even ask: treat as "enough" and say so
            verdict, forced = "sufficient", "token_cap"
        loop, stop_reason = plan_next(state, verdict, ctx)
        stop_reason = stop_reason or forced
        span.set_attribute(attr.AGENT_VERDICT, verdict)
        if stop_reason:
            span.set_attribute(attr.AGENT_STOP_REASON, stop_reason)
    return {"loop": loop, "stop_reason": stop_reason, "timings_ms": {"judge": _ms(began)}}


async def answer_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The answer layer over the chunks of every round (deduplicated by chunk id, budget split
    per sub-question: s10-3). Nothing found: the fixed not-found answer, no model call. A greeting
    gets a fixed reply."""
    ctx = runtime.context
    began = time.perf_counter()
    cfg = ctx.cfg
    with layer_span("answer", attributes={attr.AGENT_ROUND: state.get("round", 0)}) as span:
        if not state.get("needs_retrieval", True):
            answer = smalltalk_answer(state["question"], cfg.answer.language)
        else:
            context = build_context(
                group_by_sub_question(state.get("rounds") or []),
                cfg.answer.context_tokens,
                metadata_keys=cfg.metadata.llm,
            )
            answer = await generate_answer(
                state.get("standalone_question") or state["question"],
                context,
                kb=cfg.kb,
                config=cfg.answer,
                model=ctx.answer_model() if context.fragments else None,
                request_id=ctx.request_id,
                settings=ctx.settings,
            )
        span.set_attribute(attr.AGENT_TOKENS, ctx.budget.spent)
    return {"answer": answer, "timings_ms": {"answer": _ms(began)}}


async def verify_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Citation check level 1 ran inside the answer layer; this node records and passes on what
    the run found out: retrieval flags, answer flags, stop reason, tokens (s11-2, s16-8)."""
    ctx = runtime.context
    began = time.perf_counter()
    with layer_span("verify") as span:
        timings = {**(state.get("timings_ms") or {}), "verify": _ms(began)}
        result = compose_result(state, ctx, timings_ms=timings)
        span.set_attributes(
            {
                attr.ANSWER_FOUND: result.found,
                attr.ANSWER_UNSUPPORTED: result.citation_check.unsupported_sentences,
                attr.ANSWER_REMOVED_MARKERS: len(result.citation_check.removed_markers),
                attr.AGENT_ROUND: result.rounds,
                attr.AGENT_TOKENS: result.tokens_used,
            }
        )
        if result.stop_reason:
            span.set_attribute(attr.AGENT_STOP_REASON, result.stop_reason)
    return {"result": result, "timings_ms": {"verify": timings["verify"]}}
