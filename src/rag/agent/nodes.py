"""The nodes of the ask_docs graph (design s13-3, s4-2; spec P3-01, P3-02).

    route     rules (and the small model for what rules cannot decide): retrieve or not, which
              KB, ask the caller back, two-hop; clears the retrieval fields of the thread
    clarify   the question to ask the caller (hook), then `interrupt`: the run stops here and
              continues from here with the caller's answer
    rewrite   hook: standalone question, declarative query, sub-questions (P3-02); another
              wording of what was missing (P3-03)
    retrieve  the retrieval layer, once per sub-question (a blocking call, run in a thread)
    judge     hook (P3-03): enough found? plus the guards that end the loop
    answer    the answer layer (`rag.answer`) over the chunks of every round
    verify    builds the `AskResult`: retrieval flags, answer flags, stop reason

Every node runs in its own step span and writes its time into `timings_ms`. What each hook does
is in `rag.agent.hooks`; the guards around them live here, so a hook that calls a model only has
to say "insufficient" and the loop still ends at `max_rounds`, at the deadline (no round starts
when less than the answer's time is left) and at the token cap.
"""

import asyncio
import inspect
import time
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Overwrite, interrupt

from rag.agent.budget import RequestTokenCapExceeded
from rag.agent.context import AgentContext
from rag.agent.result import (
    answer_question,
    compose_result,
    group_by_sub_question,
    smalltalk_answer,
)
from rag.agent.routing import CLARIFY, TWO_HOP
from rag.agent.rules import default_clarification
from rag.agent.state import AgentState, RetrievalRound, working_question
from rag.answer import build_context, generate_answer
from rag.answer.prompts import answer_language
from rag.observe import attributes as attr
from rag.observe import layer_span
from rag.retrieve.observe import observe_retrieval

TOOL = "ask_docs"


def _ms(began: float) -> float:
    return round((time.perf_counter() - began) * 1000.0, 3)


def _sub_questions(state: AgentState) -> list[str]:
    planned = [q for q in state.get("sub_questions") or [] if q.strip()]
    return planned or [state.get("standalone_question") or working_question(state)]


async def route_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Rules first, no model. Only for what the rules leave `undecided` or `maybe`, and only
    when the knowledge base's `agent` settings allow it, the small model is asked
    (`hooks.route_fallback`); a failure there keeps the rules' view and is noted in `fallbacks`.
    Overwrites everything a former run on this thread left behind."""
    ctx = runtime.context
    began = time.perf_counter()
    agent = ctx.cfg.agent
    with layer_span("route") as span:
        decision = ctx.hooks.decide_route(
            state["question"], kb=state["kb"], history=state.get("history") or []
        )
        if inspect.isawaitable(decision):  # the rules may be async
            decision = await decision
        splitting = agent.rewrite and agent.max_sub_questions > 1
        if decision.retrieve and agent.route_model:
            wants = set()
            if agent.clarify and decision.clarity == "undecided":
                wants.add(CLARIFY)
            if splitting and decision.multi_hop == "maybe":
                wants.add(TWO_HOP)
            if wants:
                decision = await ctx.hooks.route_fallback(state, ctx, decision, frozenset(wants))
        fallbacks = [f"route:{decision.fallback}"] if decision.fallback else []
        clarify = decision.retrieve and (
            decision.needs_clarification or (agent.clarify and decision.clarity == "vague")
        )
        two_hop = decision.retrieve and (
            decision.two_hop or (splitting and decision.multi_hop == "yes")
        )
        ask = None
        if clarify:
            ask = decision.clarify_question or default_clarification(
                answer_language(state["question"], ctx.cfg.answer.language)
            )
        span.set_attributes(
            {
                attr.ROUTE_RETRIEVE: decision.retrieve,
                attr.ROUTE_REASON: decision.reason,
                attr.ROUTE_CLARITY: decision.clarity,
                attr.ROUTE_MULTI_HOP: decision.multi_hop,
                attr.ROUTE_TWO_HOP: two_hop,
                attr.ROUTE_CLARIFY: clarify,
                attr.AGENT_TOKENS: ctx.budget.spent,
            }
        )
        if decision.fallback:
            span.set_attribute(attr.ROUTE_FALLBACK, decision.fallback)
    return {
        "rounds": Overwrite([]),
        "fallbacks": Overwrite(fallbacks),
        "timings_ms": Overwrite({"route": _ms(began)}),
        "round": 0,
        "loop": False,
        "sub_questions": [],
        "standalone_question": None,
        "clarification": None,
        "clarify_question": ask,
        "clarification_answer": None,
        "clarified": False,
        "clarified_question": None,
        "stop_reason": None,
        "answer": None,
        "result": None,
        "two_hop": two_hop,
        "tokens_before_clarify": ctx.budget.spent,
        "kb": decision.kb or state["kb"],
        "needs_retrieval": decision.retrieve,
        "needs_clarification": clarify,
    }


async def clarify_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Asks the caller back, once per request. The hook only names the question (the node runs
    again when the caller answers, so the hook must be cheap and make no model call); the node
    stops the run with `interrupt`. The first time it gets here the run ends with the
    question; when the caller answers, the run continues from this node: `interrupt` returns
    the answer, and the question and answer become the question the rest of the run works on
    (the reply is cut to half of `search_max_query_chars`, so the two together stay near the
    limit of a question)."""
    ctx = runtime.context
    began = time.perf_counter()
    with layer_span("clarify", attributes={attr.CLARIFY_RESUMED: ctx.resumed}) as span:
        question = await ctx.hooks.clarify(state, ctx)
        if question:
            span.set_attribute(attr.CLARIFY_QUESTION, question[:500])
    if not question:
        return {
            "needs_clarification": False,
            "clarification": None,
            "timings_ms": {"clarify": _ms(began)},
        }
    # Outside the span: the first time, `interrupt` raises and the run stops here (the stop is
    # not a failure of this step).
    answer = str(interrupt(question)).strip()[: ctx.settings.search_max_query_chars // 2]
    return {
        "needs_clarification": False,
        "clarification": None,
        "clarified": True,
        "clarification_answer": answer,
        "clarified_question": f"{state['question'].strip()} ({answer})" if answer else None,
        "timings_ms": {"clarify": _ms(began)},
    }


async def rewrite_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The rewrite hook (standalone question, declarative query, sub-questions). The hook falls
    back to the original question itself when its model fails; the node records that in
    `fallbacks` (the request is then `degraded`) and on the span."""
    ctx = runtime.context
    began = time.perf_counter()
    update: dict[str, Any] = {}
    with layer_span("rewrite", attributes={attr.AGENT_ROUND: state.get("round", 0) + 1}) as span:
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
            if outcome.fallback:
                update["fallbacks"] = [f"rewrite:{outcome.fallback}"]
                span.set_attribute(attr.REWRITE_FALLBACK, outcome.fallback)
            span.set_attributes(
                {
                    attr.REWRITE_MODEL_USED: outcome.used_model,
                    attr.REWRITE_QUERIES: [q[:300] for q in outcome.sub_questions],
                    attr.REWRITE_STANDALONE: (outcome.standalone_question or "")[:500],
                    attr.AGENT_TOKENS: ctx.budget.spent,
                }
            )
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
                answer_question(
                    working_question(state), state.get("standalone_question"), cfg.answer.language
                ),
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
