"""The nodes of the ask_docs graph (design s13-3, s4-2; spec P3-01, P3-02).

    route     rules (and the small model for what rules cannot decide): retrieve or not, which
              KB, ask the caller back, two-hop; clears the retrieval fields of the thread
    clarify   the question to ask the caller (hook), then `interrupt`: the run stops here and
              continues from here with the caller's answer
    rewrite   hook: standalone question, declarative query, sub-questions (P3-02); from round
              two on, other phrasings of what was missing: multi-query, HyDE (P3-03)
    retrieve  the retrieval layer, once per search of the round (a blocking call, run in a thread)
    judge     hook (P3-03): enough, partly enough or not enough found, and which sub-questions
              are missing; plus the guards that end the loop
    answer    the answer layer (`rag.answer`) over the chunks of every round
    verify    builds the `AskResult`: retrieval flags, answer flags, stop reason

Every node runs in its own step span and writes its time into `timings_ms`. What each hook does
is in `rag.agent.hooks`; the guards around them live here, so a hook that calls a model only has
to say "insufficient" and the loop still ends at `max_rounds`, at the deadline (no round starts
when less than the answer's time plus one round's is left) and at the token cap.
"""

import asyncio
import contextlib
import inspect
import time
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Overwrite, interrupt

from rag.agent.budget import RequestTokenCapExceeded
from rag.agent.context import AgentContext
from rag.agent.decisions import JudgeOutcome
from rag.agent.events import progress, retrieve_detail, streamed, write
from rag.agent.result import (
    answer_question,
    compose_result,
    group_by_sub_question,
    smalltalk_answer,
)
from rag.agent.routing import CLARIFY, TWO_HOP
from rag.agent.rules import default_clarification
from rag.agent.state import AgentState, RetrievalRound, SearchQuery, working_question
from rag.answer import build_context
from rag.answer.generate import AnswerDelta, AnswerDone, AnswerRetry, stream_answer
from rag.answer.prompts import answer_language
from rag.observe import attributes as attr
from rag.observe import layer_span
from rag.retrieve.hybrid import RetrievalError
from rag.retrieve.observe import observe_retrieval

TOOL = "ask_docs"


def _ms(began: float) -> float:
    return round((time.perf_counter() - began) * 1000.0, 3)


def _sub_questions(state: AgentState) -> list[str]:
    planned = [q for q in state.get("sub_questions") or [] if q.strip()]
    return planned or [state.get("standalone_question") or working_question(state)]


def _search_plan(state: AgentState) -> list[SearchQuery]:
    """The searches of this round: what the rewrite hook planned for it (round two on), else
    one search per sub-question."""
    planned = list(state.get("search_plan") or [])
    return planned or [SearchQuery(text=q, owner=q) for q in _sub_questions(state)]


def _round_robin(plan: list[SearchQuery]) -> list[SearchQuery]:
    """The searches of a later round, one phrasing of every sub-question before a second one of
    any: when time allows only some of them, each sub-question still gets a search."""
    by_owner: dict[str, list[SearchQuery]] = {}
    for query in plan:
        by_owner.setdefault(query.owner, []).append(query)
    longest = max((len(queries) for queries in by_owner.values()), default=0)
    return [qs[i] for i in range(longest) for qs in by_owner.values() if i < len(qs)]


# What a correction that could not go on says about why the run stops: a model that was refused
# for tokens or time is those limits, anything else is "no new query".
_STOP_FOR = {"token_cap": "token_cap", "no_time": "deadline"}


def _route_detail(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieve": update["needs_retrieval"],
        "clarify": update["needs_clarification"],
    }


def _rewrite_detail(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    return {"queries": len(_search_plan({**state, **update}))}


def _judge_detail(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {"loop": update["loop"]}
    if update["sufficiency"] is not None:
        fields["verdict"] = update["sufficiency"]
    return fields


def _answer_detail(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    answer = update["answer"]
    return {"found": answer.found, "status": str(answer.status)}


def _next_round(state: AgentState) -> int:
    return state.get("round", 0) + 1


def _this_round(state: AgentState) -> int:
    return state.get("round", 0)


@streamed("route", detail=_route_detail)
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
        "search_plan": [],
        "missing": [],
        "sufficiency": None,
        "standalone_question": None,
        "clarification": None,
        "clarify_question": ask,
        "clarification_answer": None,
        "clarified": False,
        "clarified_question": None,
        "stop_reason": None,
        "round_started": None,
        "retrieval_error": None,
        "answer": None,
        "result": None,
        "two_hop": two_hop,
        "multi_hop": decision.multi_hop,
        "tokens_before_clarify": ctx.budget.spent,
        "kb": decision.kb or state["kb"],
        "needs_retrieval": decision.retrieve,
        "needs_clarification": clarify,
    }


@streamed("clarify")
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


@streamed("rewrite", round_of=_next_round, detail=_rewrite_detail)
async def rewrite_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The rewrite hook (standalone question, declarative query, sub-questions; from round two on
    the new phrasings of what is missing). The hook falls back to the original question itself
    when its model fails in round one; the node records that in `fallbacks` (the request is
    then `degraded`) and on the span. A correction that cannot make a new query ends the run
    (`stop_reason`): searching the same thing again would give the same chunks."""
    ctx = runtime.context
    began = time.perf_counter()
    update: dict[str, Any] = {"round_started": ctx.deadline.clock()}
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
            if outcome.queries:
                update["search_plan"] = list(outcome.queries)
            if outcome.fallback:
                update["fallbacks"] = [f"{outcome.technique or 'rewrite'}:{outcome.fallback}"]
                span.set_attribute(attr.REWRITE_FALLBACK, outcome.fallback)
            if outcome.exhausted:
                update["stop_reason"] = _STOP_FOR.get(outcome.fallback or "", "no_new_query")
                span.set_attribute(attr.AGENT_STOP_REASON, update["stop_reason"])
            if outcome.technique:
                span.set_attribute(attr.REWRITE_TECHNIQUE, outcome.technique)
            # A HyDE passage is generated text: the span says what kind and how long, not what.
            texts = [
                f"hyde passage, {len(q.text)} chars" if q.kind == "hyde" else q.text
                for q in outcome.queries
            ] or list(outcome.sub_questions)
            span.set_attributes(
                {
                    attr.REWRITE_MODEL_USED: outcome.used_model,
                    attr.REWRITE_QUERIES: [q[:300] for q in texts],
                    attr.REWRITE_STANDALONE: (outcome.standalone_question or "")[:500],
                    attr.AGENT_TOKENS: ctx.budget.spent,
                }
            )
    update["timings_ms"] = {"rewrite": _ms(began)}
    return update


@streamed("retrieve", round_of=_next_round, detail=retrieve_detail)
async def retrieve_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """One search per planned query (the retrieval layer is blocking: a worker thread; the
    request context and span follow it): round one searches each sub-question, a later round
    the phrasings of what was missing (one phrasing of each sub-question first).

    A search of a later round is made only when it has the least time worth spending
    (`agent_search_min_ms` before the answer's time, spec P3-06); the ones that do not fit are
    dropped, and when none fits the run stops with `deadline` and goes on to the answer, the
    round not counted. A search of a later round that fails (`RetrievalError`) is not the
    request's failure: the chunks of the earlier rounds are still there, so the failure is
    noted (`fallbacks`, `retrieval_error`), the run stops with `retrieval_error` and answers
    from them.
    A failure in round one is the request's failure: there is nothing to answer from."""
    ctx = runtime.context
    began = time.perf_counter()
    round_no = state.get("round", 0) + 1
    retrieval = ctx.cfg.retrieval
    top_k = retrieval.final_k
    plan = _search_plan(state)
    if round_no > 1:
        plan = _round_robin(plan)
    found: list[RetrievalRound] = []
    failure: RetrievalError | None = None
    for query in plan:
        if round_no > 1 and not ctx.can_search():
            break
        try:
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
                # Round one is the plain search (as `search_docs`); later rounds leave time for
                # the rerank while waiting for the query vector.
                extra = (
                    {"rerank_reserve_ms": ctx.settings.agent_rerank_reserve_ms}
                    if round_no > 1
                    else {}
                )
                result = await asyncio.to_thread(
                    ctx.retriever.retrieve,
                    query.text,
                    ctx.cfg.kb,
                    top_k,
                    filters=dict(ctx.filters or {}),
                    deadline_ms=ctx.retrieval_deadline_ms(),
                    request_id=ctx.request_id,
                    acl_groups=ctx.acl_groups,
                    **extra,
                )
                span.set_attribute(attr.HITS, len(result.chunks))
        except RetrievalError as exc:
            if round_no == 1:
                raise
            failure = exc
            break
        observe_retrieval(result, ctx.cfg.kb, TOOL)
        found.append(
            RetrievalRound.from_result(
                round_no,
                query.owner,
                result,
                query=query.text,
                kind=query.kind,
                scored_only=round_no > 1,
            )
        )
    update: dict[str, Any] = {"rounds": found, "timings_ms": {"retrieve": _ms(began)}}
    if found or round_no == 1:
        update["round"] = round_no
    if failure is not None:
        category = str(failure.category)
        update["fallbacks"] = [f"retrieve:{category}"]
        update["retrieval_error"] = category
    if failure is not None:
        update["stop_reason"] = "retrieval_error"
    elif not found:
        update["stop_reason"] = "deadline"
    return update


def as_outcome(judgement: JudgeOutcome | str, state: AgentState) -> JudgeOutcome:
    """The judge hook's answer as a `JudgeOutcome`. A P3-01 hook says `sufficient` or
    `insufficient`: the second means chunks exist but something is still wanted, for every
    sub-question searched so far."""
    if isinstance(judgement, JudgeOutcome):
        return judgement
    if judgement == "sufficient":
        return JudgeOutcome("enough")
    owners = tuple(dict.fromkeys(r.sub_question for r in state.get("rounds") or []))
    return JudgeOutcome("partly_enough", owners)


def plan_next(
    state: AgentState,
    judgement: JudgeOutcome | str,
    ctx: AgentContext,
    *,
    round_ms: float | None = None,
) -> tuple[bool, str | None]:
    """Whether another round runs, and why not when the judge wanted one.

    The limits, in this order: the KB's `max_rounds` (three rounds in which nothing passed the
    rerank threshold are a definitive "not found", no stop reason; with part of the question
    still without grounds it is `max_rounds`); a judged KB that allows no model before the
    answer, which cannot make a new phrasing (`no_new_query`); the deadline (less than the
    answer's time plus one round's is left: a round that cannot finish is not started; one
    round's time is at least `agent_round_min_ms` and, once a round has run (`round_ms`, how long
    it took), `agent_round_ratio` (0.9) times that, spec P3-06); the token cap (no tokens left
    for another rewrite and judgement after keeping the answer's)."""
    outcome = as_outcome(judgement, state)
    if outcome.verdict == "enough":
        return False, None
    agent = ctx.cfg.agent
    if outcome.verdict == "not_enough" and not agent.round2_on_not_enough:
        return False, None  # a plain miss is "not found"; only `partly_enough` goes on
    if state.get("round", 0) >= agent.max_rounds:
        return False, None if outcome.verdict == "not_enough" else "max_rounds"
    if agent.judge and not agent.rewrite:
        return False, "no_new_query"
    needed = float(ctx.settings.agent_round_min_ms)
    if round_ms:
        needed = max(needed, ctx.settings.agent_round_ratio * round_ms)
    if ctx.deadline.remaining_ms() <= ctx.reserve_ms + needed:
        return False, "deadline"
    if ctx.budget.exhausted(reserve=ctx.settings.agent_answer_reserve_tokens):
        return False, "token_cap"
    return True, None


@streamed("judge", round_of=_this_round, detail=_judge_detail)
async def judge_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The judge hook (is what was found enough, and which sub-questions are missing), then the
    guards (`plan_next`) that decide whether another round runs. A judgement that fell back to
    the rules is recorded in `fallbacks` (the request is then `degraded`)."""
    ctx = runtime.context
    began = time.perf_counter()
    forced: str | None = None
    with layer_span("judge") as span:
        try:
            outcome = as_outcome(await ctx.hooks.judge(state, ctx), state)
        except RequestTokenCapExceeded:  # could not even ask: treat as "enough" and say so
            outcome, forced = JudgeOutcome("enough"), "token_cap"
        started = state.get("round_started")
        round_ms = (ctx.deadline.clock() - started) * 1000.0 if started is not None else None
        loop, stop_reason = plan_next(state, outcome, ctx, round_ms=round_ms)
        stop_reason = stop_reason or forced
        span.set_attributes(
            {
                attr.AGENT_VERDICT: outcome.verdict,
                attr.JUDGE_MISSING: [m[:300] for m in outcome.missing],
                attr.JUDGE_MODEL_USED: outcome.used_model,
                attr.JUDGE_SKIPPED: outcome.skipped,
            }
        )
        if outcome.fallback:
            span.set_attribute(attr.JUDGE_FALLBACK, outcome.fallback)
        if stop_reason:
            span.set_attribute(attr.AGENT_STOP_REASON, stop_reason)
    update: dict[str, Any] = {
        "loop": loop,
        "stop_reason": stop_reason,
        "missing": list(outcome.missing),
        # Without `agent.judge` nothing was judged: the stub's "enough" is not a verdict.
        "sufficiency": outcome.verdict if ctx.cfg.agent.judge else None,
        "timings_ms": {"judge": _ms(began)},
    }
    if outcome.fallback:
        update["fallbacks"] = [f"judge:{outcome.fallback}"]
    return update


@streamed("answer", round_of=_this_round, detail=_answer_detail)
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
            answer = None
            async with contextlib.aclosing(
                stream_answer(
                    answer_question(
                        working_question(state),
                        state.get("standalone_question"),
                        cfg.answer.language,
                    ),
                    context,
                    kb=cfg.kb,
                    config=cfg.answer,
                    model=ctx.answer_model() if context.fragments else None,
                    request_id=ctx.request_id,
                    settings=ctx.settings,
                )
            ) as events:
                async for event in events:
                    # The text goes to the stream as the model writes it; the answer node is the
                    # only one that writes `token` events (`rag.agent.events`).
                    if isinstance(event, AnswerDelta):
                        write(runtime, "token", {"text": event.text, "attempt": event.attempt})
                    elif isinstance(event, AnswerRetry):
                        progress(runtime, "answer", "retry", attempt=event.attempt)
                    elif isinstance(event, AnswerDone):
                        answer = event.answer
            assert answer is not None  # `stream_answer` always ends with `AnswerDone`
        span.set_attribute(attr.AGENT_TOKENS, ctx.budget.spent)
    return {"answer": answer, "timings_ms": {"answer": _ms(began)}}


@streamed("verify")
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
