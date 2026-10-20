"""Running the ask_docs graph for one request (design s13-3, s16-7, s16-8; spec P3-01).

    runtime = AgentRuntime.from_settings(settings)    # one per process: it owns the session locks
    result = await ask_docs("How do I push a model?", cfg=cfg, retriever=pipeline,
                            settings=settings, runtime=runtime, session_id="s1")

One call:

1. takes the lock of the session (the caller's, in that knowledge base: `session_key`;
   requests of one session run one after another; the wait counts
   against the deadline: `AgentDeadlineExceeded` when it does not end in time) and loads the
   session's last turns;
2. runs the graph on the checkpoint thread `session_id:run_id` with `recursion_limit` set,
   inside the total deadline: when the deadline passes the run is cancelled and the result is
   built from the last checkpoint (the chunks found so far, no generated text), flagged "not
   confirmed that everything was searched"; the soft part of the deadline (no round starts when
   less than the answer's time is left) is inside the graph (`plan_next`);
3. stores the turn (question and answer text) in the session's history.

A run can stop to ask the caller (`status: need_clarification`, the clarify step's `interrupt`).
Then the call ends and the session's lock is released: the waiting run is only a checkpoint
thread, it blocks nobody. The result carries an opaque `resume_token` (`rag.agent.resume`);
`ask_docs(..., resume_token=..., answer=...)` continues that thread with the caller's answer.
The continuing call is a new call: it takes the session's lock again (and waits for it within
its own deadline), takes the token (once, only for the caller, knowledge base and session it
was issued for, only while the thread is waiting), and writes the history when it completes.
Nothing is written to the history while the question is open; the finished run writes the
question, the question asked back, the caller's answer and the answer as two turns, so the
conversation reads as it happened. The request's token cap counts both calls.

The checkpoint thread is `session_id:run_id`, with a `run_id` the server makes for each call:
the caller's request id (an `X-Request-Id` header, which anyone can repeat) is only for
correlation, so two runs never write to, or get rebuilt from, one thread.

A call that carried history is flagged `cache_allowed=False` (the exact cache of P5 neither
reads nor writes it, s12-1).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from rag.agent.budget import TokenBudget
from rag.agent.context import AgentContext, AgentRetriever
from rag.agent.events import StreamEmit, safe_emit
from rag.agent.graph import build_graph
from rag.agent.hooks import AgentHooks
from rag.agent.result import compose_result
from rag.agent.resume import MemoryResumeStore, ResumeStore, ResumeTokenInvalid
from rag.agent.session import HistoryTurn, SessionLocks, SessionStore
from rag.agent.state import MSGPACK_ALLOWED, thread_id
from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.factories import (
    CheckpointerSource,
    make_checkpointer_source,
    make_resume_store,
    make_session_locks,
    make_session_store,
)
from rag.observe import new_request_id
from rag.retrieve.deadline import Deadline
from rag.types import AnswerStatus, AskResult, ErrorCategory, StopReason

logger = logging.getLogger(__name__)

INTERRUPT = "__interrupt__"  # the key under which LangGraph reports a run stopped by `interrupt`


def _first(*candidates: Any) -> Any:
    """The first candidate that is not None."""
    return next(c for c in candidates if c is not None)


class AgentDeadlineExceeded(Exception):  # noqa: N818 - named like BudgetExceeded
    """The request ran out of its total time before the graph could start (it waited for the
    session's lock). Its category is `agent_deadline` (read by `rag.serve.errors`)."""

    error_category = ErrorCategory.AGENT_DEADLINE


class AgentRuntime:
    """What the ask_docs graph needs across requests: the session history, the session locks,
    the checkpoint saver and the hooks. Create one per process and share it."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        store: SessionStore,
        locks: SessionLocks,
        checkpointers: CheckpointerSource,
        resumes: ResumeStore | None = None,
        hooks: AgentHooks | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.store = store
        self.locks = locks
        self.checkpointers = checkpointers
        self.resumes = resumes or MemoryResumeStore(settings.agent_resume_ttl_s)
        self.hooks = hooks or AgentHooks()
        self.clock = clock
        self._chat: Any = None
        self._helper: Any = None
        self._judge: Any = None

    @classmethod
    def from_settings(
        cls,
        settings: ServiceSettings,
        *,
        hooks: AgentHooks | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> AgentRuntime:
        """Stores and locks from the connection strings in `settings` (design s16-7)."""
        serde = JsonPlusSerializer(allowed_msgpack_modules=list(MSGPACK_ALLOWED))
        store = make_session_store(
            settings.session_store_url, keep_turns=settings.agent_history_turns
        )
        checkpointers = make_checkpointer_source(settings.checkpoint_url, serde=serde)
        # Create both databases now, once, before any request: concurrent first requests that
        # each created them failed with "database is locked".
        store.prepare()
        checkpointers.prepare()
        resumes = make_resume_store(
            settings.session_store_url,
            ttl_s=settings.agent_resume_ttl_s,
            lease_s=2.0 * settings.search_max_deadline_ms / 1000.0,
        )
        resumes.prepare()
        return cls(
            settings,
            store=store,
            locks=make_session_locks(settings.session_lock_url),
            checkpointers=checkpointers,
            resumes=resumes,
            hooks=hooks,
            clock=clock,
        )

    @staticmethod
    def session_key(caller_id: str, kb: str, session_id: str) -> str:
        """The key of a session's history and lock: the caller, the knowledge base and the
        session id the caller chose. Two callers (or one caller in two knowledge bases) that
        pick the same session id get separate histories and do not wait for each other. A hash,
        so no part can run into another and the key is safe in a file name."""
        raw = json.dumps([caller_id, kb, session_id], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    def run_config(self, thread: str) -> dict[str, Any]:
        """The LangGraph run config: this request's thread and an explicit recursion limit."""
        return {
            "configurable": {"thread_id": thread},
            "recursion_limit": self.settings.agent_recursion_limit,
        }

    def default_model(self) -> Any:
        """The `online` chat model, built once (the answer layer would build one per call), with
        its output capped at `agent_answer_max_tokens` (spec P3-06): the profile's 4096 let one
        answer run to 4096 tokens and past the deadline."""
        if self._chat is None:
            from rag.llm import make_chat_model

            self._chat = make_chat_model(
                "online", self.settings, max_tokens=self.settings.agent_answer_max_tokens
            )
        return self._chat

    def default_helper_model(self) -> Any:
        """The small model of the route and rewrite steps (`agent_rewrite_profile`), built once,
        when a step first needs it: a gateway that is not configured fails there, as that
        step's fallback, not at start-up."""
        if self._helper is None:
            from rag.llm import make_chat_model

            self._helper = make_chat_model(self.settings.agent_rewrite_profile, self.settings)
        return self._helper

    def default_judge_model(self) -> Any:
        """The model of the sufficiency judge (`agent_judge_profile`), built once, when the judge
        first needs it. The same model as the helper's when the two profiles are the same."""
        if self.settings.agent_judge_profile == self.settings.agent_rewrite_profile:
            return self.default_helper_model()
        if self._judge is None:
            from rag.llm import make_chat_model

            self._judge = make_chat_model(self.settings.agent_judge_profile, self.settings)
        return self._judge

    async def state_values(self, thread: str) -> dict[str, Any]:
        """The last checkpointed state of a thread (empty for an unknown one); for tests and
        for finding out what a request did."""
        async with self.checkpointers.open() as saver:
            snapshot = await build_graph(saver).aget_state(self.run_config(thread))
        return dict(snapshot.values)

    async def ask(
        self,
        question: str,
        *,
        cfg: KnowledgeBaseConfig,
        retriever: AgentRetriever,
        session_id: str | None = None,
        request_id: str | None = None,
        caller_id: str = "local",
        acl_groups: tuple[str, ...] = ("public",),
        deadline_ms: int | None = None,
        model: Any = None,
        helper_model: Any = None,
        judge_model: Any = None,
        filters: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        resume_token: str | None = None,
        answer: str | None = None,
        emit: StreamEmit | None = None,
    ) -> AskResult:
        """Run one request, or continue one that asked the caller back (see the module doc).

        `ValueError` for a session id that is not 1 to 64 letters, digits, dot, dash or
        underscore (and not `-`), and for only one of `resume_token` and `answer`. With a
        `resume_token` the run continues from the question it stopped at and `question` is not
        used; `ResumeTokenInvalid` when the token cannot be used by this caller, knowledge base
        and session (or at all). `run_id` names the checkpoint thread of a new run (default: a
        new one); `helper_model` is the small model of the route and rewrite steps, and
        `judge_model` the one of the sufficiency judge (default: `helper_model` when given,
        else the model of `agent_judge_profile`). `emit(name, data)` receives the run's
        `progress` and `token` events as they happen (`rag.agent.events`; the streaming
        endpoint's source); it is called on the event loop and must not block, and a call that
        raises is logged and the later events are dropped. The run does not depend on it."""
        if (resume_token is None) != (answer is None):
            raise ValueError("resume_token and answer go together")
        request_id = request_id or new_request_id()
        resuming = resume_token is not None
        emit = safe_emit(emit)
        settings = self.settings
        total_ms = deadline_ms if deadline_ms is not None else settings.agent_deadline_ms
        began = time.perf_counter()
        deadline = Deadline.after_ms(total_ms, clock=self.clock)
        ctx = AgentContext(
            settings=settings,
            cfg=cfg,
            retriever=retriever,
            deadline=deadline,
            total_ms=total_ms,
            budget=TokenBudget(settings.agent_token_cap, settings.agent_answer_reserve_tokens),
            model=model if model is not None else self.default_model,
            helper_model=helper_model if helper_model is not None else self.default_helper_model,
            judge_model=_first(judge_model, helper_model, self.default_judge_model),
            hooks=self.hooks,
            request_id=request_id,
            session_id=session_id,
            thread_id="" if resuming else thread_id(session_id, run_id or new_request_id()),
            acl_groups=tuple(acl_groups),
            filters=filters,
        )
        async with AsyncExitStack() as stack:
            history: list[HistoryTurn] = []
            key = ""
            if session_id is not None:
                key = self.session_key(caller_id, cfg.kb, session_id)
                await self._hold_session(stack, key, deadline, emit)
                if not resuming:
                    history = await self.store.load(key)
            claimed: tuple[str, str] | None = None  # the token and the id of our claim
            try:
                inputs: Mapping[str, Any] | Command | None
                if resume_token is not None:
                    history, inputs, claim_id = await self._take_token(
                        resume_token, answer or "", ctx, caller_id=caller_id, session_id=session_id
                    )
                    claimed = (resume_token, claim_id)
                else:
                    inputs = {
                        "request_id": request_id,
                        "session_id": session_id,
                        "kb": cfg.kb,
                        "question": question,
                        "history": history,
                    }
                ctx.used_history = bool(history)
                result, values = await self._run(inputs, ctx, emit)
                if claimed is not None:  # the run ended, whatever it ended with: used up
                    await self.resumes.finish(*claimed)
                    claimed = None
                result = result.model_copy(
                    update={
                        "timings_ms": {
                            **result.timings_ms,
                            "total": round((time.perf_counter() - began) * 1000.0, 3),
                        }
                    }
                )
                if result.status == AnswerStatus.NEED_CLARIFICATION:
                    await self._drop_abandoned_questions()
                    token = await self.resumes.create(
                        caller_id=caller_id,
                        kb=cfg.kb,
                        session_id=session_id,
                        thread_id=ctx.thread_id,
                    )
                    result = result.model_copy(update={"resume_token": token})
                elif session_id is not None and (result.answer or values.get("clarified")):
                    result = await self._remember(key, values, question, request_id, result)
            except BaseException:
                # The call raised or was cancelled before the run ended: the caller's reply is
                # not lost, the token can be used again and the thread goes on where it stopped.
                if claimed is not None:
                    with contextlib.suppress(Exception):
                        spent = ctx.budget.spent  # what this request has used so far
                        await asyncio.shield(self.resumes.release(*claimed, tokens=spent))
                raise
        return result

    async def _remember(
        self,
        key: str,
        values: Mapping[str, Any],
        question: str,
        request_id: str,
        result: AskResult,
    ) -> AskResult:
        """Write the turns of a finished run to the session's history. The answer is already
        made (and a resume token used up), so a store that fails must not take it away: the
        failure is logged and the result says so (`history:write_failed`, `degraded`, not for
        the exact cache)."""
        try:
            for turn in self._turns(values, question, request_id, result):
                await self.store.append(key, turn)
        except Exception:
            logger.warning("could not write the session history", exc_info=True)
            return result.model_copy(
                update={
                    "degraded": True,
                    "fallbacks": [*result.fallbacks, "history:write_failed"],
                    "cache_allowed": False,
                }
            )
        return result

    async def _drop_abandoned_questions(self) -> None:
        """Delete the checkpoint threads of questions nobody answered in time (their tokens have
        expired). Done when a new question is asked back; a failure is logged, never a failure of
        the request."""
        try:
            threads = await self.resumes.purge_expired()
            if threads:
                async with self.checkpointers.open() as saver:
                    for thread in threads:
                        await saver.adelete_thread(thread)
        except Exception:
            logger.warning("could not clean up abandoned clarification threads", exc_info=True)

    @staticmethod
    def _turns(
        values: Mapping[str, Any], question: str, request_id: str, result: AskResult
    ) -> list[HistoryTurn]:
        """What the session remembers of a finished run: question and answer; after a
        clarification, the question with the question asked back as its answer, then the
        caller's answer with the real answer (so a later follow-up reads the conversation as
        it happened). A resumed run that was cut off has no answer text: the exchange is still
        written, the reply with the notice that the run stopped."""
        asked = values.get("question") or question
        if values.get("clarified") and values.get("clarify_question"):
            return [
                HistoryTurn(
                    request_id=values.get("request_id") or request_id,
                    question=asked,
                    answer=values["clarify_question"],
                ),
                HistoryTurn(
                    request_id=request_id,
                    question=values.get("clarification_answer") or "",
                    answer=result.answer or result.notice or "",
                ),
            ]
        return [HistoryTurn(request_id=request_id, question=asked, answer=result.answer)]

    async def _take_token(
        self, token: str, answer: str, ctx: AgentContext, *, caller_id: str, session_id: str | None
    ) -> tuple[list[HistoryTurn], Command | None, str]:
        """Use the resume token for this call and point the call at its thread. Checks that
        the token is this caller's, for this knowledge base and session (`claim`), and that the
        thread belongs to this session and either waits for the answer (the input is
        `Command(resume=answer)`) or is a resumed run that failed after the question was
        answered (no input: the run goes on from the step that failed, with the first answer).
        Anything else is refused and the token is used up. Returns the history the run
        started with (kept in its checkpoint), the input for the graph and the id of our claim;
        carries over the tokens the request had spent (before the question was asked back and
        in failed attempts), so the request's cap counts them all."""
        record = await self.resumes.claim(
            token, caller_id=caller_id, kb=ctx.cfg.kb, session_id=session_id
        )
        ctx.thread_id = record.thread_id
        ctx.resumed = True
        try:
            async with self.checkpointers.open() as saver:
                snapshot = await build_graph(saver).aget_state(self.run_config(record.thread_id))
            values = dict(snapshot.values)
            waiting = bool(snapshot.next) and any(task.interrupts for task in snapshot.tasks)
            failed_after_answer = bool(snapshot.next) and bool(values.get("clarified"))
            problem = None
            if not (waiting or failed_after_answer):
                problem = "the run is not waiting for an answer"
            elif values.get("session_id") != session_id:
                problem = "the thread belongs to another session"
        except BaseException:  # could not even look: the claim goes back
            with contextlib.suppress(Exception):
                await asyncio.shield(self.resumes.release(token, record.claim_id))
            raise
        if problem is not None:
            # A token that cannot be used is used up: asking again says so.
            await self.resumes.finish(token, record.claim_id)
            raise ResumeTokenInvalid(problem)
        ctx.budget.add(max(int(values.get("tokens_before_clarify") or 0), record.tokens))
        inputs = Command(resume=answer) if waiting else None
        return list(values.get("history") or []), inputs, record.claim_id

    async def _hold_session(
        self, stack: AsyncExitStack, key: str, deadline: Deadline, emit: StreamEmit | None
    ) -> None:
        """Take the session's lock within the request's deadline; say so when it has to wait."""
        if emit is not None and self.locks.busy(key):
            emit("progress", {"node": "queued", "phase": "waiting"})
        waiting = asyncio.timeout(deadline.remaining_s())
        try:
            async with waiting:
                await stack.enter_async_context(self.locks.hold(key))
        except TimeoutError:
            if not waiting.expired():
                raise
            raise AgentDeadlineExceeded(
                "waited for the session's lock until the deadline"
            ) from None

    async def _run(
        self, inputs: Mapping[str, Any] | Command | None, ctx: AgentContext, emit: StreamEmit | None
    ) -> tuple[AskResult, Mapping[str, Any]]:
        """Run (or continue) the graph; the result and the state it was made from. The one
        implementation behind `ask` and the stream: the graph's `custom` events go to `emit`
        (when there is one), the rest of the loop is `ainvoke`'s own."""
        async with self.checkpointers.open() as saver:
            graph = build_graph(saver)
            config = self.run_config(ctx.thread_id)
            running = asyncio.timeout(ctx.deadline.remaining_s())
            try:
                async with running:
                    final = await self._stream_graph(graph, inputs, config, ctx, emit)
            except TimeoutError:
                if not running.expired():
                    raise
                return await self._salvage(graph, config, ctx, "deadline")
            except GraphRecursionError:
                return await self._salvage(graph, config, ctx, "recursion_limit")
        result: AskResult | None = final.get("result")
        if result is None:
            # No `result` in the state: the run stopped at an `interrupt` (the clarify step).
            # The question to ask is the interrupt's value; the answer comes with a new call.
            interrupts = final.get("__interrupt__") or []
            if not interrupts:
                raise RuntimeError("the graph ended without a result and without an interrupt")
            values = {**final, "clarification": str(interrupts[0].value)}
            return (
                compose_result(
                    values, ctx, timings_ms=dict(final.get("timings_ms") or {}), stop_reason=None
                ),
                values,
            )
        return result, final

    @staticmethod
    async def _stream_graph(
        graph: Any,
        inputs: Mapping[str, Any] | Command | None,
        config: Mapping[str, Any],
        ctx: AgentContext,
        emit: StreamEmit | None,
    ) -> dict[str, Any]:
        """The graph's final state, with its interrupts under `__interrupt__` as `ainvoke`
        returns it (it is built the same way: from the `values` and `updates` streams), while
        the nodes' `custom` events are passed to `emit`. "sync": each step's checkpoint is
        written before the next step starts, so a run cut off by the deadline or by a closed
        client leaves every finished step (and its chunks) in the checkpoint that `_salvage`
        reads. LangGraph's default is "async"."""
        latest: dict[str, Any] = {}
        interrupts: list[Any] = []
        async for mode, payload in graph.astream(
            inputs,
            config,
            context=ctx,
            stream_mode=["custom", "updates", "values"],
            durability="sync",
        ):
            if mode == "custom":
                if emit is not None:
                    emit(payload["event"], payload["data"])
            elif mode == "updates":
                if isinstance(payload, dict) and payload.get(INTERRUPT) is not None:
                    interrupts.extend(payload[INTERRUPT])
            else:
                latest = payload
        return {**latest, INTERRUPT: interrupts} if interrupts else latest

    async def _salvage(
        self, graph: Any, config: Mapping[str, Any], ctx: AgentContext, reason: StopReason
    ) -> tuple[AskResult, Mapping[str, Any]]:
        """The result of a run that was stopped: built from the last checkpoint, which holds the
        rounds finished so far (and the answer, if it was written before the stop)."""
        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values)
        timings = dict(values.get("timings_ms") or {})
        return compose_result(values, ctx, timings_ms=timings, stop_reason=reason), values


async def ask_docs(
    question: str,
    *,
    cfg: KnowledgeBaseConfig,
    retriever: AgentRetriever,
    settings: ServiceSettings,
    runtime: AgentRuntime | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    caller_id: str = "local",
    acl_groups: tuple[str, ...] = ("public",),
    deadline_ms: int | None = None,
    model: Any = None,
    helper_model: Any = None,
    judge_model: Any = None,
    filters: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    resume_token: str | None = None,
    answer: str | None = None,
    emit: StreamEmit | None = None,
) -> AskResult:
    """Answer `question` from knowledge base `cfg` with the ask_docs graph (see the module doc).
    A run that asked the caller back continues with `resume_token` and the caller's `answer`.

    `retriever` is the KB's `SearchPipeline`; `acl_groups` the caller's document groups.
    Requests that may overlap must share one `runtime`: it owns the session locks. Without one
    a runtime is made for this call alone."""
    runtime = runtime or AgentRuntime.from_settings(settings)
    return await runtime.ask(
        question,
        cfg=cfg,
        retriever=retriever,
        session_id=session_id,
        request_id=request_id,
        caller_id=caller_id,
        acl_groups=acl_groups,
        deadline_ms=deadline_ms,
        model=model,
        helper_model=helper_model,
        judge_model=judge_model,
        filters=filters,
        run_id=run_id,
        resume_token=resume_token,
        answer=answer,
        emit=emit,
    )
