"""Running the ask_docs graph for one request (design s13-3, s16-7, s16-8; spec P3-01).

    runtime = AgentRuntime.from_settings(settings)    # one per process: it owns the session locks
    result = await ask_docs("How do I push a model?", cfg=cfg, retriever=pipeline,
                            settings=settings, runtime=runtime, session_id="s1")

One call:

1. takes the lock of the session (the caller's, in that knowledge base: `session_key`;
   requests of one session run one after another; the wait counts
   against the deadline: `AgentDeadlineExceeded` when it does not end in time) and loads the
   session's last turns;
2. runs the graph on the checkpoint thread `session_id:request_id` with `recursion_limit` set,
   inside the total deadline: when the deadline passes the run is cancelled and the result is
   built from the last checkpoint (the chunks found so far, no generated text), flagged "not
   confirmed that everything was searched"; the soft part of the deadline (no round starts when
   less than the answer's time is left) is inside the graph (`plan_next`);
3. stores the turn (question and answer text) in the session's history.

The checkpoint thread is `session_id:run_id`, with a `run_id` the server makes for each call:
the caller's request id (an `X-Request-Id` header, which anyone can repeat) is only for
correlation, so two runs never write to, or get rebuilt from, one thread.

A call that carried history is flagged `cache_allowed=False` (the exact cache of P5 neither
reads nor writes it, s12-1).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError

from rag.agent.budget import TokenBudget
from rag.agent.context import AgentContext, AgentRetriever
from rag.agent.graph import build_graph
from rag.agent.hooks import AgentHooks
from rag.agent.result import compose_result
from rag.agent.session import HistoryTurn, SessionLocks, SessionStore
from rag.agent.state import MSGPACK_ALLOWED, thread_id
from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.factories import (
    CheckpointerSource,
    make_checkpointer_source,
    make_session_locks,
    make_session_store,
)
from rag.observe import new_request_id
from rag.retrieve.deadline import Deadline
from rag.types import AnswerStatus, AskResult, ErrorCategory, StopReason


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
        hooks: AgentHooks | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.store = store
        self.locks = locks
        self.checkpointers = checkpointers
        self.hooks = hooks or AgentHooks()
        self.clock = clock
        self._chat: Any = None

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
        return cls(
            settings,
            store=store,
            locks=make_session_locks(settings.session_lock_url),
            checkpointers=checkpointers,
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
        """The `online` chat model, built once (the answer layer would build one per call)."""
        if self._chat is None:
            from rag.llm import make_chat_model

            self._chat = make_chat_model("online", self.settings)
        return self._chat

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
        filters: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> AskResult:
        """Run one request; see the module doc. `ValueError` for a session id that is not
        1 to 64 letters, digits, dot, dash or underscore (and not `-`). `run_id` names the
        checkpoint thread (default: a new one); only a caller that deliberately continues a
        thread passes it (P3-02 resumes with the run id of its `resume_token`)."""
        request_id = request_id or new_request_id()
        thread = thread_id(session_id, run_id or new_request_id())
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
            hooks=self.hooks,
            request_id=request_id,
            session_id=session_id,
            thread_id=thread,
            acl_groups=tuple(acl_groups),
            filters=filters,
        )
        async with AsyncExitStack() as stack:
            history: list[HistoryTurn] = []
            key = ""
            if session_id is not None:
                key = self.session_key(caller_id, cfg.kb, session_id)
                await self._hold_session(stack, key, deadline)
                history = await self.store.load(key)
            ctx.used_history = bool(history)
            result = await self._run(question, history, ctx)
            result = result.model_copy(
                update={
                    "timings_ms": {
                        **result.timings_ms,
                        "total": round((time.perf_counter() - began) * 1000.0, 3),
                    }
                }
            )
            if (
                session_id is not None
                and result.answer
                and result.status != AnswerStatus.NEED_CLARIFICATION
            ):
                await self.store.append(
                    key,
                    HistoryTurn(request_id=request_id, question=question, answer=result.answer),
                )
        return result

    async def _hold_session(self, stack: AsyncExitStack, key: str, deadline: Deadline) -> None:
        """Take the session's lock within the request's deadline."""
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

    async def _run(self, question: str, history: list[HistoryTurn], ctx: AgentContext) -> AskResult:
        inputs = {
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "kb": ctx.cfg.kb,
            "question": question,
            "history": history,
        }
        async with self.checkpointers.open() as saver:
            graph = build_graph(saver)
            config = self.run_config(ctx.thread_id)
            running = asyncio.timeout(ctx.deadline.remaining_s())
            try:
                async with running:
                    # "sync": each step's checkpoint is written before the next step starts, so
                    # a run cut off by the deadline leaves every finished step (and its chunks)
                    # in the checkpoint that `_salvage` reads. LangGraph's default is "async".
                    final = await graph.ainvoke(inputs, config, context=ctx, durability="sync")
            except TimeoutError:
                if not running.expired():
                    raise
                return await self._salvage(graph, config, ctx, "deadline")
            except GraphRecursionError:
                return await self._salvage(graph, config, ctx, "recursion_limit")
        result: AskResult | None = final.get("result")
        if result is None:
            # No `result` in the state: the run stopped at an `interrupt` (the clarify hook,
            # P3-02). The question to ask is the interrupt's value; resuming is P3-02.
            interrupts = final.get("__interrupt__") or []
            if not interrupts:
                raise RuntimeError("the graph ended without a result and without an interrupt")
            values = {**final, "clarification": str(interrupts[0].value)}
            result = compose_result(
                values, ctx, timings_ms=dict(final.get("timings_ms") or {}), stop_reason=None
            )
        return result

    async def _salvage(
        self, graph: Any, config: Mapping[str, Any], ctx: AgentContext, reason: StopReason
    ) -> AskResult:
        """The result of a run that was stopped: built from the last checkpoint, which holds the
        rounds finished so far (and the answer, if it was written before the stop)."""
        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values)
        timings = dict(values.get("timings_ms") or {})
        return compose_result(values, ctx, timings_ms=timings, stop_reason=reason)


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
    filters: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> AskResult:
    """Answer `question` from knowledge base `cfg` with the ask_docs graph (see the module doc).

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
        filters=filters,
        run_id=run_id,
    )
