"""Fakes for the `ask_docs` graph tests (spec P3-01): scripted retrieval per question, a clock
the test moves by hand, a chat model that never finishes, and settings with in-memory stores.

Nothing here calls a network or a real model.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr

from rag.agent.budget import TokenBudget
from rag.agent.context import AgentContext
from rag.agent.hooks import AgentHooks
from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig, ServiceSettings
from rag.retrieve.deadline import Deadline
from rag.testing.fakes import FakeChatModel, StreamingFakeChatModel
from rag.testing.observe import observe_settings
from rag.testing.serve import FakeSearchPipeline, fake_chunk
from rag.types import RetrievalResult, RouteHit, ThresholdDecision


def pages_result(
    kb: str,
    pages: Sequence[int],
    *,
    degraded: bool = False,
    rerank_timeout: bool = False,
    not_found: bool = False,
    timings_ms: Mapping[str, float] | None = None,
) -> RetrievalResult:
    """A retrieval result whose chunks are the pages `pages` (best first), so a test can tell
    which request a chunk came from."""
    chunks = [fake_chunk(kb, n, rerank_score=round(0.95 - i / 10, 2)) for i, n in enumerate(pages)]
    return RetrievalResult(
        request_id="",
        kb=kb,
        index_version=f"kb_{kb}_v1",
        data_generation=7,
        degraded=degraded,
        rerank_timeout=rerank_timeout,
        timings_ms=dict(timings_ms or {"search": 1.0, "rerank": 2.0, "total": 3.0}),
        chunks=[] if not_found else chunks,
        below_threshold=chunks[:3] if not_found else [],
        not_found=not_found,
        rerank=[
            RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r, score=c.rerank_score)
            for r, c in enumerate(chunks, 1)
        ],
        threshold=ThresholdDecision(
            threshold=0.3, top_score=chunks[0].rerank_score if chunks else None, kept=len(chunks)
        ),
        config_version="cfg-fake",
    )


def scored_result(
    kb: str,
    scores: Sequence[float],
    *,
    pages: Sequence[int] | None = None,
    texts: Sequence[str] | None = None,
    threshold: float = 0.3,
) -> RetrievalResult:
    """A retrieval result for candidates with the given rerank `scores` (best first), cut by
    `threshold` the way the real chain does it: chunks at or above it are `chunks`; when none
    is, `not_found` is set and the best three are in `below_threshold`. `pages` names the page
    of each candidate (default 1, 2, ...), `texts` their text."""
    numbers = list(pages) if pages is not None else list(range(1, len(scores) + 1))
    candidates = [fake_chunk(kb, n, rerank_score=s) for n, s in zip(numbers, scores, strict=True)]
    if texts is not None:
        candidates = [
            c.model_copy(update={"text": t}) for c, t in zip(candidates, texts, strict=True)
        ]
    kept = [c for c in candidates if (c.rerank_score or 0.0) >= threshold]
    return RetrievalResult(
        request_id="",
        kb=kb,
        index_version=f"kb_{kb}_v1",
        data_generation=7,
        timings_ms={"search": 1.0, "rerank": 2.0, "total": 3.0},
        chunks=kept,
        below_threshold=[] if kept else candidates[:3],
        not_found=not kept,
        rerank=[
            RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r, score=c.rerank_score)
            for r, c in enumerate(candidates, 1)
        ],
        threshold=ThresholdDecision(
            threshold=threshold,
            top_score=candidates[0].rerank_score if candidates else None,
            passed=bool(kept),
            kept=len(kept),
        ),
        config_version="cfg-fake",
    )


class ScriptedSearchBackend(FakeSearchPipeline):
    """A `SearchBackend` that answers each question with the result scripted for it (else
    `default`), records every call, and runs `on_call(question)` first: a test uses it to move
    a fake clock, to block, or to count calls that run at the same time."""

    def __init__(
        self,
        kb: str,
        *,
        by_question: Mapping[str, RetrievalResult] | None = None,
        default: RetrievalResult | None = None,
        on_call: Callable[[str], None] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        super().__init__(kb, raises=raises)
        self.by_question = dict(by_question or {})
        self.default = default or pages_result(kb, [1, 2, 3])
        self.on_call = on_call

    def retrieve(
        self,
        question: str,
        kb: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "question": question,
                "kb": kb,
                "top_k": top_k,
                "filters": dict(filters or {}),
                "deadline_ms": deadline_ms,
                "request_id": request_id,
                "acl_groups": tuple(acl_groups) if acl_groups is not None else None,
            }
        )
        if self.on_call is not None:
            self.on_call(question)
        if self.raises is not None:
            raise self.raises
        result = self.by_question.get(question, self.default)
        return result.model_copy(
            update={
                "request_id": request_id or uuid.uuid4().hex,
                "kb": kb,
                "chunks": result.chunks[:top_k],
                "config_version": self.config_version,
            }
        )


class FakeClock:
    """A monotonic clock that only moves when the test says so."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        with self._lock:
            self.now += ms / 1000.0


class HangingChatModel(StreamingFakeChatModel):
    """Streams nothing and waits `hang_s` seconds before the first piece: a model gateway that
    does not answer. `started` is set when the stream begins."""

    hang_s: float = 30.0
    _started: threading.Event = PrivateAttr(default_factory=threading.Event)

    @property
    def started(self) -> threading.Event:
        return self._started

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._started.set()
        await asyncio.sleep(self.hang_s)
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


def agent_settings(**overrides: Any) -> ServiceSettings:
    """Observability in memory, checkpoints, sessions and locks in process memory."""
    defaults: dict[str, Any] = {
        "checkpoint_url": "memory://",
        "session_store_url": "memory://",
        "session_lock_url": "memory://",
    }
    return observe_settings(**{**defaults, **overrides})


def kb_config(kb: str = "hf_docs", *, max_rounds: int = 3, **agent: Any) -> KnowledgeBaseConfig:
    """An English KB open to every caller with `agent.max_rounds` set. The rewrite, clarify and
    route-model switches are off unless a test passes them (`rewrite=True`, ...), so a test
    that is not about them sees the P3-01 behaviour and calls no small model."""
    return KnowledgeBaseConfig(
        kb=kb,
        description=f"{kb} pages",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(max_rounds=max_rounds, **agent),
    )


def structured_reply(schema: str, usage: tuple[int, int] | None = None, **fields: Any) -> AIMessage:
    """What a model answers when asked for structured output with function calling: a tool call
    named after the schema class (`RewritePlan`, `RouteJudgement`) with `fields` as arguments.
    `usage` is (input tokens, output tokens) as the gateway would report them."""
    usage_metadata = (
        {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage)}
        if usage
        else None
    )
    return AIMessage(
        content="",
        tool_calls=[{"name": schema, "args": fields, "id": f"call-{schema}", "type": "tool_call"}],
        usage_metadata=usage_metadata,  # type: ignore[arg-type]
    )


def judge_reply(*grounded: bool, usage: tuple[int, int] | None = None) -> AIMessage:
    """The sufficiency judge's structured answer: one verdict per numbered sub-question."""
    return structured_reply(
        "SufficiencyJudgement",
        usage=usage,
        sub_questions=[{"n": i, "grounded": g} for i, g in enumerate(grounded, 1)],
    )


def multi_query_reply(*queries: tuple[int, list[str]], usage: tuple[int, int] | None = None) -> Any:
    """The correction step's multi-query answer: `(sub-question number, [phrasings])` pairs."""
    return structured_reply(
        "MultiQueryPlan",
        usage=usage,
        by_sub_question=[{"n": n, "queries": list(q)} for n, q in queries],
    )


def hyde_reply(*passages: tuple[int, str], usage: tuple[int, int] | None = None) -> Any:
    """The correction step's HyDE answer: `(sub-question number, passage)` pairs."""
    return structured_reply(
        "HydePlan",
        usage=usage,
        by_sub_question=[{"n": n, "passage": p} for n, p in passages],
    )


def make_context(
    *,
    cfg: KnowledgeBaseConfig | None = None,
    settings: ServiceSettings | None = None,
    helper_model: Any = None,
    judge_model: Any = None,
    model: Any = None,
    retriever: Any = None,
    total_ms: int = 30_000,
    clock: Any = None,
    cap: int | None = None,
) -> AgentContext:
    """An `AgentContext` for a test that calls a hook or a step on its own (no graph)."""
    settings = settings or agent_settings()
    cfg = cfg or kb_config("hf_docs", judge=True)
    kwargs: dict[str, Any] = {"clock": clock} if clock is not None else {}
    return AgentContext(
        settings=settings,
        cfg=cfg,
        retriever=retriever if retriever is not None else ScriptedSearchBackend(cfg.kb),
        deadline=Deadline.after_ms(total_ms, **kwargs),
        total_ms=total_ms,
        budget=TokenBudget(cap or settings.agent_token_cap, settings.agent_answer_reserve_tokens),
        model=model,
        helper_model=helper_model,
        judge_model=judge_model,
        hooks=AgentHooks(),
        request_id="req-test",
        session_id=None,
        thread_id="-:test",
    )


class RaisingChatModel(FakeChatModel):
    """A chat model whose every call raises `error` (a gateway that is down)."""

    error: Any = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        raise self.error


class SlowChatModel(FakeChatModel):
    """A chat model that waits `delay_s` seconds (a gateway that is slow) and then answers its
    scripted reply. The wait is a cancellable `asyncio.sleep`, so a timeout around the call
    ends it."""

    delay_s: float = 30.0

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(self.delay_s)
        return self._generate(messages, stop, None, **kwargs)


class SlowStreamingChatModel(StreamingFakeChatModel):
    """A streaming chat model that waits `delay_s` before each piece (a gateway that writes at
    the speed of a model). `pieces_sent` counts the pieces handed out, `started` is set when a
    stream begins and `closed` when it ends or is cancelled: a test that cancels a request looks
    at them to see that the model stopped. The wait is a cancellable `asyncio.sleep`."""

    delay_s: float = 0.05
    _sent: list[str] = PrivateAttr(default_factory=list)
    _started: threading.Event = PrivateAttr(default_factory=threading.Event)
    _closed: threading.Event = PrivateAttr(default_factory=threading.Event)

    @property
    def pieces_sent(self) -> int:
        return len(self._sent)

    @property
    def started(self) -> threading.Event:
        return self._started

    @property
    def closed(self) -> threading.Event:
        return self._closed

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._started.set()
        try:
            for chunk in self._pieces(messages):
                await asyncio.sleep(self.delay_s)
                self._sent.append(chunk.text)
                if run_manager is not None:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        finally:
            self._closed.set()
