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

from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig, ServiceSettings
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
