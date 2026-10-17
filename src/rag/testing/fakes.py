"""Fake embedding and chat models, and a helper that makes LlamaIndex `Settings` safe."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import Field, PrivateAttr

from rag.llama_guard import install_llama_guard

_UINT32_MAX = float(2**32 - 1)


class FakeEmbedding(BaseEmbedding):
    """Deterministic, hash-based unit vectors of size `dim`.

    The same text gives the same vector in every process (SHA-256, not Python `hash()`), and
    query and text embeddings of the same string are identical.
    """

    model_name: str = "fake-embedding"
    dim: int = Field(default=8, gt=0)

    @classmethod
    def class_name(cls) -> str:
        return "FakeEmbedding"

    def _vector(self, text: str) -> Embedding:
        values: list[float] = []
        block = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{block}\x00{text}".encode()).digest()
            values.extend(2.0 * (u / _UINT32_MAX) - 1.0 for u in struct.unpack("<8I", digest))
            block += 1
        values = values[: self.dim]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._vector(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return [self._vector(t) for t in texts]


class FakeChatModel(BaseChatModel):
    """LangChain chat model that returns scripted responses in order.

    - A `str` response becomes `AIMessage(content=...)`; an `AIMessage` is returned as is, so a
      script can carry `tool_calls` for `with_structured_output(..., method="function_calling")`.
    - Every call's input messages are recorded in `calls`.
    - When the script is exhausted the model raises instead of cycling, so a test that makes an
      unexpected extra model call fails; with `default_response` set it answers that instead
      (the `CHAT_BACKEND=fake` fallback, spec P1-01).
    """

    responses: list[str | AIMessage] = Field(default_factory=list)
    default_response: str | AIMessage | None = None
    _cursor: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        """Record the call and return the next scripted reply."""
        self._calls.append(list(messages))
        if self._cursor < len(self.responses):
            scripted = self.responses[self._cursor]
            self._cursor += 1
        elif self.default_response is not None:
            scripted = self.default_response
        else:
            raise AssertionError(
                f"FakeChatModel script exhausted: call {len(self._calls)} but only "
                f"{len(self.responses)} scripted responses"
            )
        return scripted if isinstance(scripted, AIMessage) else AIMessage(content=scripted)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # Scripted AIMessages carry their own tool calls; the tool list is not needed.
        return self

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: Literal["function_calling"] = "function_calling",
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | Any]:
        """Accepts `method="function_calling"` like ChatOpenAI; other methods are not faked."""
        if method != "function_calling":
            raise NotImplementedError(
                f"FakeChatModel only fakes method='function_calling', got {method!r}"
            )
        return super().with_structured_output(schema, include_raw=include_raw, **kwargs)


class StreamingFakeChatModel(FakeChatModel):
    """`FakeChatModel` whose streams (`stream`, `astream`) send each scripted text reply in
    pieces of `chunk_chars` characters, like a gateway sending tokens (spec P1-02).

    `invoke` still returns the whole reply. Replies with tool calls cannot be streamed: use
    `FakeChatModel` for structured output.
    """

    chunk_chars: int = Field(default=5, gt=0)

    def _pieces(self, messages: list[BaseMessage]) -> list[ChatGenerationChunk]:
        message = self._next_message(messages)
        if message.tool_calls:
            raise NotImplementedError("StreamingFakeChatModel does not stream tool calls")
        content = str(message.content)
        size = self.chunk_chars
        texts = [content[i : i + size] for i in range(0, len(content), size)] or [""]
        return [ChatGenerationChunk(message=AIMessageChunk(content=text)) for text in texts]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._pieces(messages):
            if run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._pieces(messages):
            if run_manager is not None:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


@contextmanager
def use_safe_llama_settings(dim: int = 8) -> Iterator[None]:
    """Within the block, `Settings.llm` raises when called and `Settings.embed_model` is a
    `FakeEmbedding(dim)`. The previous values are restored afterwards."""
    previous_llm = Settings._llm
    previous_embed = Settings._embed_model
    install_llama_guard(FakeEmbedding(dim=dim))
    try:
        yield
    finally:
        Settings._llm = previous_llm
        Settings._embed_model = previous_embed
