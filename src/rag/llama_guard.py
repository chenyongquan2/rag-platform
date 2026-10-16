"""Keep LlamaIndex from calling a generation model (design s11-1).

`Settings.llm = None` is not enough: LlamaIndex then swaps in MockLLM, which silently echoes the
prompt. Instead `Settings.llm` is a stub that raises on every call, and `Settings.embed_model`
is always set explicitly (LlamaIndex would otherwise default to OpenAI).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.llm import LLM


class LLMCallForbiddenError(RuntimeError):
    """A LlamaIndex component tried to call a generation model."""


class RaisingLLM(LLM):
    """LlamaIndex LLM that raises `LLMCallForbiddenError` on any chat/complete call.

    Reading `metadata` is allowed (LlamaIndex reads it when building helpers); calling is not.
    """

    @classmethod
    def class_name(cls) -> str:
        return "RaisingLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="raising-llm", is_chat_model=True)

    def _forbidden(self, method: str) -> NoReturn:
        raise LLMCallForbiddenError(
            f"LlamaIndex called Settings.llm.{method}(); the retrieval layer must not call a "
            "generation model. Pass an explicit LLM where generation is intended."
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        self._forbidden("chat")

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self._forbidden("complete")

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        self._forbidden("stream_chat")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        self._forbidden("stream_complete")

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        self._forbidden("achat")

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self._forbidden("acomplete")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        self._forbidden("astream_chat")

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        self._forbidden("astream_complete")


def install_llama_guard(embed_model: BaseEmbedding) -> None:
    """Set `Settings.llm` to `RaisingLLM` and `Settings.embed_model` to an explicit model."""
    Settings.llm = RaisingLLM()
    Settings.embed_model = embed_model
