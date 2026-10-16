"""Deterministic fakes for unit tests (never LlamaIndex MockLLM: it silently echoes the prompt)."""

from rag.llama_guard import LLMCallForbiddenError, RaisingLLM
from rag.testing.fakes import FakeChatModel, FakeEmbedding, use_safe_llama_settings

__all__ = [
    "FakeChatModel",
    "FakeEmbedding",
    "LLMCallForbiddenError",
    "RaisingLLM",
    "use_safe_llama_settings",
]
