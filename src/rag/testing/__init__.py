"""Deterministic fakes for unit tests (never LlamaIndex MockLLM: it silently echoes the prompt)."""

from rag.llama_guard import LLMCallForbiddenError, RaisingLLM
from rag.testing.fakes import FakeChatModel, FakeEmbedding, use_safe_llama_settings
from rag.testing.retrieval import FakeRetriever

__all__ = [
    "FakeChatModel",
    "FakeEmbedding",
    "FakeRetriever",
    "LLMCallForbiddenError",
    "RaisingLLM",
    "use_safe_llama_settings",
]
