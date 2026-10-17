"""Deterministic fakes for unit tests (never LlamaIndex MockLLM: it silently echoes the prompt)."""

from rag.llama_guard import LLMCallForbiddenError, RaisingLLM
from rag.testing.fakes import FakeChatModel, FakeEmbedding, use_safe_llama_settings
from rag.testing.observe import memory_observability, metric_points, observe_settings
from rag.testing.retrieval import FakeRetriever

__all__ = [
    "FakeChatModel",
    "FakeEmbedding",
    "FakeRetriever",
    "LLMCallForbiddenError",
    "RaisingLLM",
    "memory_observability",
    "metric_points",
    "observe_settings",
    "use_safe_llama_settings",
]
