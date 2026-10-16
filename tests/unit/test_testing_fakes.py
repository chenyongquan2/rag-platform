from __future__ import annotations

import asyncio
import math

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from rag.llama_guard import LLMCallForbiddenError
from rag.testing import FakeChatModel, FakeEmbedding, RaisingLLM, use_safe_llama_settings


def test_fake_embedding_deterministic_and_dim() -> None:
    emb = FakeEmbedding(dim=16)
    a1 = emb.get_text_embedding("push_to_hub")
    a2 = FakeEmbedding(dim=16).get_text_embedding("push_to_hub")
    b = emb.get_text_embedding("from_pretrained")

    assert len(a1) == 16
    assert a1 == a2
    assert a1 != b
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)
    # Query and text embeddings of the same string are identical, so an exact match ranks first.
    assert emb.get_query_embedding("push_to_hub") == a1
    assert emb.get_text_embedding_batch(["push_to_hub", "from_pretrained"]) == [a1, b]
    assert asyncio.run(emb.aget_query_embedding("push_to_hub")) == a1
    assert len(FakeEmbedding(dim=1024).get_text_embedding("x")) == 1024
    with pytest.raises(ValueError):
        FakeEmbedding(dim=0)


def test_fake_chat_model_scripted() -> None:
    model = FakeChatModel(responses=["first [1]", AIMessage(content="second")])

    assert model.invoke("q1").content == "first [1]"
    assert model.invoke([HumanMessage(content="q2")]).content == "second"
    assert len(model.calls) == 2
    assert model.calls[0][0].content == "q1"
    assert model.calls[1][0].content == "q2"
    with pytest.raises(AssertionError, match="exhausted"):
        model.invoke("q3")


class _Route(BaseModel):
    route: str


def test_fake_chat_model_structured_output() -> None:
    scripted = AIMessage(
        content="",
        tool_calls=[{"name": "_Route", "args": {"route": "search"}, "id": "call-1"}],
    )
    model = FakeChatModel(responses=[scripted])

    structured = model.with_structured_output(_Route, method="function_calling")

    assert structured.invoke("which route?") == _Route(route="search")


def test_raising_llm_raises() -> None:
    llm = RaisingLLM()
    msg = [ChatMessage(role="user", content="hi")]
    assert llm.metadata.model_name == "raising-llm"
    with pytest.raises(LLMCallForbiddenError):
        llm.complete("hi")
    with pytest.raises(LLMCallForbiddenError):
        llm.chat(msg)
    with pytest.raises(LLMCallForbiddenError):
        llm.stream_complete("hi")
    with pytest.raises(LLMCallForbiddenError):
        llm.stream_chat(msg)
    with pytest.raises(LLMCallForbiddenError):
        asyncio.run(llm.acomplete("hi"))
    with pytest.raises(LLMCallForbiddenError):
        asyncio.run(llm.achat(msg))
    with pytest.raises(LLMCallForbiddenError):
        asyncio.run(llm.astream_complete("hi"))
    with pytest.raises(LLMCallForbiddenError):
        asyncio.run(llm.astream_chat(msg))


def test_settings_llm_is_raising_stub_in_tests() -> None:
    # Installed by the autouse fixture in tests/conftest.py.
    assert isinstance(Settings.llm, RaisingLLM)
    assert isinstance(Settings.embed_model, FakeEmbedding)
    with pytest.raises(LLMCallForbiddenError):
        Settings.llm.complete("anything")

    with use_safe_llama_settings(dim=4):
        assert len(Settings.embed_model.get_text_embedding("x")) == 4
    assert isinstance(Settings.llm, RaisingLLM)
