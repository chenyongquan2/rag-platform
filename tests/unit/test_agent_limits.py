"""Spec P3-01 AC9-AC11: total deadline, token cap and recursion limit of the ask_docs graph."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

from rag.agent import (
    AgentDeadlineExceeded,
    AgentHooks,
    AgentRuntime,
    RequestTokenCapExceeded,
    TokenBudget,
    ask_docs,
    guard_model,
)
from rag.cost import CostLedger
from rag.llm import make_chat_model
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel
from rag.testing.agent import (
    FakeClock,
    HangingChatModel,
    ScriptedSearchBackend,
    agent_settings,
    kb_config,
)
from rag.types import AnswerStatus

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


async def _never_enough(state: Any, ctx: Any) -> str:
    return "insufficient"


def _hooks() -> AgentHooks:
    return AgentHooks(judge=_never_enough)


def test_deadline_answers_from_collected_chunks() -> None:
    clock = FakeClock()
    backend = ScriptedSearchBackend(KB, on_call=lambda q: clock.advance_ms(7000))
    settings = agent_settings(agent_deadline_ms=10_000, agent_answer_reserve_ms=4_000)
    runtime = AgentRuntime.from_settings(settings, hooks=_hooks(), clock=clock)
    model = StreamingFakeChatModel(responses=["Use push_to_hub [1]."])

    result = asyncio.run(
        ask_docs(
            "How do I push a model?",
            cfg=kb_config(KB, max_rounds=5),
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=model,
        )
    )

    # Round 1 used 7 of 10 s; 3 s are left and the answer needs 4: no second round, but the
    # answer is still written from what was found, with the flag.
    assert result.rounds == 1 and len(backend.calls) == 1 and len(model.calls) == 1
    assert result.status == AnswerStatus.ANSWERED and result.found
    assert (result.coverage_unconfirmed, result.stop_reason) == (True, "deadline")
    assert result.notice and "Not confirmed" in result.notice
    assert "agent_deadline" in result.error_categories
    assert result.cache_allowed is False  # incomplete: never cached
    # The search was given the total time minus the time kept for the answer.
    assert backend.calls[0]["deadline_ms"] == 6000


def test_hard_deadline_returns_collected_chunks() -> None:
    backend = ScriptedSearchBackend(KB)
    model = HangingChatModel(responses=["Use push_to_hub [1]."], hang_s=30.0)
    runtime = AgentRuntime.from_settings(agent_settings())
    started = time.monotonic()
    result = asyncio.run(
        ask_docs(
            "How do I push a model?",
            cfg=kb_config(KB),
            retriever=backend,
            settings=agent_settings(),
            runtime=runtime,
            model=model,
            deadline_ms=500,
        )
    )
    assert time.monotonic() - started < 10  # stopped by the deadline, not by the model
    assert model.started.is_set()
    # Cut off while the model was working: the chunks already found, no generated answer.
    assert (result.status, result.found, result.answer) == (AnswerStatus.NOT_FOUND, False, "")
    assert [c.chunk_id for c in result.sources] == [
        "hf_docs:p1.md#0",
        "hf_docs:p2.md#0",
        "hf_docs:p3.md#0",
    ]
    assert result.rounds == 1
    assert (result.coverage_unconfirmed, result.stop_reason) == (True, "deadline")
    assert result.notice and "agent_deadline" in result.error_categories
    assert result.timings_ms["total"] >= 400
    assert result.cache_allowed is False


def test_waiting_for_the_lock_counts_against_the_deadline() -> None:
    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings())

    async def main() -> float:
        async with runtime.locks.hold(
            runtime.session_key("local", KB, "s1")
        ):  # another request of this session is running
            began = time.monotonic()
            with pytest.raises(AgentDeadlineExceeded) as info:
                await ask_docs(
                    "q",
                    cfg=kb_config(KB),
                    retriever=backend,
                    settings=agent_settings(),
                    runtime=runtime,
                    model=FakeChatModel(responses=[]),
                    session_id="s1",
                    request_id="r2",
                    deadline_ms=300,
                )
            assert info.value.error_category == "agent_deadline"
            return time.monotonic() - began

    assert asyncio.run(main()) < 5
    assert backend.calls == []


def test_tokens_counted_from_usage_or_estimate() -> None:
    budget = TokenBudget(cap=300, answer_reserve=100)
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    inner = FakeChatModel(responses=[AIMessage(content="ok", usage_metadata=usage), "plain reply"])
    model = guard_model(inner, budget)

    asyncio.run(model.ainvoke("hello"))
    assert budget.spent == 15  # the gateway's own count when there is one
    asyncio.run(model.ainvoke("a" * 900))
    assert budget.spent >= 15 + 300  # else our estimate of prompt and reply (3 characters each)
    assert not budget.cap_hit
    with pytest.raises(RequestTokenCapExceeded):
        asyncio.run(model.ainvoke("one more"))
    assert budget.cap_hit and len(inner.calls) == 2  # refused before the model was called

    # Calls that are not the answer leave room for it.
    other = TokenBudget(cap=100, answer_reserve=30)
    other.add(60)
    assert not other.exhausted() and other.exhausted(reserve=50)
    assert other.remaining() == 40


def test_token_cap_stops_further_model_calls() -> None:
    settings = agent_settings(agent_token_cap=100, agent_answer_reserve_tokens=0)
    backend = ScriptedSearchBackend(KB)
    # No citation in the first reply: the answer layer would retry, but the cap is used up.
    model = StreamingFakeChatModel(responses=["no citation here", "still none"])
    result = asyncio.run(
        ask_docs(
            "How do I push a model?",
            cfg=kb_config(KB),
            retriever=backend,
            settings=settings,
            model=model,
        )
    )
    assert len(model.calls) == 1
    assert result.tokens_used >= 100
    assert (result.coverage_unconfirmed, result.stop_reason) == (True, "token_cap")
    assert "token_cap_reached" in result.error_categories
    assert "budget_exceeded" not in result.error_categories and result.generation_error is None
    assert (result.status, result.answer) == (AnswerStatus.NOT_FOUND, "")
    assert len(result.sources) == 3  # the chunks found so far are still returned


def test_recursion_limit_is_set_and_enforced() -> None:
    settings = agent_settings(agent_recursion_limit=8)
    runtime = AgentRuntime.from_settings(settings, hooks=_hooks())
    assert runtime.run_config("s1:r1") == {
        "configurable": {"thread_id": "s1:r1"},
        "recursion_limit": 8,
    }
    backend = ScriptedSearchBackend(KB)
    result = asyncio.run(
        ask_docs(
            "q",
            cfg=kb_config(KB, max_rounds=100),  # the KB would allow 100 rounds
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=StreamingFakeChatModel(default_response="Answer [1]."),
        )
    )
    assert 1 <= len(backend.calls) <= 3  # stopped by the recursion limit, not by max_rounds
    assert (result.coverage_unconfirmed, result.stop_reason) == (True, "recursion_limit")
    assert "agent_deadline" in result.error_categories
    assert len(result.sources) == 3 and result.status == AnswerStatus.NOT_FOUND


def _sse_gateway(content: str, usage: dict[str, int]) -> httpx.MockTransport:
    """An OpenAI-compatible gateway that streams `content` and ends with a usage event."""

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def handler(request: httpx.Request) -> httpx.Response:
        events = [chunk({"role": "assistant", "content": ""})]
        events += [chunk({"content": content[i : i + 6]}) for i in range(0, len(content), 6)]
        events += [chunk({}, "stop"), {**chunk({}), "choices": [], "usage": usage}]
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=payload.encode()
        )

    return httpx.MockTransport(handler)


def test_gateway_usage_is_counted_against_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("CHAT_BACKEND", "CHAT_ONLINE", "CHAT_MODEL", "MODEL_PRICES", "BUDGET_STOP_CNY"):
        monkeypatch.delenv(key, raising=False)
    settings = agent_settings(
        llm_base_url="http://gateway.test/v1",
        llm_api_key="sk-unit-test-key-0123456789",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
        agent_token_cap=1000,
        agent_answer_reserve_tokens=0,
    )
    usage = {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020}
    # A real chat model of the factory over a mocked gateway: the stream ends with usage.
    model = make_chat_model(
        "online",
        settings,
        ledger=CostLedger.from_settings(settings),
        async_transport=_sse_gateway("Use push_to_hub [1].", usage),
    )
    result = asyncio.run(
        ask_docs(
            "How do I push a model?",
            cfg=kb_config(KB),
            retriever=ScriptedSearchBackend(KB),
            settings=settings,
            model=model,
        )
    )
    assert result.found and result.tokens_used == 1020  # the gateway's count, not an estimate
    assert not result.coverage_unconfirmed  # the cap was passed by the last call, not blocked


def test_cut_run_reports_the_history_it_was_given() -> None:
    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings())
    model = StreamingFakeChatModel(default_response="Answer [1].")

    async def main() -> Any:
        await ask_docs(
            "q1", cfg=kb_config(KB), retriever=backend, settings=agent_settings(),
            runtime=runtime, model=model, session_id="s1",
        )  # fmt: skip
        # Cut before the first checkpoint exists: the flags come from the loaded history.
        return await ask_docs(
            "q2", cfg=kb_config(KB), retriever=backend, settings=agent_settings(),
            runtime=runtime, model=model, session_id="s1", deadline_ms=1,
        )  # fmt: skip

    result = asyncio.run(main())
    assert result.stop_reason == "deadline"
    assert (result.used_history, result.cache_allowed) == (True, False)


def test_checkpoint_is_written_before_the_next_step_starts() -> None:
    """A saver that is slow to write: with `durability="sync"` every step's checkpoint is
    complete before the next step starts, so a run cut off later can be rebuilt from it."""
    from langgraph.checkpoint.memory import InMemorySaver

    class SlowSaver(InMemorySaver):
        completed = 0

        async def aput(self, *args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.1)
            result = await super().aput(*args, **kwargs)
            self.completed += 1
            return result

    class Source:
        def __init__(self, saver: SlowSaver) -> None:
            self.saver = saver

        def open(self) -> Any:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def opened() -> Any:
                yield self.saver

            return opened()

    saver = SlowSaver()
    seen: list[int] = []
    backend = ScriptedSearchBackend(KB, on_call=lambda q: seen.append(saver.completed))
    runtime = AgentRuntime.from_settings(agent_settings())
    runtime.checkpointers = Source(saver)  # type: ignore[assignment]
    asyncio.run(
        ask_docs(
            "q", cfg=kb_config(KB), retriever=backend, settings=agent_settings(),
            runtime=runtime, model=StreamingFakeChatModel(default_response="Answer [1]."),
        )
    )  # fmt: skip
    # Before the search starts every earlier checkpoint is written (four for this graph; with
    # LangGraph's default, durability "async", none is written yet at that point).
    assert seen == [4]
