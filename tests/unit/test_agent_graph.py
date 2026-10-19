"""Spec P3-01 AC1, AC3-AC7, AC12, AC14: the ask_docs graph (nodes, state, history, stubs).

Fake retrieval (`ScriptedSearchBackend`) and fake chat models only; checkpoints in memory unless
a test is about SQLite.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import (
    AgentHooks,
    AgentRuntime,
    RewriteOutcome,
    RouteDecision,
    ask_docs,
    decide_route,
)
from rag.answer import NOT_FOUND
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel
from rag.testing.agent import ScriptedSearchBackend, agent_settings, kb_config, pages_result
from rag.types import AnswerStatus, AskResult

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _ask(
    question: str,
    backend: ScriptedSearchBackend,
    model: Any,
    *,
    runtime: AgentRuntime | None = None,
    max_rounds: int = 3,
    **kwargs: Any,
) -> AskResult:
    settings = kwargs.pop("settings", None) or agent_settings()
    return asyncio.run(
        ask_docs(
            question,
            cfg=kb_config(KB, max_rounds=max_rounds),
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=model,
            **kwargs,
        )
    )


def _page_ids(result: AskResult) -> list[str]:
    return [c.chunk_id for c in result.sources]


def test_ask_docs_happy_path() -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1, 2, 3]))
    model = StreamingFakeChatModel(responses=["Use push_to_hub [1]. Or upload_folder [2]."])
    result = _ask(
        "How do I push a model?", backend, model, session_id="s1", request_id="req-1",
        acl_groups=("public", "team-a"),
    )  # fmt: skip

    assert (result.status, result.found) == (AnswerStatus.ANSWERED, True)
    assert result.answer == "Use push_to_hub. Or upload_folder."
    assert [c.chunk_id for s in result.sentences for c in s.citations] == [
        "hf_docs:p1.md#0",
        "hf_docs:p2.md#0",
    ]
    assert _page_ids(result)[:2] == ["hf_docs:p1.md#0", "hf_docs:p2.md#0"]
    assert (result.request_id, result.session_id) == ("req-1", "s1")
    # The checkpoint thread is the session plus a server-made run id, not the caller's request id.
    assert result.thread_id.startswith("s1:") and "req-1" not in result.thread_id
    assert (result.kb, result.rounds, result.resume_token) == (KB, 1, None)
    assert (result.index_version, result.data_generation) == ("kb_hf_docs_v1", 7)
    assert result.config_version == "cfg-fake"
    assert not result.coverage_unconfirmed and result.stop_reason is None
    assert result.error_categories == []
    assert result.tokens_used > 0
    assert set(result.timings_ms) >= {"route", "retrieve", "answer", "verify", "total"}

    (call,) = backend.calls  # one retrieval for the one question
    assert (call["question"], call["kb"], call["request_id"]) == (
        "How do I push a model?",
        KB,
        "req-1",
    )
    assert call["top_k"] == 8 and call["acl_groups"] == ("public", "team-a")
    assert 0 < call["deadline_ms"] <= 10_000  # the search gets part of the total time
    assert len(model.calls) == 1


def test_verify_propagates_retrieval_flags() -> None:
    degraded = pages_result(KB, [1, 2], degraded=True, rerank_timeout=True)
    backend = ScriptedSearchBackend(KB, default=degraded)
    model = StreamingFakeChatModel(responses=["Use push_to_hub [1]."])
    result = _ask("How do I push a model?", backend, model)
    assert result.found and result.degraded and result.rerank_timeout
    assert result.error_categories == ["degraded", "rerank_timeout"]
    assert result.cache_allowed is False  # an incomplete result is never cached (s12-1)

    # The answer layer's own failure travels too: no citation twice -> chunks only.
    model = StreamingFakeChatModel(responses=["no citation here", "still none"])
    result = _ask("How do I push a model?", ScriptedSearchBackend(KB), model)
    assert (result.status, result.found, result.answer) == (AnswerStatus.NOT_FOUND, False, "")
    assert result.answer_parse_failed and "answer_parse_failed" in result.error_categories
    assert result.citation_check.parse_attempts == 2
    assert result.cache_allowed is False
    assert len(result.sources) == 3  # the caller still gets the retrieved chunks


def test_not_found_makes_no_model_call() -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1, 2, 3], not_found=True))
    model = FakeChatModel(responses=[])  # any call raises
    result = _ask("What is the airspeed of a swallow?", backend, model)
    assert (result.status, result.found, result.sources) == (AnswerStatus.NOT_FOUND, False, [])
    assert result.answer == NOT_FOUND["en"]
    assert model.calls == [] and len(backend.calls) == 1


def test_route_rules_skip_retrieval_for_greetings() -> None:
    backend = ScriptedSearchBackend(KB)
    model = FakeChatModel(responses=[])
    for text in ("hello!", "Thanks", "  Thank you.  "):
        result = _ask(text, backend, model)
        assert (result.status, result.found, result.rounds) == (AnswerStatus.ANSWERED, False, 0)
        assert result.sources == [] and "documentation" in result.answer
    zh = _ask("你好", backend, model)
    assert "文档" in zh.answer and zh.rounds == 0
    assert backend.calls == [] and model.calls == []

    # A question that merely contains a greeting is a question.
    backend = ScriptedSearchBackend(KB)
    result = _ask(
        "How do I say hello in the pipeline API?",
        backend,
        StreamingFakeChatModel(responses=["Use pipeline [1]."]),
    )
    assert result.found and len(backend.calls) == 1


def test_route_calls_no_model() -> None:
    # The rules are a plain synchronous function: they are given no model to call.
    assert not inspect.iscoroutinefunction(decide_route)
    assert list(inspect.signature(decide_route).parameters) == ["question", "kb", "history"]
    assert decide_route("How do I log in?", kb=KB, history=[]) == RouteDecision(
        retrieve=True, kb=KB, needs_clarification=False, reason="rules"
    )
    assert decide_route("hi", kb=KB, history=[]).retrieve is False
    assert decide_route("", kb=KB, history=[]).retrieve is False


def _stepwise_hooks() -> AgentHooks:
    """Round 1 of "q-a" is judged insufficient and rewritten to "q-b"; everything else is
    sufficient at once."""

    async def rewrite(state: Any, ctx: Any) -> RewriteOutcome:
        if state["round"] >= 1:
            return RewriteOutcome(sub_questions=("q-b",))
        return RewriteOutcome()

    async def judge(state: Any, ctx: Any) -> str:
        return "insufficient" if state["question"] == "q-a" and state["round"] < 2 else "sufficient"

    return AgentHooks(rewrite=rewrite, judge=judge)


def test_rounds_accumulate_and_route_clears_them() -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "q-a": pages_result(KB, [1, 2]),
            "q-b": pages_result(KB, [3]),
            "q-c": pages_result(KB, [9]),
        },
    )
    settings = agent_settings()
    runtime = AgentRuntime.from_settings(settings, hooks=_stepwise_hooks())
    model = StreamingFakeChatModel(default_response="Answer [1].")

    first = _ask("q-a", backend, model, runtime=runtime, session_id="s1", run_id="same-run")
    # Both rounds' chunks are in the context of the one answer (reducer: append across rounds).
    assert first.rounds == 2
    assert sorted(_page_ids(first)) == ["hf_docs:p1.md#0", "hf_docs:p2.md#0", "hf_docs:p3.md#0"]
    assert [c["question"] for c in backend.calls] == ["q-a", "q-b"]

    # The same thread run again (a resumed or misused thread): route clears what the first run
    # left in the state, so only this run's chunks are in the answer.
    second = _ask("q-c", backend, model, runtime=runtime, session_id="s1", run_id="same-run")
    assert second.thread_id == first.thread_id == "s1:same-run"
    assert second.rounds == 1
    assert _page_ids(second) == ["hf_docs:p9.md#0"]
    values = asyncio.run(runtime.state_values(second.thread_id))
    assert [r.sub_question for r in values["rounds"]] == ["q-c"]


def test_same_session_requests_do_not_share_chunks() -> None:
    backend = ScriptedSearchBackend(
        KB, by_question={"q1": pages_result(KB, [1, 2]), "q2": pages_result(KB, [5])}
    )
    runtime = AgentRuntime.from_settings(agent_settings())
    model = StreamingFakeChatModel(default_response="Answer [1].")

    r1 = _ask("q1", backend, model, runtime=runtime, session_id="s1", request_id="r1")
    r2 = _ask("q2", backend, model, runtime=runtime, session_id="s1", request_id="r2")

    assert r1.thread_id != r2.thread_id and r2.thread_id.startswith("s1:")
    assert _page_ids(r1) == ["hf_docs:p1.md#0", "hf_docs:p2.md#0"]
    assert _page_ids(r2) == ["hf_docs:p5.md#0"]  # nothing of request 1
    state_2 = asyncio.run(runtime.state_values(r2.thread_id))
    assert [c.chunk_id for r in state_2["rounds"] for c in r.chunks] == ["hf_docs:p5.md#0"]
    # What the second request does carry over is the text of the first turn, nothing else.
    (turn,) = state_2["history"]
    assert (turn.request_id, turn.question) == ("r1", "q1")
    assert "hf_docs" not in turn.model_dump_json()
    assert [
        t.request_id
        for t in asyncio.run(runtime.store.load(runtime.session_key("local", KB, "s1")))
    ] == ["r1", "r2"]


def test_call_with_history_is_not_cache_eligible() -> None:
    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings())
    model = StreamingFakeChatModel(default_response="Answer [1].")
    first = _ask("q1", backend, model, runtime=runtime, session_id="s1", request_id="r1")
    assert (first.used_history, first.cache_allowed) == (False, True)
    second = _ask("q2", backend, model, runtime=runtime, session_id="s1", request_id="r2")
    assert (second.used_history, second.cache_allowed) == (True, False)
    other = _ask("q3", backend, model, runtime=runtime, session_id="s2", request_id="r3")
    assert (other.used_history, other.cache_allowed) == (False, True)
    nosession = _ask("q4", backend, model, runtime=runtime, request_id="r4")
    assert nosession.session_id is None and nosession.thread_id.startswith("-:")
    assert (nosession.used_history, nosession.cache_allowed) == (False, True)
    # No session, no stored turn.
    assert [
        t.request_id
        for t in asyncio.run(runtime.store.load(runtime.session_key("local", KB, "s2")))
    ] == ["r3"]


def test_extension_points_are_wired() -> None:
    model = StreamingFakeChatModel(default_response="Answer [1].")

    # rewrite: one retrieval per sub-question, the chunks of both reach the answer.
    async def rewrite(state: Any, ctx: Any) -> RewriteOutcome:
        return RewriteOutcome(sub_questions=("part a", "part b"))

    backend = ScriptedSearchBackend(
        KB, by_question={"part a": pages_result(KB, [1]), "part b": pages_result(KB, [2])}
    )
    runtime = AgentRuntime.from_settings(agent_settings(), hooks=AgentHooks(rewrite=rewrite))
    result = _ask("compare a and b", backend, model, runtime=runtime)
    assert [c["question"] for c in backend.calls] == ["part a", "part b"]
    assert sorted(_page_ids(result)) == ["hf_docs:p1.md#0", "hf_docs:p2.md#0"]
    assert result.rounds == 1  # both sub-questions are one round

    # judge: "insufficient" loops through rewrite and retrieve until the KB's max_rounds.
    async def never_enough(state: Any, ctx: Any) -> str:
        return "insufficient"

    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings(), hooks=AgentHooks(judge=never_enough))
    result = _ask("q", backend, model, runtime=runtime, max_rounds=2)
    assert result.rounds == 2 and len(backend.calls) == 2
    assert (result.coverage_unconfirmed, result.stop_reason) == (True, "max_rounds")
    assert result.cache_allowed is False
    assert result.found and result.error_categories == []  # a limit of the KB, not a failure

    # clarify: the route flags an unclear question, the run ends with the question to ask.
    def vague(question: str, *, kb: str, history: Any) -> RouteDecision:
        return RouteDecision(retrieve=True, kb=kb, needs_clarification=question == "it?")

    async def clarify(state: Any, ctx: Any) -> str | None:
        return "Which library do you mean?"

    backend = ScriptedSearchBackend(KB)
    fresh_model = FakeChatModel(responses=[])
    runtime = AgentRuntime.from_settings(
        agent_settings(), hooks=AgentHooks(decide_route=vague, clarify=clarify)
    )
    result = _ask("it?", backend, fresh_model, runtime=runtime, session_id="s1", request_id="r1")
    assert result.status == AnswerStatus.NEED_CLARIFICATION
    assert result.answer == "Which library do you mean?"
    assert result.resume_token == result.thread_id and result.thread_id.startswith("s1:")
    assert backend.calls == [] and fresh_model.calls == []
    # A clarifying question is not an answer: it is not stored as a turn.
    assert asyncio.run(runtime.store.load(runtime.session_key("local", KB, "s1"))) == []


def test_sqlite_checkpointer_persists_state_per_thread(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = agent_settings(checkpoint_url=f"sqlite://{(tmp_path / 'cp.db').as_posix()}")
    backend = ScriptedSearchBackend(
        KB, by_question={"q1": pages_result(KB, [1]), "q2": pages_result(KB, [2])}
    )
    model = StreamingFakeChatModel(default_response="Answer [1].")
    with caplog.at_level(logging.WARNING):
        r1 = _ask("q1", backend, model, settings=settings, session_id="s1", request_id="r1")
        r2 = _ask("q2", backend, model, settings=settings, session_id="s1", request_id="r2")
        # A new runtime object (a restarted process) reads both threads from the file.
        runtime = AgentRuntime.from_settings(settings)
        s1 = asyncio.run(runtime.state_values(r1.thread_id))
        s2 = asyncio.run(runtime.state_values(r2.thread_id))
        missing = asyncio.run(runtime.state_values("s1:nope"))
    assert [c.chunk_id for r in s1["rounds"] for c in r.chunks] == ["hf_docs:p1.md#0"]
    assert [c.chunk_id for r in s2["rounds"] for c in r.chunks] == ["hf_docs:p2.md#0"]
    assert type(s1["rounds"][0].chunks[0]).__name__ == "Chunk"
    assert missing == {}
    # Our own types are on the serializer's allow-list: no "unregistered type" warnings.
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]
    assert (tmp_path / "cp.db").exists()


def test_default_model_is_not_built_when_no_call_is_needed() -> None:
    # No gateway configured (no LLM_BASE_URL): a greeting and a not-found search still answer,
    # because the chat model is only made when there is a fragment to answer from.
    settings = agent_settings(chat_backend="gateway", llm_base_url="", llm_api_key="")
    runtime = AgentRuntime.from_settings(settings)
    for text, result_pages in (
        ("hello", None),
        ("What is X?", pages_result(KB, [1], not_found=True)),
    ):
        backend = ScriptedSearchBackend(KB, default=result_pages)
        result = asyncio.run(
            ask_docs(
                text,
                cfg=kb_config(KB),
                retriever=backend,
                settings=settings,
                runtime=runtime,
            )
        )
        assert result.sources == [] and result.error_categories == []
    assert runtime._chat is None


def test_route_hook_may_be_async() -> None:
    # P3-02 puts a small model into the route decision: the node awaits an awaitable result.
    async def decide(question: str, *, kb: str, history: Any) -> RouteDecision:
        await asyncio.sleep(0)
        return RouteDecision(retrieve=False, kb=kb, reason="async")

    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings(), hooks=AgentHooks(decide_route=decide))
    result = _ask("How do I log in?", backend, FakeChatModel(responses=[]), runtime=runtime)
    assert result.rounds == 0 and backend.calls == []


def test_interrupt_in_a_hook_returns_need_clarification() -> None:
    # The design's clarify node stops the graph with `interrupt` (P3-02). Such a run ends
    # without a `result` in the state: the runtime must still answer need_clarification.
    from langgraph.types import interrupt

    def vague(question: str, *, kb: str, history: Any) -> RouteDecision:
        return RouteDecision(retrieve=True, kb=kb, needs_clarification=True)

    async def clarify(state: Any, ctx: Any) -> str | None:
        return str(interrupt("Which library do you mean?"))

    backend = ScriptedSearchBackend(KB)
    model = FakeChatModel(responses=[])
    runtime = AgentRuntime.from_settings(
        agent_settings(), hooks=AgentHooks(decide_route=vague, clarify=clarify)
    )
    result = _ask("it?", backend, model, runtime=runtime, session_id="s1", request_id="r1")
    assert result.status == AnswerStatus.NEED_CLARIFICATION
    assert result.answer == "Which library do you mean?"
    assert result.resume_token == result.thread_id and result.thread_id.startswith("s1:")
    assert result.cache_allowed is False
    assert backend.calls == [] and model.calls == []
    assert asyncio.run(runtime.store.load(runtime.session_key("local", KB, "s1"))) == []
