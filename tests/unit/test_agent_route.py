"""Spec P3-02 AC11: the route step asks the small model only for what the rules cannot decide."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from rag.agent import AgentHooks, AgentRuntime, ask_docs
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.testing.agent import (
    RaisingChatModel,
    ScriptedSearchBackend,
    agent_settings,
    kb_config,
    structured_reply,
)
from rag.types import AskResult

KB = "hf_docs"
ENABLED: dict[str, Any] = {"clarify": True, "route_model": True}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def judgement(*, clarify: bool = False, question: str = "", two_hop: bool = False) -> Any:
    return structured_reply(
        "RouteJudgement",
        needs_clarification=clarify,
        clarifying_question=question,
        two_hop=two_hop,
    )


def _ask(
    question: str,
    backend: ScriptedSearchBackend,
    helper: Any,
    *,
    agent: dict[str, Any] | None = None,
    hooks: AgentHooks | None = None,
    **kwargs: Any,
) -> AskResult:
    settings = agent_settings()
    runtime = AgentRuntime.from_settings(settings, hooks=hooks) if hooks else None
    return asyncio.run(
        ask_docs(
            question,
            cfg=kb_config(KB, **(ENABLED if agent is None else agent)),
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=StreamingFakeChatModel(default_response="Use it [1]."),
            helper_model=helper,
            **kwargs,
        )
    )


def test_route_rules_decide_without_a_model() -> None:
    # Clear: it has topic words (rewrite is off here, so nothing else calls a model).
    helper = FakeChatModel(responses=[])
    backend = ScriptedSearchBackend(KB)
    result = _ask("How do I push a model to the hub?", backend, helper)
    assert result.status == "answered" and len(backend.calls) == 1

    # Vague: nothing to search for. The rules ask the caller back with the fixed question.
    backend = ScriptedSearchBackend(KB)
    result = _ask("How do I fix this error?", backend, helper)
    assert result.status == "need_clarification"
    assert "Could you be more specific?" in result.answer and result.resume_token
    assert backend.calls == [] and result.rounds == 0
    zh = _ask("怎么办", backend, helper)
    assert zh.status == "need_clarification" and "具体" in zh.answer

    # A follow-up with history is clear: the rewrite resolves it, the caller is not asked.
    assert helper.calls == []


def test_route_asks_the_model_only_when_undecided() -> None:
    question = "How do I save this model to the hub?"  # points at "this model": undecided
    helper = FakeChatModel(
        responses=[judgement(clarify=True, question="Which model do you want to save?")]
    )
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert result.status == "need_clarification"
    assert result.answer == "Which model do you want to save?"  # the model's own question
    assert backend.calls == []
    (call,) = helper.calls
    prompt = "\n".join(str(m.content) for m in call)
    assert question in prompt and "needs_clarification and clarifying_question" in prompt
    assert "two_hop" not in prompt  # it was not asked about splitting: rewrite is off

    # The model says the question is fine: it is searched, and the model was asked once.
    helper = FakeChatModel(responses=[judgement(clarify=False)])
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert result.status == "answered" and len(backend.calls) == 1 and len(helper.calls) == 1

    # An empty question from the model falls back to the fixed one.
    helper = FakeChatModel(responses=[judgement(clarify=True, question="  ")])
    result = _ask(question, ScriptedSearchBackend(KB), helper)
    assert "Could you be more specific?" in result.answer

    # `route_model` off: the rules' "undecided" is searched as it is, no model.
    helper = FakeChatModel(responses=[])
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper, agent={"clarify": True, "route_model": False})
    assert result.status == "answered" and helper.calls == [] and len(backend.calls) == 1

    # `clarify` off: a vague question is searched too (nothing to ask with).
    backend = ScriptedSearchBackend(KB)
    result = _ask("How do I fix this error?", backend, helper, agent={"clarify": False})
    assert result.status == "answered" and len(backend.calls) == 1 and helper.calls == []


def test_route_decides_two_hop_with_the_model_when_rules_say_maybe() -> None:
    question = "How do I fine-tune a model and push the checkpoint to the hub?"  # "maybe"
    agent = {"rewrite": True, "route_model": True, "max_sub_questions": 2}
    helper = FakeChatModel(
        responses=[
            judgement(two_hop=True),
            structured_reply(
                "RewritePlan",
                standalone_question=question,
                search_queries=["Fine-tune a model with the Trainer", "Push the checkpoint"],
            ),
        ]
    )
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper, agent=agent)
    assert [c["question"] for c in backend.calls] == [
        "Fine-tune a model with the Trainer",
        "Push the checkpoint",
    ]
    route_prompt = "\n".join(str(m.content) for m in helper.calls[0])
    assert "two_hop" in route_prompt and "needs_clarification" not in route_prompt
    assert "at most 2" in "\n".join(str(m.content) for m in helper.calls[1])
    assert result.fallbacks == []

    # The model says it is one lookup: the rewrite is told to return exactly one query.
    helper = FakeChatModel(
        responses=[
            judgement(two_hop=False),
            structured_reply(
                "RewritePlan", standalone_question=question, search_queries=["Fine-tune and push"]
            ),
        ]
    )
    backend = ScriptedSearchBackend(KB)
    _ask(question, backend, helper, agent=agent)
    assert "Return exactly one search query" in "\n".join(str(m.content) for m in helper.calls[1])
    assert len(backend.calls) == 1

    # Rules say "yes" (a comparison): no route call at all, the rewrite splits directly.
    helper = FakeChatModel(
        responses=[
            structured_reply(
                "RewritePlan",
                standalone_question="What is the difference between A and B?",
                search_queries=["A difference", "B difference"],
            )
        ]
    )
    backend = ScriptedSearchBackend(KB)
    _ask("What is the difference between A and B?", backend, helper, agent=agent)
    assert len(helper.calls) == 1 and len(backend.calls) == 2


@pytest.mark.parametrize(
    ("helper", "reason"),
    [
        (RaisingChatModel(error=httpx.ConnectError("down")), "model_error"),
        (FakeChatModel(responses=["not a tool call"]), "parse_failed"),
    ],
)
def test_route_model_failure_keeps_the_rules_decision(helper: Any, reason: str) -> None:
    question = "How do I save this model to the hub?"
    backend = ScriptedSearchBackend(KB)
    with memory_observability() as obs:
        result = _ask(question, backend, helper)
        route = next(s for s in obs.span_exporter.get_finished_spans() if s.name == "route")
    # Undecided stays undecided: nothing is asked back, the question is searched, and the
    # result says that the small model could not be used.
    assert result.status == "answered" and [c["question"] for c in backend.calls] == [question]
    assert result.fallbacks == [f"route:{reason}"]
    assert result.degraded and "degraded" in result.error_categories
    assert route.attributes[attr.ROUTE_FALLBACK] == reason
    assert route.attributes[attr.ROUTE_CLARITY] == "undecided"


def test_a_hook_decision_is_honoured_and_the_rules_are_a_default() -> None:
    # A route hook that decides for itself (P3-01 behaviour) is not second-guessed.
    from rag.agent import RouteDecision

    def decide(question: str, *, kb: str, history: Any) -> RouteDecision:
        return RouteDecision(retrieve=True, kb=kb, needs_clarification=True)

    backend = ScriptedSearchBackend(KB)
    result = _ask(
        "How do I push a model to the hub?",
        backend,
        FakeChatModel(responses=[]),
        agent={},
        hooks=AgentHooks(decide_route=decide),
    )
    assert result.status == "need_clarification" and backend.calls == []


def test_the_question_asked_back_by_the_model_is_capped() -> None:
    # It goes to the caller, the checkpoint and the history: never a page of text.
    long_question = "Which model do you mean? " * 100
    helper = FakeChatModel(responses=[judgement(clarify=True, question=long_question)])
    result = _ask("How do I save this model to the hub?", ScriptedSearchBackend(KB), helper)
    assert result.status == "need_clarification"
    assert 0 < len(result.answer) <= 300 and result.answer.startswith("Which model do you mean?")
