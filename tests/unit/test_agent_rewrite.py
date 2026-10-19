# ruff: noqa: RUF001 (Chinese questions use full-width punctuation)
"""Spec P3-02 AC3-AC10, AC20: the rewrite step of the ask_docs graph.

Scripted small models (`FakeChatModel` with tool-call replies) and fake retrieval only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.messages import BaseMessage

from rag.agent import AgentRuntime, QueryRewriter, RewriteOutcome, ask_docs
from rag.observe import attributes as attr
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.testing.agent import (
    RaisingChatModel,
    ScriptedSearchBackend,
    SlowChatModel,
    agent_settings,
    kb_config,
    pages_result,
    structured_reply,
)
from rag.types import AskResult

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def plan(standalone: str, *queries: str, usage: tuple[int, int] | None = None) -> Any:
    return structured_reply(
        "RewritePlan", usage=usage, standalone_question=standalone, search_queries=list(queries)
    )


def _ask(
    question: str,
    backend: ScriptedSearchBackend,
    helper: Any,
    *,
    answer_model: Any = None,
    runtime: AgentRuntime | None = None,
    settings: Any = None,
    agent: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AskResult:
    settings = settings or agent_settings()
    return asyncio.run(
        ask_docs(
            question,
            cfg=kb_config(KB, **{"rewrite": True, **(agent or {})}),
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=answer_model or StreamingFakeChatModel(default_response="Use it [1]."),
            helper_model=helper,
            **kwargs,
        )
    )


def _text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def test_rule_only_questions_make_no_model_call() -> None:
    # A statement-form query with no history: normalised, searched as it is. The helper raises
    # on any call (its script is empty), so a model call would fail the test.
    helper = FakeChatModel(responses=[])
    backend = ScriptedSearchBackend(KB)
    result = _ask("  push_to_hub   parameters ", backend, helper)
    assert [c["question"] for c in backend.calls] == ["push_to_hub parameters"]
    assert result.search_queries == ["push_to_hub parameters"] and result.fallbacks == []
    assert result.found and not result.degraded and helper.calls == []

    # A question, but the KB turned the rewrite (or the declarative form) off.
    for agent, question in (
        ({"rewrite": False}, "How do I push a model?"),
        ({"declarative": False}, "How do I push a model?"),
    ):
        backend = ScriptedSearchBackend(KB)
        result = _ask(question, backend, helper, agent=agent)
        assert [c["question"] for c in backend.calls] == [question]
        assert result.fallbacks == []

    # A greeting never searches or rewrites.
    backend = ScriptedSearchBackend(KB)
    assert _ask("hello", backend, helper).rounds == 0 and backend.calls == []
    assert helper.calls == []


def test_follow_up_becomes_standalone_with_history() -> None:
    backend = ScriptedSearchBackend(KB)
    runtime = AgentRuntime.from_settings(agent_settings())
    answers = StreamingFakeChatModel(
        responses=["BERT tokenizers split text into word pieces [1].", "It pads to max length [1]."]
    )
    # Turn one is a statement: no rewrite, no model; it only fills the session's history.
    helper = FakeChatModel(
        responses=[
            plan(
                "How does the BERT tokenizer pad its output?",
                "The BERT tokenizer pads its output to the maximum length",
            )
        ]
    )
    first = _ask("bert tokenizer usage", backend, helper, answer_model=answers, runtime=runtime,
                 session_id="s1")  # fmt: skip
    assert helper.calls == [] and first.used_history is False

    result = _ask("and how does it pad?", backend, helper, answer_model=answers, runtime=runtime,
                  session_id="s1")  # fmt: skip
    assert result.used_history and result.cache_allowed is False
    (call,) = helper.calls
    prompt = _text(call)
    assert "bert tokenizer usage" in prompt  # the earlier question
    assert "BERT tokenizers split text into word pieces" in prompt  # and its answer
    assert "and how does it pad?" in prompt  # the follow-up itself
    assert "Return exactly one search query" in prompt
    # Retrieval got the statement; the answer was written for the standalone question.
    assert (
        backend.calls[-1]["question"] == "The BERT tokenizer pads its output to the maximum length"
    )
    assert result.standalone_question == "How does the BERT tokenizer pad its output?"
    assert "How does the BERT tokenizer pad its output?" in _text(answers.calls[-1])
    assert "and how does it pad?" not in _text(answers.calls[-1])
    assert result.fallbacks == [] and not result.degraded


def test_declarative_query_is_retrieved_and_the_question_is_answered() -> None:
    question = "How do I push a model to the hub?"
    statement = "Push a model to the hub with push_to_hub"
    helper = FakeChatModel(responses=[plan(question, statement)])
    answers = StreamingFakeChatModel(default_response="Call push_to_hub [1].")
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper, answer_model=answers)

    assert [c["question"] for c in backend.calls] == [statement]
    prompt = _text(answers.calls[0])
    assert question in prompt and statement not in prompt  # the answer sees the question
    assert result.search_queries == [statement]
    assert "declarative statement" in _text(helper.calls[0])
    assert "No earlier conversation." in _text(helper.calls[0])

    # With `declarative` off the model is not asked to write a statement (and, with no history
    # and no split, is not called at all for this question).
    helper = FakeChatModel(responses=[])
    backend = ScriptedSearchBackend(KB)
    _ask(question, backend, helper, agent={"declarative": False})
    assert helper.calls == [] and backend.calls[0]["question"] == question


def test_two_hop_question_is_split_and_capped() -> None:
    question = "What is the difference between Trainer and SFTTrainer?"
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "Trainer is the base training loop": pages_result(KB, [1]),
            "SFTTrainer adds supervised fine-tuning": pages_result(KB, [2]),
            "A third thing": pages_result(KB, [3]),
        },
    )
    helper = FakeChatModel(
        responses=[
            plan(
                question,
                "Trainer is the base training loop",
                "SFTTrainer adds supervised fine-tuning",
                "A third thing",
                "A fourth thing",
                "A fifth thing",
            )
        ]
    )
    result = _ask(question, backend, helper, agent={"max_sub_questions": 3})
    assert "at most 3" in _text(helper.calls[0])
    # Five queries came back; three are searched, one search each, in order.
    assert [c["question"] for c in backend.calls] == [
        "Trainer is the base training loop",
        "SFTTrainer adds supervised fine-tuning",
        "A third thing",
    ]
    assert len(result.search_queries) == 3 and result.rounds == 1
    assert sorted(c.chunk_id for c in result.sources) == [
        "hf_docs:p1.md#0",
        "hf_docs:p2.md#0",
        "hf_docs:p3.md#0",
    ]

    # `max_sub_questions: 1` never splits: the model is asked for exactly one query.
    helper = FakeChatModel(responses=[plan(question, "Trainer and SFTTrainer differ", "extra")])
    backend = ScriptedSearchBackend(KB)
    _ask(question, backend, helper, agent={"max_sub_questions": 1})
    assert "Return exactly one search query" in _text(helper.calls[0])
    assert [c["question"] for c in backend.calls] == ["Trainer and SFTTrainer differ"]


def test_lost_identifiers_fall_back_to_the_original_query() -> None:
    question = "How do I use push_to_hub in 4.36.0 with `device_map=auto`?"
    kept = "Use push_to_hub in 4.36.0 with device_map=auto"
    helper = FakeChatModel(responses=[plan(question, kept)])
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert [c["question"] for c in backend.calls] == [kept] and result.fallbacks == []
    # The model is told what to keep.
    assert "push_to_hub | 4.36.0 | device_map=auto" in _text(helper.calls[0])

    # Each kind of loss is caught: the version, the code token, the case of an identifier.
    for lost in (
        "Use push_to_hub with device_map=auto",
        "Use push_to_hub in 4.36.0 with a device map",
        "Use push_to_Hub in 4.36.0 with device_map=auto",
    ):
        helper = FakeChatModel(responses=[plan(question, lost)])
        backend = ScriptedSearchBackend(KB)
        answers = StreamingFakeChatModel(default_response="Yes [1].")
        result = _ask(question, backend, helper, answer_model=answers)
        assert [c["question"] for c in backend.calls] == [question], lost
        assert result.fallbacks == ["rewrite:identifiers_lost"]
        assert result.degraded and "degraded" in result.error_categories
        assert result.found and result.cache_allowed is False
        assert question in _text(answers.calls[0])  # the answer got the original question

    # A split question may spread the identifiers over its queries.
    question = "What is the difference between Trainer.train and push_to_hub in 4.36.0?"
    helper = FakeChatModel(
        responses=[plan(question, "Trainer.train runs training", "push_to_hub in 4.36.0 uploads")]
    )
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert result.fallbacks == [] and len(backend.calls) == 2

    # A standalone question that lost an identifier is not used (the answer gets the original),
    # the queries are fine and stay.
    question = "How do I use push_to_hub?"
    helper = FakeChatModel(responses=[plan("How do I upload a model?", "Use push_to_hub")])
    answers = StreamingFakeChatModel(default_response="Yes [1].")
    result = _ask(question, ScriptedSearchBackend(KB), helper, answer_model=answers)
    assert result.standalone_question is None and question in _text(answers.calls[0])
    assert result.search_queries == ["Use push_to_hub"] and result.fallbacks == []


def test_rewrite_keeps_the_questions_language() -> None:
    question = "怎么用 push_to_hub 上传模型到仓库？"
    helper = FakeChatModel(
        responses=[plan(question, "使用 push_to_hub 把模型上传到仓库")]  # same language: used
    )
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert backend.calls[0]["question"] == "使用 push_to_hub 把模型上传到仓库"
    assert result.fallbacks == []
    assert "Keep the language" in _text(helper.calls[0])

    # Translated to English (the identifier survives): refused, the question is searched.
    helper = FakeChatModel(responses=[plan(question, "Upload a model to the hub with push_to_hub")])
    backend = ScriptedSearchBackend(KB)
    result = _ask(question, backend, helper)
    assert [c["question"] for c in backend.calls] == ["怎么用 push_to_hub 上传模型到仓库?"]
    assert result.fallbacks == ["rewrite:language_changed"]

    # And the other way round.
    helper = FakeChatModel(responses=[plan("How do I push a model?", "如何上传模型")])
    result = _ask("How do I push a model?", ScriptedSearchBackend(KB), helper)
    assert result.fallbacks == ["rewrite:language_changed"]


@pytest.mark.parametrize(
    ("helper", "settings", "reason"),
    [
        (RaisingChatModel(error=httpx.ConnectError("gateway down")), {}, "model_error"),
        (
            SlowChatModel(delay_s=5.0, responses=[plan("q?", "q statement")]),
            {"agent_rewrite_timeout_ms": 50},
            "timeout",
        ),
        (FakeChatModel(responses=["this is not a tool call"]), {}, "parse_failed"),
        (
            FakeChatModel(responses=[structured_reply("RewritePlan", standalone_question="x")]),
            {},
            "parse_failed",
        ),  # a required field is missing
        (FakeChatModel(responses=[plan("q?", "   ", "")]), {}, "empty"),
    ],
    ids=["gateway_error", "timeout", "garbage", "missing_field", "empty_queries"],
)
def test_model_failures_fall_back_and_flag_degraded(
    helper: Any, settings: dict[str, Any], reason: str
) -> None:
    question = "How do I push a model to the hub?"
    backend = ScriptedSearchBackend(KB)
    answers = StreamingFakeChatModel(default_response="Call push_to_hub [1].")
    result = _ask(
        question, backend, helper, answer_model=answers, settings=agent_settings(**settings)
    )
    # The request is answered, from the original question, and says that it fell back.
    assert [c["question"] for c in backend.calls] == [question]
    assert result.status == "answered" and result.found
    assert result.fallbacks == [f"rewrite:{reason}"]
    assert result.degraded and "degraded" in result.error_categories
    assert result.cache_allowed is False
    assert question in _text(answers.calls[0])


def test_no_model_available_falls_back_too() -> None:
    # The factory of the helper model fails (no gateway configured): a fallback, not an error.
    def no_gateway() -> Any:
        raise ValueError("LLM_BASE_URL is not set")

    backend = ScriptedSearchBackend(KB)
    result = _ask("How do I push a model?", backend, no_gateway)
    assert result.fallbacks == ["rewrite:unavailable"] and result.found
    assert [c["question"] for c in backend.calls] == ["How do I push a model?"]


def test_token_cap_before_rewrite_falls_back_and_still_answers() -> None:
    # Cap 7,000 with 6,000 kept for the answer: one small call fits, a second does not. The
    # route's model call (a question that points at "this model") uses 1,100 tokens, so the
    # rewrite is refused before it is sent, and the answer is still written.
    settings = agent_settings(agent_token_cap=7_000, agent_answer_reserve_tokens=6_000)
    helper = FakeChatModel(
        responses=[
            structured_reply(
                "RouteJudgement",
                usage=(900, 200),
                needs_clarification=False,
                clarifying_question="",
                two_hop=False,
            )
        ]
    )
    backend = ScriptedSearchBackend(KB)
    question = "How do I save this model to the hub?"
    result = _ask(
        question,
        backend,
        helper,
        settings=settings,
        agent={"clarify": True, "route_model": True},
    )
    assert len(helper.calls) == 1  # the rewrite call never left
    assert [c["question"] for c in backend.calls] == [question]
    assert result.fallbacks == ["rewrite:token_cap"]
    assert result.stop_reason == "token_cap" and "token_cap_reached" in result.error_categories
    assert result.tokens_used >= 1_100
    assert result.status == "answered" and result.found  # the answer's tokens were kept back


def test_next_round_signature_takes_what_was_missing() -> None:
    # Round one is built; round two is P3-03's. The hook already hands it what was missing.
    seen: list[tuple[int, list[str]]] = []

    class Next(QueryRewriter):
        async def next_round(self, state: Any, ctx: Any, *, missing: Any) -> RewriteOutcome:
            seen.append((state["round"], list(missing)))
            return RewriteOutcome(sub_questions=("another wording",))

    state: Any = {"question": "q", "round": 1, "missing": ["part b"]}
    ctx = SimpleNamespace()
    assert asyncio.run(Next()(state, ctx)).sub_questions == ("another wording",)
    assert seen == [(1, ["part b"])]
    # The default keeps the queries of the last round.
    assert asyncio.run(QueryRewriter()(state, ctx)) == RewriteOutcome()


def test_spans_and_record_carry_the_rewrite() -> None:
    question = "How do I push a model to the hub?"
    statement = "Push a model to the hub with push_to_hub"
    helper = FakeChatModel(responses=[plan(question, statement, usage=(800, 150))])
    with memory_observability() as obs:
        _ask(question, ScriptedSearchBackend(KB), helper)
        spans = {s.name: s for s in obs.span_exporter.get_finished_spans()}
    rewrite = spans["rewrite"].attributes
    assert rewrite[attr.REWRITE_MODEL_USED] is True
    assert list(rewrite[attr.REWRITE_QUERIES]) == [statement]
    assert rewrite[attr.REWRITE_STANDALONE] == question
    assert rewrite[attr.NODE_TOKENS] == 950 and rewrite[attr.AGENT_TOKENS] == 950
    assert attr.REWRITE_FALLBACK not in rewrite
    route = spans["route"].attributes
    assert route[attr.ROUTE_CLARITY] == "clear" and route[attr.ROUTE_MULTI_HOP] == "no"

    failing = RaisingChatModel(error=httpx.ConnectError("down"))
    with memory_observability() as obs:
        _ask(question, ScriptedSearchBackend(KB), failing)
        spans = {s.name: s for s in obs.span_exporter.get_finished_spans()}
    assert spans["rewrite"].attributes[attr.REWRITE_FALLBACK] == "model_error"
    assert list(spans["rewrite"].attributes[attr.REWRITE_QUERIES]) == [question]


def test_model_calls_and_cost_of_the_step_are_on_its_span(tmp_path: Any) -> None:
    # A chat model from the real factory over a fake gateway: its calls reach the cost ledger,
    # and the rewrite span reports the lines written during that step (calls, CNY, tokens).
    import json

    from rag.cost import CostLedger
    from rag.llm import make_chat_model

    question = "How do I push a model to the hub?"
    arguments = {"standalone_question": question, "search_queries": ["Push a model to the hub"]}
    usage = {
        "prompt_tokens": 800,
        "completion_tokens": 150,
        "total_tokens": 950,
        "completion_tokens_details": {"reasoning_tokens": 60},
    }

    def gateway(request: httpx.Request) -> httpx.Response:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "RewritePlan", "arguments": json.dumps(arguments)},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4.1-flash",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": usage,
            },
        )

    settings = agent_settings(
        llm_base_url="http://gateway.test/v1",
        llm_api_key="sk-unit-test-key-0123456789",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    helper = make_chat_model("cheap", settings, async_transport=httpx.MockTransport(gateway))
    with memory_observability() as obs:
        result = _ask(question, ScriptedSearchBackend(KB), helper, settings=settings)
        spans = {s.name: s for s in obs.span_exporter.get_finished_spans()}
    assert result.search_queries == ["Push a model to the hub"] and result.fallbacks == []
    rewrite = spans["rewrite"].attributes
    assert rewrite[attr.NODE_MODEL_CALLS] == 1 and rewrite[attr.NODE_TOKENS] == 950
    assert rewrite[attr.NODE_REASONING_TOKENS] == 60
    assert rewrite[attr.GEN_AI_USAGE_INPUT_TOKENS] == 800
    assert rewrite[attr.GEN_AI_USAGE_OUTPUT_TOKENS] == 150
    (line,) = CostLedger.from_settings(settings).entries()
    assert line["profile"] == "cheap" and line["reasoning_tokens"] == 60
    assert rewrite[attr.NODE_COST_CNY] == pytest.approx(line["cny"])
