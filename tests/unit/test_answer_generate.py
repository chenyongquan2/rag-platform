# ruff: noqa: RUF001 (Chinese sample text)
"""Spec P1-02: streaming answer generation, retry, failures, spans and cost (design s11-1, s16-8).

Models are `FakeChatModel`s, and once the real factory with the gateway replaced by
`httpx.MockTransport`; nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

import rag.answer.generate as generate_module
from rag.answer import (
    AnswerContext,
    AnswerDelta,
    AnswerDone,
    AnswerRetry,
    build_context,
    generate_answer,
    stream_answer,
)
from rag.answer.prompts import NOT_FOUND
from rag.config import AnswerConfig, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.llm import make_chat_model
from rag.observe import attributes as attr
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.types import AnswerStatus, Chunk, ErrorCategory

QUESTION = "What is the default local attention window of LongT5?"
CONFIG = AnswerConfig()


def make_context(n: int = 3, *, suspicious: int | None = None) -> AnswerContext:
    chunks = [
        Chunk(
            chunk_id=f"longt5#{i}",
            doc_id="longt5",
            kb="hf_docs",
            text=f"Fragment {i} about LongT5 local attention.",
            title_path=["LongT5", f"Part {i}"],
            source_rel="docs/longt5.md",
            suspicious_instruction=(i == suspicious),
        )
        for i in range(n)
    ]
    return build_context([chunks], budget_tokens=CONFIG.context_tokens)


async def collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]


def events_of(model: BaseChatModel, ctx: AnswerContext, **kwargs: Any) -> list[Any]:
    kwargs.setdefault("config", CONFIG)
    stream = stream_answer(QUESTION, ctx, kb="hf_docs", model=model, request_id="req-1", **kwargs)
    return asyncio.run(collect(stream))


def final(events: list[Any]):  # type: ignore[no-untyped-def]
    assert isinstance(events[-1], AnswerDone)
    assert not any(isinstance(e, AnswerDone) for e in events[:-1])
    return events[-1].answer


def test_stream_yields_deltas_and_final_answer() -> None:
    reply = "LongT5 attends to 127 tokens on each side [2]. Local attention adds no parameters [2]."
    ctx = make_context()
    events = events_of(StreamingFakeChatModel(responses=[reply], chunk_chars=7), ctx)

    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    assert len(deltas) > 5 and all(d.attempt == 1 for d in deltas)
    assert "".join(d.text for d in deltas) == reply
    answer = final(events)
    assert answer.found and answer.status is AnswerStatus.ANSWERED
    assert answer.request_id == "req-1" and answer.kb == "hf_docs"
    assert [s.text for s in answer.sentences] == [
        "LongT5 attends to 127 tokens on each side.",
        "Local attention adds no parameters.",
    ]
    assert [[c.chunk_id for c in s.citations] for s in answer.sentences] == [["longt5#1"]] * 2
    # Marker n is chunks[n - 1]: the answer carries the numbered fragments in order.
    assert answer.chunks == ctx.chunks
    assert answer.citation_check.model_dump() == {
        "removed_markers": [],
        "unsupported_sentences": 0,
        "parse_attempts": 1,
    }
    assert answer.error_categories == [] and answer.generation_error is None
    assert "generate" in answer.timings_ms

    # The non-streaming call is the same stream, consumed to the end.
    same = asyncio.run(
        generate_answer(
            QUESTION,
            ctx,
            kb="hf_docs",
            config=CONFIG,
            model=FakeChatModel(responses=[reply]),
            request_id="req-1",
        )
    )
    assert same.model_dump(exclude={"timings_ms"}) == answer.model_dump(exclude={"timings_ms"})
    assert json.loads(same.model_dump_json())["citation_check"]["parse_attempts"] == 1


def test_default_model_is_online_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    def fake_factory(profile: str, settings: ServiceSettings | None = None) -> BaseChatModel:
        built.append(profile)
        return FakeChatModel(responses=["LongT5 uses local attention [1]."])

    monkeypatch.setattr(generate_module, "make_chat_model", fake_factory)
    answer = asyncio.run(generate_answer(QUESTION, make_context(), kb="hf_docs", config=CONFIG))
    assert built == ["online"]
    assert answer.found


@pytest.mark.parametrize(
    ("question", "reply", "language"),
    [
        (QUESTION, f"{NOT_FOUND['en']}  ", "en"),
        (QUESTION, f"**{NOT_FOUND['en'].rstrip('.')}**", "en"),
        ("LongT5 的默认窗口是多少？", NOT_FOUND["zh"], "zh"),
        ("LongT5 的默认窗口是多少？", NOT_FOUND["en"], "zh"),
    ],
)
def test_not_found_sentence(question: str, reply: str, language: str) -> None:
    model = FakeChatModel(responses=[reply])
    answer = asyncio.run(
        generate_answer(question, make_context(), kb="hf_docs", config=CONFIG, model=model)
    )
    assert not answer.found and answer.status is AnswerStatus.NOT_FOUND
    assert answer.text == NOT_FOUND[language]
    assert answer.sentences[0].citations == []
    assert not answer.answer_parse_failed and answer.error_categories == []
    assert answer.citation_check.unsupported_sentences == 0
    assert len(model.calls) == 1  # no retry


def test_empty_context_skips_model() -> None:
    model = FakeChatModel(responses=[])  # any call would fail the test
    empty = build_context([[]], budget_tokens=4000)
    too_small = build_context([make_context().chunks], budget_tokens=1)
    for ctx in (empty, too_small):
        answer = asyncio.run(
            generate_answer(QUESTION, ctx, kb="hf_docs", config=CONFIG, model=model)
        )
        assert not answer.found and answer.status is AnswerStatus.NOT_FOUND
        assert answer.text == NOT_FOUND["en"]
        assert answer.citation_check.parse_attempts == 0
    assert model.calls == []


def test_parse_retry_recovers() -> None:
    model = StreamingFakeChatModel(
        responses=["LongT5 uses local attention.", "LongT5 uses local attention [1]."],
        chunk_chars=10,
    )
    events = events_of(model, make_context())

    retries = [e for e in events if isinstance(e, AnswerRetry)]
    assert retries == [AnswerRetry(attempt=2, reason=ErrorCategory.ANSWER_PARSE_FAILED)]
    index = events.index(retries[0])
    assert {e.attempt for e in events[:index] if isinstance(e, AnswerDelta)} == {1}
    assert {e.attempt for e in events[index:] if isinstance(e, AnswerDelta)} == {2}
    answer = final(events)
    assert answer.found and not answer.answer_parse_failed
    assert answer.citation_check.parse_attempts == 2

    first, second = model.calls
    assert second[:2] == first
    assert isinstance(second[2], AIMessage) and second[2].content == "LongT5 uses local attention."
    assert isinstance(second[3], HumanMessage) and "[n]" in str(second[3].content)


def test_parse_failure_retries_once_then_flags() -> None:
    ctx = make_context()
    model = FakeChatModel(responses=["No markers at all.", "Still none [9]."])
    answer = asyncio.run(generate_answer(QUESTION, ctx, kb="hf_docs", config=CONFIG, model=model))

    assert len(model.calls) == 2
    assert answer.answer_parse_failed and not answer.found
    assert answer.status is AnswerStatus.NOT_FOUND
    assert answer.sentences == [] and answer.text == ""
    assert answer.chunks == ctx.chunks  # the retrieved fragments only
    assert answer.error_categories == [ErrorCategory.ANSWER_PARSE_FAILED]
    assert answer.citation_check.model_dump() == {
        "removed_markers": [9],
        "unsupported_sentences": 1,
        "parse_attempts": 2,
    }

    # parse_retries is the config's: 0 means one attempt only.
    once = FakeChatModel(responses=["No markers at all."])
    no_retry = AnswerConfig(parse_retries=0)
    answer = asyncio.run(generate_answer(QUESTION, ctx, kb="hf_docs", config=no_retry, model=once))
    assert answer.answer_parse_failed and len(once.calls) == 1


class FailingChatModel(FakeChatModel):
    """Streams `before` (if any) and then raises `error`."""

    error: BaseException
    before: str = ""

    async def _astream(  # type: ignore[override]
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.calls.append(list(messages))
        if self.before:
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.before))
        raise self.error


@pytest.mark.parametrize(
    ("error", "before", "category"),
    [
        (httpx.ReadTimeout("slow"), "", ErrorCategory.LLM_TIMEOUT),
        (httpx.ReadTimeout("slow"), "LongT5 uses", ErrorCategory.LLM_TIMEOUT),
        (httpx.ConnectError("refused"), "", ErrorCategory.LLM_ERROR),
        (BudgetExceeded("over the stop"), "", ErrorCategory.BUDGET_EXCEEDED),
    ],
)
def test_model_failure_returns_chunks_only(
    error: BaseException, before: str, category: ErrorCategory
) -> None:
    ctx = make_context()
    model = FailingChatModel(error=error, before=before)
    events = events_of(model, ctx)

    assert [e.text for e in events if isinstance(e, AnswerDelta)] == ([before] if before else [])
    answer = final(events)
    assert answer.generation_error == category
    assert answer.error_categories == [category]
    assert not answer.found and answer.sentences == [] and answer.chunks == ctx.chunks
    assert not answer.answer_parse_failed
    assert len(model.calls) == 1  # the transport already retried; no parse retry either


def test_model_bug_propagates() -> None:
    model = FailingChatModel(error=ValueError("a bug, not a gateway failure"))
    with pytest.raises(ValueError, match="a bug"):
        events_of(model, make_context())


def _span(obs: Any, name: str = "generate") -> Any:
    (span,) = [s for s in obs.span_exporter.get_finished_spans() if s.name == name]
    return span


def test_generate_span_attributes() -> None:
    ctx = make_context(4, suspicious=2)
    with memory_observability() as obs:
        answer = asyncio.run(
            generate_answer(
                QUESTION,
                ctx,
                kb="hf_docs",
                config=CONFIG,
                model=FakeChatModel(responses=["A [1]. B [7]. C."]),
            )
        )
        span = _span(obs)
    a = dict(span.attributes)
    assert a[attr.STEP] == "generate"
    assert a[attr.CONTEXT_FRAGMENTS] == 4
    assert a[attr.CONTEXT_TOKENS] == ctx.tokens
    assert a[attr.CONTEXT_BUDGET] == 4000
    assert a[attr.CONTEXT_SUSPICIOUS] == 1
    assert a[attr.CONTEXT_DROPPED] == 0 and a[attr.CONTEXT_DUPLICATES] == 0
    assert a[attr.ANSWER_LANGUAGE] == "en"
    assert a[attr.ANSWER_FOUND] is True
    assert a[attr.ANSWER_SENTENCES] == 3
    assert a[attr.ANSWER_UNSUPPORTED] == 2
    assert a[attr.ANSWER_REMOVED_MARKERS] == 1
    assert a[attr.ANSWER_ATTEMPTS] == 1
    assert a[attr.ANSWER_PARSE_FAILED] is False
    assert attr.ERROR_CATEGORY not in a
    assert a[attr.ANSWER_MODEL_CALLS] == 0  # fake model: no ledger line
    assert answer.citation_check.removed_markers == [7]

    with memory_observability() as obs:
        asyncio.run(
            generate_answer(
                QUESTION,
                make_context(),
                kb="hf_docs",
                config=CONFIG,
                model=FakeChatModel(responses=["none", "none"]),
            )
        )
        failed = dict(_span(obs).attributes)
    assert failed[attr.ANSWER_PARSE_FAILED] is True
    assert failed[attr.ANSWER_ATTEMPTS] == 2
    assert failed[attr.ERROR_CATEGORY] == ErrorCategory.ANSWER_PARSE_FAILED


# ---- the real factory with a mocked gateway ---------------------------------------------------

USAGE = {
    "prompt_tokens": 900,
    "completion_tokens": 120,
    "total_tokens": 1020,
    "completion_tokens_details": {"reasoning_tokens": 80},
}


def _sse_gateway(content: str) -> httpx.MockTransport:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        events = [chunk({"role": "assistant", "content": ""})]
        events += [chunk({"content": content[i : i + 6]}) for i in range(0, len(content), 6)]
        events += [chunk({}, "stop"), {**chunk({}), "choices": [], "usage": USAGE}]
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=payload.encode()
        )

    return httpx.MockTransport(handler)


def test_generate_cost_from_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("CHAT_BACKEND", "CHAT_ONLINE", "CHAT_MODEL", "MODEL_PRICES", "BUDGET_STOP_CNY"):
        monkeypatch.delenv(key, raising=False)
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="http://gateway.test/v1",
        llm_api_key="sk-unit-test-key-0123456789",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    ledger = CostLedger.from_settings(settings)
    transport = _sse_gateway("LongT5 uses local attention with r=127 [2].")
    model = make_chat_model("online", settings, ledger=ledger, async_transport=transport)

    with memory_observability() as obs:
        answer = asyncio.run(
            generate_answer(QUESTION, make_context(), kb="hf_docs", config=CONFIG, model=model)
        )
        span = dict(_span(obs).attributes)

    assert answer.found and answer.sentences[0].citations[0].chunk_id == "longt5#1"
    (row,) = ledger.entries()
    assert row["profile"] == "online" and row["stream"] is True
    assert span[attr.ANSWER_MODEL_CALLS] == 1
    assert span[attr.GEN_AI_USAGE_INPUT_TOKENS] == 900
    assert span[attr.GEN_AI_USAGE_OUTPUT_TOKENS] == 120
    assert span[attr.ANSWER_COST_CNY] == pytest.approx(row["cny"])
    assert span[attr.USAGE_ESTIMATED] is False
