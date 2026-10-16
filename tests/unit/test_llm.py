"""Spec P1-01: chat model factory, ledger lines, budget, structured output, error categories.

The gateway is replaced by `httpx.MockTransport`; nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from rag.config import ModelPrice, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger, UnknownModelPriceError
from rag.llm import (
    StructuredOutputError,
    chat_error_category,
    make_chat_model,
    resolve_chat_profile,
    structured_output,
)
from rag.testing import FakeChatModel
from rag.types import ErrorCategory

MODEL = "deepseek-v4-flash"
SERVED = "deepseek-v4.1-flash"
KEY = "sk-unit-test-key-0123456789"
USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 40,
    "total_tokens": 52,
    "completion_tokens_details": {"reasoning_tokens": 30},
}

_CHAT_ENV = [
    "CHAT_BACKEND",
    "CHAT_FAKE_REPLY",
    "CHAT_STRUCTURED_OUTPUT_METHOD",
    "CHAT_ONLINE",
    "CHAT_OFFLINE",
    "CHAT_CHEAP",
    "CHAT_MODEL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "MODEL_PRICES",
    "BUDGET_CNY",
    "BUDGET_STOP_CNY",
]


@pytest.fixture(autouse=True)
def _clean_env_and_no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CHAT_ENV:
        monkeypatch.delenv(key, raising=False)

    # The openai SDK waits 0.5 s and more between retries; unit tests do not wait.
    async def _no_async_sleep(self: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        openai._base_client.SyncAPIClient, "_sleep_for_retry", lambda self, **kwargs: None
    )
    monkeypatch.setattr(openai._base_client.AsyncAPIClient, "_sleep_for_retry", _no_async_sleep)


def make_settings(tmp_path: Path, **overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_base_url": "http://gateway.test/v1",
        "llm_api_key": KEY,
        "cost_ledger_path": str(tmp_path / "ledger.jsonl"),
    }
    values.update(overrides)
    return ServiceSettings(**values)


# ---- fake gateway ---------------------------------------------------------------------------


class FakeChatGateway:
    """`httpx.MockTransport` handler that answers like an OpenAI-compatible `/chat/completions`.

    `script` holds what the next requests get, one item per request: an exception to raise, an
    `httpx.Response`, a message dict to answer with, or `None` for the default reply. After the
    script every answer is the default reply. Streams send one SSE event per body chunk and the
    usage event last when the request asks for it. `on_request` runs inside every request.
    """

    def __init__(
        self,
        script: list[object] | None = None,
        *,
        usage: dict[str, Any] | None = None,
        content: str = "pong pong",
        on_request: Any = None,
    ) -> None:
        self.script = list(script or [])
        self.usage = usage or USAGE
        self.content = content
        self.on_request = on_request
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.on_request is not None:
            self.on_request(request)
        item = self.script.pop(0) if self.script else None
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, httpx.Response):
            return item
        message = item if isinstance(item, dict) else self.reply()
        if body.get("stream"):
            return self._stream(message, include_usage=bool(body.get("stream_options")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": SERVED,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": self.usage,
            },
        )

    def reply(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self.content, "reasoning_content": "thinking"}

    def _stream(self, message: dict[str, Any], *, include_usage: bool) -> httpx.Response:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": SERVED,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        content = str(message.get("content") or "")
        events = [chunk({"role": "assistant", "content": "", "reasoning_content": "think"})]
        events += [chunk({"content": content[i : i + 3]}) for i in range(0, len(content), 3)]
        events.append(chunk({}, "stop"))
        if include_usage:
            events.append({**chunk({}), "choices": [], "usage": self.usage})
        payloads = [f"data: {json.dumps(e)}\n\n".encode() for e in events] + [b"data: [DONE]\n\n"]

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_Events(payloads)
        )


class _Events(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body sent in several chunks, one SSE event each, to sync and async clients."""

    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = payloads

    def __iter__(self) -> Iterator[bytes]:
        yield from self._payloads

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for payload in self._payloads:
            yield payload


def gateway_model(
    tmp_path: Path,
    gateway: FakeChatGateway,
    profile: Literal["online", "offline", "cheap"] = "online",
    **overrides: Any,
) -> tuple[ChatOpenAI, CostLedger]:
    settings = make_settings(tmp_path, **overrides)
    ledger = CostLedger.from_settings(settings)
    transport = httpx.MockTransport(gateway)
    model = make_chat_model(
        profile, settings, ledger=ledger, transport=transport, async_transport=transport
    )
    assert isinstance(model, ChatOpenAI)
    return model, ledger


# ---- settings -------------------------------------------------------------------------------


def test_chat_profile_settings_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    s = ServiceSettings(_env_file=None)
    assert s.chat_backend == "gateway"
    assert s.chat_structured_output_method == "function_calling"
    online = resolve_chat_profile("online", s)
    offline = resolve_chat_profile("offline", s)
    cheap = resolve_chat_profile("cheap", s)
    assert online.model == offline.model == cheap.model == s.chat_model == MODEL
    assert online.max_retries == cheap.max_retries == 1
    assert offline.max_retries > online.max_retries
    assert online.timeout_s <= 30 and cheap.timeout_s <= online.timeout_s
    assert offline.timeout_s > online.timeout_s
    assert offline.retry_after_max_s > online.retry_after_max_s
    # The model is a reasoning model: max_tokens covers reasoning and answer, so it must be large.
    assert min(online.max_tokens, offline.max_tokens, cheap.max_tokens) >= 2048
    assert s.model_prices[MODEL].estimated is True
    with pytest.raises(ValueError, match="online, offline, cheap"):
        resolve_chat_profile("fast", s)  # type: ignore[arg-type]

    # A partial override keeps the other values of that profile, not the class defaults.
    monkeypatch.setenv("CHAT_ONLINE", '{"timeout_s": 7}')
    monkeypatch.setenv("CHAT_CHEAP", '{"model": "other-model", "temperature": null}')
    monkeypatch.setenv("CHAT_BACKEND", "fake")
    monkeypatch.setenv("CHAT_STRUCTURED_OUTPUT_METHOD", "json_schema")
    s = ServiceSettings(_env_file=None)
    assert s.chat_online.timeout_s == 7
    assert s.chat_online.max_retries == online.max_retries
    assert s.chat_online.max_tokens == online.max_tokens
    assert resolve_chat_profile("cheap", s).model == "other-model"
    assert s.chat_cheap.temperature is None
    assert s.chat_cheap.timeout_s == cheap.timeout_s
    assert s.chat_backend == "fake"
    assert s.chat_structured_output_method == "json_schema"

    with pytest.raises(ValidationError, match="chat_backend"):
        ServiceSettings(_env_file=None, chat_backend="openai")
    with pytest.raises(ValidationError, match="timeout_s"):
        ServiceSettings(_env_file=None, chat_online={"timeout_s": 0})
    with pytest.raises(ValidationError, match="extra"):
        ServiceSettings(_env_file=None, chat_online={"timeout": 5})


# ---- factory --------------------------------------------------------------------------------


def test_factory_builds_profile_models(tmp_path: Path) -> None:
    for profile in ("online", "offline", "cheap"):
        gateway = FakeChatGateway()
        model, _ = gateway_model(tmp_path, gateway, profile)
        p = resolve_chat_profile(profile, make_settings(tmp_path))
        assert model.model_name == MODEL
        assert model.max_tokens == p.max_tokens
        assert model.max_retries == p.max_retries
        assert model.request_timeout == p.timeout_s
        assert model.openai_api_base == "http://gateway.test/v1"

        assert model.invoke("ping").content == "pong pong"
        list(model.stream("ping"))
        plain, streamed = gateway.bodies
        assert plain["model"] == MODEL
        assert plain["max_completion_tokens"] == p.max_tokens
        assert plain["temperature"] == p.temperature
        assert "stream_options" not in plain
        assert streamed["stream"] is True
        assert streamed["stream_options"] == {"include_usage": True}
        request = gateway.requests[0]
        assert str(request.url) == "http://gateway.test/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.extensions["timeout"]["read"] == p.timeout_s


def test_factory_requires_gateway_and_price(tmp_path: Path) -> None:
    gateway = FakeChatGateway()
    transport = httpx.MockTransport(gateway)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        make_chat_model("online", make_settings(tmp_path, llm_base_url=""), transport=transport)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        make_chat_model("online", make_settings(tmp_path, llm_api_key=""), transport=transport)
    with pytest.raises(UnknownModelPriceError, match="other-model"):
        make_chat_model(
            "cheap",
            make_settings(tmp_path, chat_cheap={"model": "other-model"}),
            transport=transport,
        )
    assert gateway.requests == []
    assert not (tmp_path / "ledger.jsonl").exists()


# ---- ledger lines ---------------------------------------------------------------------------


def test_invoke_records_usage_with_reasoning_tokens(tmp_path: Path) -> None:
    gateway = FakeChatGateway()
    model, ledger = gateway_model(tmp_path, gateway)

    message = model.invoke("ping")

    assert message.content == "pong pong"
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_token_details"].get("reasoning") == 30
    (row,) = ledger.entries()
    assert row["kind"] == "chat"
    assert row["model"] == MODEL
    assert row["served_model"] == SERVED
    assert row["profile"] == "online"
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 40  # completion_tokens already includes reasoning
    assert row["reasoning_tokens"] == 30
    assert row["usage_estimated"] is False
    assert row["stream"] is False
    assert row["price_estimated"] is True
    assert row["cny"] == pytest.approx((12 * 2.0 + 40 * 8.0) / 1_000_000)

    # A gateway that leaves reasoning tokens out of completion_tokens (total shows them).
    gateway.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 45,
        "completion_tokens_details": {"reasoning_tokens": 30},
    }
    model.invoke("ping")
    row = ledger.entries()[-1]
    assert (row["completion_tokens"], row["reasoning_tokens"]) == (35, 30)

    # No usage in the answer: estimated from the request size and max_tokens.
    gateway.usage = None  # type: ignore[assignment]
    gateway.script = [
        httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": SERVED,
                "choices": [{"index": 0, "message": gateway.reply(), "finish_reason": "stop"}],
            },
        )
    ]
    model.invoke("ping")
    row = ledger.entries()[-1]
    assert row["usage_estimated"] is True
    assert row["completion_tokens"] == model.max_tokens
    assert ledger.reserved() == 0


def test_stream_records_usage_from_last_chunk(tmp_path: Path) -> None:
    gateway = FakeChatGateway()
    model, ledger = gateway_model(tmp_path, gateway)

    chunks = list(model.stream("ping"))

    assert "".join(str(c.content) for c in chunks) == "pong pong"
    assert sum(1 for c in chunks if c.content) >= 2
    (row,) = ledger.entries()
    assert row["stream"] is True
    assert row["usage_estimated"] is False
    assert (row["prompt_tokens"], row["completion_tokens"], row["reasoning_tokens"]) == (12, 40, 30)
    assert ledger.reserved() == 0


def test_stream_closed_early_records_estimate(tmp_path: Path) -> None:
    gateway = FakeChatGateway(content="a long answer that the reader stops halfway through")
    model, ledger = gateway_model(tmp_path, gateway)

    stream = model.stream("ping")
    next(stream)
    next(stream)
    stream.close()  # the user closed the page (design s16-8, streaming row)

    (row,) = ledger.entries()
    assert row["stream"] is True
    assert row["usage_estimated"] is True
    assert row["error"] == "closed before usage"
    assert row["completion_tokens"] == model.max_tokens
    assert ledger.reserved() == 0


def test_async_invoke_records_usage(tmp_path: Path) -> None:
    gateway = FakeChatGateway()
    model, ledger = gateway_model(tmp_path, gateway)

    async def run() -> list[str]:
        replies = await asyncio.gather(model.ainvoke("a"), model.ainvoke("b"))
        chunks = [str(c.content) async for c in model.astream("c")]
        return [str(r.content) for r in replies] + ["".join(chunks)]

    assert asyncio.run(run()) == ["pong pong"] * 3
    rows = ledger.entries()
    assert len(rows) == 3
    assert all(r["usage_estimated"] is False and r["completion_tokens"] == 40 for r in rows)
    assert sorted(r["stream"] for r in rows) == [False, False, True]
    assert ledger.reserved() == 0


# ---- budget ---------------------------------------------------------------------------------


def test_budget_reserved_before_send_and_refused_over_stop(tmp_path: Path) -> None:
    held: list[float] = []
    gateway = FakeChatGateway(on_request=lambda request: held.append(ledger.reserved()))
    # online max_tokens 4096 at 8 CNY per million output tokens: about 0.033 CNY per attempt.
    model, ledger = gateway_model(tmp_path, gateway, budget_cny=1.0, budget_stop_cny=0.05)

    model.invoke("ping")

    body_chars = len(gateway.requests[0].content.decode())
    assert held == [pytest.approx(ledger.estimate(MODEL, body_chars, model.max_tokens))]
    assert ledger.reserved() == 0

    small, small_ledger = gateway_model(tmp_path, gateway, budget_cny=1.0, budget_stop_cny=0.03)
    with pytest.raises(BudgetExceeded, match="CHAT_BACKEND=fake") as caught:
        small.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.BUDGET_EXCEEDED
    assert len(gateway.requests) == 1  # nothing was sent
    assert small_ledger.reserved() == 0
    assert len(small_ledger.entries()) == 1


# ---- errors ---------------------------------------------------------------------------------


def test_timeout_is_retried_recorded_and_classified(tmp_path: Path) -> None:
    gateway = FakeChatGateway([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
    model, ledger = gateway_model(tmp_path, gateway)  # online: 1 retry

    with pytest.raises(openai.APITimeoutError) as caught:
        model.invoke("ping")

    assert chat_error_category(caught.value) is ErrorCategory.LLM_TIMEOUT
    assert len(gateway.requests) == 2
    rows = ledger.entries()
    assert [(r["usage_estimated"], r["error"]) for r in rows] == [(True, "ReadTimeout")] * 2
    assert ledger.reserved() == 0

    # A connect timeout never reached the gateway: classified the same, but not billed.
    gateway.script = [httpx.ConnectTimeout("no route"), httpx.ConnectTimeout("no route")]
    with pytest.raises(openai.APITimeoutError) as caught:
        model.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.LLM_TIMEOUT
    assert len(ledger.entries()) == 2


def test_gateway_errors_classified_and_retry_after_capped(tmp_path: Path) -> None:
    gateway = FakeChatGateway([httpx.Response(503), httpx.Response(503)])
    model, ledger = gateway_model(tmp_path, gateway)

    with pytest.raises(openai.InternalServerError) as caught:
        model.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.LLM_ERROR
    assert len(gateway.requests) == 2

    # Retry-After longer than the online profile waits: no retry.
    gateway.script = [httpx.Response(429, headers={"Retry-After": "30"})]
    with pytest.raises(openai.RateLimitError) as caught:
        model.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.LLM_ERROR
    assert len(gateway.requests) == 3

    # A short Retry-After is honoured by the SDK's retry.
    gateway.script = [httpx.Response(429, headers={"Retry-After": "1"})]
    assert model.invoke("ping").content == "pong pong"
    assert len(gateway.requests) == 5

    gateway.script = [httpx.ConnectError("refused"), httpx.ConnectError("refused")]
    with pytest.raises(openai.APIConnectionError) as caught:
        model.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.LLM_ERROR

    # Only the one answered request is in the ledger; errors are not billed.
    assert len(ledger.entries()) == 1
    assert ledger.reserved() == 0
    assert chat_error_category(KeyError("unrelated bug")) is None


def test_error_body_masks_api_key(tmp_path: Path) -> None:
    gateway = FakeChatGateway(
        [httpx.Response(401, json={"error": {"message": f"bad key: Bearer {KEY}"}})]
    )
    model, _ = gateway_model(tmp_path, gateway)

    with pytest.raises(openai.AuthenticationError) as caught:
        model.invoke("ping")

    assert KEY not in str(caught.value)
    assert KEY not in json.dumps(caught.value.body)
    assert "***" in str(caught.value)


# ---- structured output ----------------------------------------------------------------------


class Route(BaseModel):
    route: Literal["search", "answer"]
    reason: str


def _tool_call(args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "Route", "args": args, "id": "call-1"}])


def test_structured_output_parses_with_fake_model(tmp_path: Path) -> None:
    fake = FakeChatModel(
        responses=[
            _tool_call({"route": "search", "reason": "needs docs"}),
            _tool_call({"route": "answer", "reason": "chit-chat"}),
        ]
    )

    explicit = structured_output(fake, Route, method="function_calling")
    assert explicit.invoke("which route?") == Route(route="search", reason="needs docs")

    # The method comes from the settings when not given.
    configured = structured_output(fake, Route, settings=make_settings(tmp_path))
    assert configured.invoke("hello") == Route(route="answer", reason="chit-chat")


def test_structured_output_failures_are_classified() -> None:
    fake = FakeChatModel(responses=["just text", _tool_call({"route": "dance", "reason": "x"})])
    runnable = structured_output(fake, Route, method="function_calling")

    with pytest.raises(StructuredOutputError, match="no Route") as no_call:
        runnable.invoke("q1")
    with pytest.raises(StructuredOutputError, match="route") as invalid:
        runnable.invoke("q2")

    for caught in (no_call, invalid):
        assert chat_error_category(caught.value) is ErrorCategory.STRUCTURED_OUTPUT_FAILED


def test_structured_output_methods_request_shape(tmp_path: Path) -> None:
    args = {"route": "search", "reason": "needs docs"}
    tool_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "Route", "arguments": json.dumps(args)},
            }
        ],
    }
    json_message = {"role": "assistant", "content": json.dumps(args)}
    gateway = FakeChatGateway([tool_message, json_message])
    model, ledger = gateway_model(tmp_path, gateway, profile="cheap")

    assert structured_output(model, Route, method="function_calling").invoke("q") == Route(**args)
    assert structured_output(model, Route, method="json_schema").invoke("q") == Route(**args)

    calling, schema = gateway.bodies
    assert [t["function"]["name"] for t in calling["tools"]] == ["Route"]
    assert calling["tool_choice"] == {"type": "function", "function": {"name": "Route"}}
    assert "response_format" not in calling
    assert schema["response_format"]["type"] == "json_schema"
    assert schema["response_format"]["json_schema"]["name"] == "Route"
    assert "tools" not in schema
    assert [r["profile"] for r in ledger.entries()] == ["cheap", "cheap"]


# ---- fallback -------------------------------------------------------------------------------


def test_fake_backend_fallback(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the fake backend must not send requests")

    settings = make_settings(
        tmp_path, chat_backend="fake", chat_fake_reply="[fake model]", llm_base_url=""
    )
    for profile in ("online", "offline", "cheap"):
        model = make_chat_model(profile, settings, transport=httpx.MockTransport(refuse))
        assert isinstance(model, FakeChatModel)
        assert model.invoke("q1").content == "[fake model]"
        assert model.invoke("q2").content == "[fake model]"
    assert not (tmp_path / "ledger.jsonl").exists()

    # The price table entry is required only for the gateway backend.
    prices = {"other": ModelPrice(input_cny_per_m=1.0)}
    unpriced = make_settings(tmp_path, chat_backend="fake", model_prices=prices)
    assert isinstance(make_chat_model("online", unpriced), FakeChatModel)
