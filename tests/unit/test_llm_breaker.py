"""Spec P8-01 AC6: the chat gateway circuit breaker (design s16-8).

The gateway is an `httpx.MockTransport`; the breaker's clock is a fake. Each test fails if the
breaker does not stop sending (the mock counts the requests that reached it).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from rag.config import AccessConfig, KnowledgeBaseConfig, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.llm import chat_error_category, make_chat_model
from rag.llm.breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpenError,
    reset_shared_breakers,
    shared_breaker,
)
from rag.llm.transport import AsyncLedgerTransport, LedgerTransport
from rag.observe import shutdown_observability
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import FakeChatModel, memory_observability, metric_points
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder
from rag.types import ErrorCategory

MODEL = "deepseek-v4-flash"
KEY = "sk-unit-test-key-0123456789"  # not a key: the fake gateway checks nothing


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    shutdown_observability()
    reset_shared_breakers()
    yield
    reset_shared_breakers()
    shutdown_observability()


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def breaker(failures: int = 3, open_s: float = 30.0) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    return CircuitBreaker(failures=failures, open_s=open_s, clock=clock), clock


def test_opens_after_n_failures_in_a_row_and_fails_fast() -> None:
    b, _ = breaker(failures=3)
    for _ in range(2):
        b.before_call()
        b.record_failure()
    assert b.state == CLOSED
    b.before_call()
    b.record_failure()
    assert b.state == OPEN
    with pytest.raises(CircuitOpenError):
        b.before_call()


def test_a_success_resets_the_count() -> None:
    b, _ = breaker(failures=3)
    for outcome in ("f", "f", "s", "f", "f"):
        b.before_call()
        b.record_failure() if outcome == "f" else b.record_success()
    assert b.state == CLOSED


def test_half_open_lets_one_trial_through_and_closes_on_success() -> None:
    b, clock = breaker(failures=1, open_s=30.0)
    b.before_call()
    b.record_failure()
    clock.now += 29.9
    with pytest.raises(CircuitOpenError):
        b.before_call()
    clock.now += 0.2
    assert b.state == HALF_OPEN
    b.before_call()  # the trial
    with pytest.raises(CircuitOpenError):
        b.before_call()  # a second caller while the trial runs
    b.record_success()
    assert b.state == CLOSED
    b.before_call()


def test_a_failed_trial_opens_again_for_the_full_time() -> None:
    b, clock = breaker(failures=1, open_s=30.0)
    b.before_call()
    b.record_failure()
    clock.now += 31
    b.before_call()
    b.record_failure()
    assert b.state == OPEN
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        b.before_call()


def test_a_released_trial_frees_the_slot() -> None:
    b, clock = breaker(failures=1)
    b.before_call()
    b.record_failure()
    clock.now += 31
    b.before_call()
    b.release()  # e.g. the budget stop refused it before sending
    b.before_call()


def test_shared_breaker_is_one_per_gateway_and_off_at_zero() -> None:
    a = shared_breaker("http://g1", failures=2, open_s=5)
    assert shared_breaker("http://g1", failures=2, open_s=5) is a
    assert shared_breaker("http://g2", failures=2, open_s=5) is not a
    assert shared_breaker("http://g1", failures=0, open_s=5) is None


# ---- in the transports -------------------------------------------------------------------------


class Gateway:
    """Answers each request with the next scripted item (a status code or an exception), then
    with 200 and a chat completion."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        item = self.script.pop(0) if self.script else 200
        if isinstance(item, BaseException):
            raise item
        if item != 200:
            return httpx.Response(item, json={"error": {"message": "nope"}})
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


def ledger(tmp_path: Path, stop: float = 10.0) -> CostLedger:
    settings = ServiceSettings(_env_file=None, cost_ledger_path=str(tmp_path / "l.jsonl"))  # type: ignore[call-arg]
    return CostLedger(settings.cost_ledger_path, prices=settings.model_prices, budget_stop_cny=stop)


def request() -> httpx.Request:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}
    return httpx.Request("POST", "http://gateway.test/v1/chat/completions", json=body)


def sync_transport(tmp_path: Path, gateway: Gateway, b: CircuitBreaker, **kw: Any) -> Any:
    return LedgerTransport(
        kw.pop("ledger", None) or ledger(tmp_path),
        profile="online",
        max_retries=0,
        backoff_s=0,
        backoff_max_s=0,
        deadline_s=None,
        transport=httpx.MockTransport(gateway),
        breaker=b,
    )


@pytest.mark.parametrize(
    "failure", [503, 500, 429, 408, httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]
)
def test_gateway_failures_open_the_breaker_and_later_calls_are_not_sent(
    tmp_path: Path, failure: Any
) -> None:
    gateway = Gateway([failure, failure])
    b, _ = breaker(failures=2)
    t = sync_transport(tmp_path, gateway, b)
    for _ in range(2):
        with contextlib.suppress(httpx.TransportError):
            t.handle_request(request())
    assert b.state == OPEN
    with pytest.raises(CircuitOpenError):
        t.handle_request(request())
    assert gateway.requests == 2


def test_4xx_answers_and_budget_stops_do_not_count(tmp_path: Path) -> None:
    gateway = Gateway([400, 401, 404, 400])
    b, _ = breaker(failures=2)
    t = sync_transport(tmp_path, gateway, b)
    for _ in range(4):
        assert t.handle_request(request()).status_code >= 400
    assert b.state == CLOSED
    poor = sync_transport(tmp_path, gateway, b, ledger=ledger(tmp_path / "x", stop=0.0))
    for _ in range(3):
        with pytest.raises(BudgetExceeded):
            poor.handle_request(request())
    assert b.state == CLOSED and gateway.requests == 4


def test_the_async_transport_uses_the_breaker_too(tmp_path: Path) -> None:
    gateway = Gateway([503, 503])
    b, clock = breaker(failures=2, open_s=10)
    t = AsyncLedgerTransport(
        ledger(tmp_path),
        profile="online",
        max_retries=0,
        backoff_s=0,
        backoff_max_s=0,
        deadline_s=None,
        transport=httpx.MockTransport(gateway),
        breaker=b,
    )

    async def main() -> None:
        for _ in range(2):
            await t.handle_async_request(request())
        with pytest.raises(CircuitOpenError):
            await t.handle_async_request(request())
        clock.now += 11
        response = await t.handle_async_request(request())  # the trial, and it works
        await response.aread()

    asyncio.run(main())
    assert gateway.requests == 3 and b.state == CLOSED


def test_the_open_breaker_is_the_existing_llm_error_category(tmp_path: Path) -> None:
    gateway = Gateway([503, 503])
    settings = ServiceSettings(  # type: ignore[call-arg]
        _env_file=None,
        llm_base_url="http://gateway.test/v1",
        llm_api_key=KEY,
        cost_ledger_path=str(tmp_path / "l.jsonl"),
        chat_online={"max_retries": 0},
        chat_breaker_failures=2,
    )
    transport = httpx.MockTransport(gateway)
    model = make_chat_model("online", settings, transport=transport, async_transport=transport)
    for _ in range(2):
        with pytest.raises(openai.InternalServerError):
            model.invoke("ping")
    with pytest.raises(openai.APIConnectionError) as caught:
        model.invoke("ping")
    assert chat_error_category(caught.value) is ErrorCategory.LLM_ERROR
    # Another profile on the same gateway shares the breaker.
    other = make_chat_model("cheap", settings, transport=transport, async_transport=transport)
    with pytest.raises(openai.APIConnectionError):
        other.invoke("ping")
    assert gateway.requests == 2


def test_breaker_events_are_counted(tmp_path: Path) -> None:
    with memory_observability() as obs:
        b, clock = breaker(failures=1, open_s=5)
        b.before_call()
        b.record_failure()
        with pytest.raises(CircuitOpenError):
            b.before_call()
        clock.now += 6
        b.before_call()
        b.record_success()
        points = metric_points(obs, "rag.model.breaker")
    by_event = {p[0]["rag.breaker.event"]: p[1].value for p in points}
    assert by_event == {"opened": 1, "rejected": 1, "closed": 1}


# ---- the service: ask_docs answers with the passages, search_docs works ----------------------


def test_ask_docs_with_an_open_breaker_returns_passages_without_calling_the_gateway(
    tmp_path: Path,
) -> None:
    gateway = Gateway([503] * 10)
    settings = agent_settings(
        llm_base_url="http://gateway.test/v1",
        llm_api_key=KEY,
        cost_ledger_path=str(tmp_path / "l.jsonl"),
        chat_online={"max_retries": 0},
        chat_breaker_failures=2,
    )
    transport = httpx.MockTransport(gateway)
    answer = make_chat_model("online", settings, transport=transport, async_transport=transport)
    backend = ScriptedSearchBackend("hf_docs")
    service = SearchService(
        settings,
        configs={"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=answer,
        helper_model=FakeChatModel(default_response="{}"),
    )
    service.load()
    creds = Credentials(transport="http")
    responses = [
        asyncio.run(service.ask_docs(creds, kb="hf_docs", question=f"push a model {i}?"))
        for i in range(4)
    ]
    sent = gateway.requests
    assert sent <= 3, sent  # at most the failures before it opened (plus an answer retry)
    for response in responses:
        assert ErrorCategory.LLM_ERROR in response.error_categories
        assert response.sources, json.dumps(response.model_dump(mode="json"))[:500]
    assert service.search_docs(creds, kb="hf_docs", query="push a model").chunks
    assert gateway.requests == sent
