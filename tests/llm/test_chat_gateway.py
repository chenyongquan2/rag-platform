"""Spec P1-01 AC13-AC16: short calls through the real gateway (a few hundred tokens each).

Run once when a spec says so: `uv run pytest -m llm tests/llm/test_chat_gateway.py -s`.
Reads the gateway URL and key from `.env` and appends every call to the real cost ledger.
Every probe caps output at `PROBE_MAX_TOKENS` (reasoning and answer together), so the whole file
costs at most about 0.05 CNY at the estimated price, usually far less. Prints what it observed.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag.config import ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.llm import chat_error_category, make_chat_model, structured_output

pytestmark = pytest.mark.llm

PROBE_MAX_TOKENS = 1024
PROBES = 6  # calls in this file


@pytest.fixture(scope="module")
def settings() -> ServiceSettings:
    capped = {"max_tokens": PROBE_MAX_TOKENS}
    s = ServiceSettings(chat_online=capped, chat_cheap=capped)
    if s.chat_backend != "gateway" or not s.llm_base_url:
        pytest.skip("the chat gateway is not configured (LLM_BASE_URL, CHAT_BACKEND)")
    ledger = CostLedger.from_settings(s)
    try:
        ledger.check_budget(PROBES * ledger.estimate(s.chat_model, 2000, PROBE_MAX_TOKENS))
    except BudgetExceeded as exc:
        pytest.skip(str(exc))
    return s


def _new_rows(ledger: CostLedger, before: int) -> list[dict[str, Any]]:
    return ledger.entries()[before:]


def test_real_gateway_reply_counts_reasoning_tokens(settings: ServiceSettings) -> None:
    ledger = CostLedger.from_settings(settings)
    before = len(ledger.entries())
    model = make_chat_model("online", settings)

    start = time.perf_counter()
    reply = model.invoke(
        [
            SystemMessage("Answer with one word."),
            HumanMessage("What is the capital of France?"),
        ]
    )
    seconds = time.perf_counter() - start

    (row,) = _new_rows(ledger, before)
    usage = reply.usage_metadata or {}
    print(
        f"\n[reply] {seconds:.2f} s, content={reply.content!r}, "
        f"finish={reply.response_metadata.get('finish_reason')}, usage={usage}, ledger={row}"
    )
    assert "paris" in str(reply.content).lower()
    assert row["usage_estimated"] is False
    assert row["prompt_tokens"] > 0
    assert row["completion_tokens"] >= row["reasoning_tokens"]
    assert row["reasoning_tokens"] == usage.get("output_token_details", {}).get("reasoning", 0)
    assert row["served_model"]


def test_real_gateway_max_tokens_caps_reasoning(settings: ServiceSettings) -> None:
    """A tight cap is spent on reasoning first: the answer is cut off or empty."""
    ledger = CostLedger.from_settings(settings)
    before = len(ledger.entries())
    tight = ServiceSettings(chat_cheap={"max_tokens": 16})
    model = make_chat_model("cheap", tight)

    reply = model.invoke("List the first twenty prime numbers.")

    (row,) = _new_rows(ledger, before)
    print(
        f"\n[cap 16] content={reply.content!r}, "
        f"finish={reply.response_metadata.get('finish_reason')}, ledger={row}"
    )
    assert reply.response_metadata.get("finish_reason") == "length"
    assert row["completion_tokens"] <= 16


class Verdict(BaseModel):
    """Whether a statement is true."""

    is_true: bool
    reason: str = Field(description="At most ten words.")


def test_real_gateway_structured_output_methods(settings: ServiceSettings) -> None:
    ledger = CostLedger.from_settings(settings)
    before = len(ledger.entries())
    model = make_chat_model("cheap", settings)
    question = "Statement: 7 is a prime number. Is the statement true?"

    results: dict[str, str] = {}
    for method in ("function_calling", "json_schema"):
        start = time.perf_counter()
        try:
            verdict = structured_output(model, Verdict, method=method).invoke(question)  # type: ignore[arg-type]
        except Exception as exc:  # the probe records every failure
            category = chat_error_category(exc)
            results[method] = f"failed: {category} {type(exc).__name__}: {str(exc)[:300]}"
        else:
            results[method] = "ok" if verdict.is_true else f"wrong answer: {verdict}"
        print(f"\n[{method}] {time.perf_counter() - start:.2f} s: {results[method]}")

    for row in _new_rows(ledger, before):
        print(f"[ledger] {row}")
    assert results[settings.chat_structured_output_method] == "ok"


def test_real_gateway_streams_chunks(settings: ServiceSettings) -> None:
    ledger = CostLedger.from_settings(settings)
    before = len(ledger.entries())
    model = make_chat_model("online", settings)

    start = time.perf_counter()
    first_content: float | None = None
    pieces: list[str] = []
    chunks = 0
    for chunk in model.stream("Count from 1 to 10, separated by spaces. Output only the numbers."):
        chunks += 1
        if chunk.content:
            first_content = first_content or time.perf_counter() - start
            pieces.append(str(chunk.content))
    seconds = time.perf_counter() - start

    (row,) = _new_rows(ledger, before)
    print(
        f"\n[stream] {chunks} chunks, {len(pieces)} with content, first content after "
        f"{first_content:.2f} s, total {seconds:.2f} s, text={''.join(pieces)!r}, ledger={row}"
    )
    assert len(pieces) >= 2
    assert "10" in "".join(pieces)
    assert row["stream"] is True
    assert row["usage_estimated"] is False
