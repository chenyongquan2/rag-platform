# ruff: noqa: RUF001 (fullwidth citation brackets are what the format check counts)
"""Spec P1-02 AC15: one real answer from three given Hugging Face doc chunks.

Run once when a spec says so: `uv run pytest -m llm tests/llm/test_answer_gateway.py -s`.
Uses the `online` profile (`deepseek-v4-flash`) with the output capped at `MAX_TOKENS` per call
(reasoning and answer together). Before calling, the worst case (first call plus one parse retry,
both using the whole cap) is checked to stay under `LIMIT_CNY` and within the budget stop. Prints
the raw answer and how well the model follows the `[n]` format.

The chunks (`fixtures/hf_longt5_chunks.json`) are three sections of the Transformers docs, cut at
headings from the pinned corpus: the answer is in fragment [2], [1] and [3] are near misses.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

from rag.answer import (
    AnswerDelta,
    AnswerDone,
    AnswerRetry,
    build_context,
    build_messages,
    estimate_tokens,
    parse_answer,
    stream_answer,
)
from rag.config import AnswerConfig, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.llm import resolve_chat_profile
from rag.types import Chunk

pytestmark = pytest.mark.llm

FIXTURE = Path(__file__).parent / "fixtures" / "hf_longt5_chunks.json"
MAX_TOKENS = 512
LIMIT_CNY = 0.02
RETRY_EXTRA_TOKENS = MAX_TOKENS + 100  # the first answer and the reminder, sent again
_CANONICAL = re.compile(r"\[\d+\]")
_ANY_MARKER = re.compile(r"[\[【［][^\]】］\n]{1,20}[\]】］]")


async def _events(*args: Any, **kwargs: Any) -> list[Any]:
    return [event async for event in stream_answer(*args, **kwargs)]


def test_real_answer_cites_fragments() -> None:
    settings = ServiceSettings(chat_online={"max_tokens": MAX_TOKENS})
    if settings.chat_backend != "gateway" or not settings.llm_base_url:
        pytest.skip("the chat gateway is not configured (LLM_BASE_URL, CHAT_BACKEND)")
    data = json.loads(FIXTURE.read_text("utf-8"))
    question: str = data["question"]
    config = AnswerConfig(context_tokens=4000, language="auto", parse_retries=1)
    context = build_context([[Chunk(**c) for c in data["chunks"]]], config.context_tokens)
    assert len(context.fragments) == 3

    # Cost check before the call: the worst case must stay under the limit and the budget stop.
    model = resolve_chat_profile("online", settings).model
    ledger = CostLedger.from_settings(settings)
    prompt = sum(estimate_tokens(str(m.content)) for m in build_messages(question, context, "en"))
    worst = ledger.estimate(model, prompt, MAX_TOKENS) + ledger.estimate(
        model, prompt + RETRY_EXTRA_TOKENS, MAX_TOKENS
    )
    print(f"\n[cost] estimated prompt {prompt} tokens; worst case {worst:.6f} CNY")
    assert worst < LIMIT_CNY
    try:
        ledger.check_budget(worst)
    except BudgetExceeded as exc:
        pytest.skip(str(exc))
    before = len(ledger.entries())

    start = time.perf_counter()
    events = asyncio.run(_events(question, context, kb="hf_docs", config=config, settings=settings))
    seconds = time.perf_counter() - start

    rows = ledger.entries()[before:]
    spent = sum(float(r["cny"]) for r in rows)
    done = events[-1]
    assert isinstance(done, AnswerDone)
    answer = done.answer
    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    first = "".join(d.text for d in deltas if d.attempt == 1)
    parsed = parse_answer(first)
    cited_first = sum(1 for s in parsed.sentences if s.markers)
    markers = _ANY_MARKER.findall(first)
    print(
        f"[answer] {seconds:.2f} s, {len(deltas)} deltas, retries: "
        f"{sum(isinstance(e, AnswerRetry) for e in events)}"
    )
    print(f"[raw first attempt] {first!r}")
    print(
        f"[format] first attempt: {cited_first}/{len(parsed.sentences)} sentences with a marker; "
        f"{len(_CANONICAL.findall(first))}/{len(markers)} markers written as [n]; "
        f"not_found={parsed.not_found}"
    )
    print(f"[checked] {answer.citation_check.model_dump()}, found={answer.found}")
    for sentence in answer.sentences:
        print(f"  {sentence.text!r} -> {[c.marker for c in sentence.citations]}")
    for row in rows:
        print(f"[ledger] {row}")
    print(f"[spent] {spent:.6f} CNY in {len(rows)} calls")

    assert spent < LIMIT_CNY
    assert rows and all(r["profile"] == "online" for r in rows)
    assert answer.generation_error is None and not answer.answer_parse_failed
    assert answer.found
    cited = {c.chunk_id for s in answer.sentences for c in s.citations}
    assert data["answer_chunk_id"] in cited
    assert "127" in answer.text
    assert len(deltas) > 1  # streamed, not one block
