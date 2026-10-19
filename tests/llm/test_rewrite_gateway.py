# ruff: noqa: RUF001 (a Chinese question uses full-width punctuation)
"""Spec P3-02 AC21: the rewrite and route steps with the real `cheap` model.

Run once when a spec says so: `uv run pytest -m llm tests/llm/test_rewrite_gateway.py -s`.
Reads the gateway URL and key from `.env`, appends every call to the real cost ledger, prints
what it observed (queries, latency, tokens, reasoning tokens, CNY). At most `CASES` questions,
about 0.05 CNY at the estimated price in the worst case, usually well under half of that.
Retrieval and the answer model are fakes: only the small model's calls are real.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest

from rag.agent import AgentRuntime, ask_docs
from rag.agent.rules import missing_tokens, protected_tokens
from rag.config import ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.observe import shutdown_observability
from rag.testing import StreamingFakeChatModel
from rag.testing.agent import ScriptedSearchBackend, kb_config

pytestmark = pytest.mark.llm

KB = "hf_docs"
LIMIT_CNY = 0.05
SKIP_AFTER_CNY = 0.035  # stop asking more questions once this much is spent

# (label, session history question, question)
CASES: list[tuple[str, str | None, str]] = [
    (
        "follow-up",
        "How do I push a fine-tuned model to the Hugging Face Hub with push_to_hub?",
        "And how do I make that repository private?",
    ),
    (
        "declarative with identifiers",
        None,
        "How do I load a model in 8-bit with `load_in_8bit=True` in transformers 4.36.0?",
    ),
    (
        "two-hop",
        None,
        "What is the difference between Trainer and SFTTrainer, and how do I push the "
        "result of SFTTrainer to the Hub?",
    ),
    ("chinese", None, "怎么用 push_to_hub 把训练好的模型上传到 Hub？"),
    (
        "undecided (route model)",
        None,
        "How do I save this model to the hub?",
    ),
]


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture(scope="module")
def settings() -> ServiceSettings:
    s = ServiceSettings(
        otel_mode="memory",
        otel_auto_instrument=False,
        business_record_path="",
        log_dir="",
        log_stderr_level="OFF",
        checkpoint_url="memory://",
        session_store_url="memory://",
        session_lock_url="memory://",
    )
    if s.chat_backend != "gateway" or not s.llm_base_url:
        pytest.skip("the chat gateway is not configured (LLM_BASE_URL, CHAT_BACKEND)")
    ledger = CostLedger.from_settings(s)
    try:
        ledger.check_budget(
            len(CASES) * ledger.estimate(s.chat_model, 800, s.chat_cheap.max_tokens)
        )
    except BudgetExceeded as exc:
        pytest.skip(str(exc))
    return s


AGENT = {"rewrite": True, "clarify": True, "route_model": True, "max_sub_questions": 3}


async def _ask(
    runtime: AgentRuntime,
    settings: ServiceSettings,
    text: str,
    backend: Any,
    answers: Any,
    session: str,
) -> Any:
    return await ask_docs(
        text,
        cfg=kb_config(KB, **AGENT),
        retriever=backend,
        settings=settings,
        runtime=runtime,
        model=answers,
        session_id=session,
    )


def test_real_cheap_model_rewrites_and_routes(settings: ServiceSettings) -> None:
    ledger = CostLedger.from_settings(settings)
    start_total = ledger.total_spent()
    print(f"\n[ledger before] {start_total:.4f} CNY spent of {settings.budget_cny}")
    began_ledger = len(ledger.entries())
    runtime = AgentRuntime.from_settings(settings)
    results: list[str] = []

    async def main() -> None:
        # One event loop for every case: the gateway client the runtime builds once is bound to
        # the loop it first ran on (a service has one loop; `asyncio.run` per call would not).
        for label, earlier, question in CASES:
            if ledger.total_spent() - start_total > SKIP_AFTER_CNY:
                print(f"[{label}] skipped: the spending limit of this test is reached")
                continue
            backend = ScriptedSearchBackend(KB)
            answers = StreamingFakeChatModel(default_response="Use push_to_hub [1].")
            session = f"s-{label.split()[0]}"
            if earlier:  # fills the session's history (a model call too: a question)
                await _ask(runtime, settings, earlier, backend, answers, session)
                backend.calls.clear()
            before = len(ledger.entries())
            t0 = time.perf_counter()
            result = await _ask(runtime, settings, question, backend, answers, session)
            seconds = time.perf_counter() - t0
            rows = ledger.entries()[before:]
            queries = [c["question"] for c in backend.calls]
            print(
                f"[{label}] {seconds:.2f} s, status={result.status}, "
                f"fallbacks={result.fallbacks}, standalone={result.standalone_question!r}, "
                f"queries={result.search_queries}, searched={queries}, "
                f"tokens_used={result.tokens_used}, calls={len(rows)}, "
                f"reasoning={[r.get('reasoning_tokens') for r in rows]}, "
                f"completion={[r.get('completion_tokens') for r in rows]}, "
                f"cny={sum(float(r.get('cny') or 0) for r in rows):.5f}"
            )
            tokens = protected_tokens(question)
            lost = missing_tokens(tokens, *result.search_queries)
            print(f"[{label}] protected={tokens}, lost={lost}")
            results.append(label)
            # Whatever the model did, the request was served and says what happened.
            assert result.status in {"answered", "need_clarification"}
            assert not any(f.endswith(":error") for f in result.fallbacks), result.fallbacks
            if result.status == "answered":
                assert queries, "something was searched"
                if not result.fallbacks:
                    # A rewrite that was used kept every identifier of the question.
                    assert not lost, (tokens, queries)
            for row in rows:
                assert row["usage_estimated"] is False

    asyncio.run(main())
    spent = ledger.total_spent() - start_total
    print(f"[ledger after] spent {spent:.5f} CNY in {len(ledger.entries()) - began_ledger} calls")
    assert results, "at least one case ran"
    assert spent < LIMIT_CNY
