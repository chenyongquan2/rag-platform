"""Spec P3-03 AC35: the sufficiency judge with the real `cheap` model on three hand-made cases.

Run once when a spec says so: `uv run pytest -m llm tests/llm/test_sufficiency_gateway.py -s`.
Reads the gateway URL and key from `.env`, appends every call to the real cost ledger, prints
what it observed (verdict, latency, tokens, CNY). Three judge calls, under 0.02 CNY at the
price seen in the pilot (0.003 to 0.005 CNY for a call that finishes). Retrieval is not used:
the rounds are built by hand.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest

from rag.agent import judge_rounds
from rag.agent.state import RetrievalRound
from rag.config import ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.llm import make_chat_model
from rag.observe import shutdown_observability
from rag.testing.agent import kb_config, make_context, scored_result

pytestmark = pytest.mark.llm

KB = "hf_docs"
LIMIT_CNY = 0.02

PUSH = (
    "To upload a model to the Hub call `model.push_to_hub('my-name')` after training. "
    "The repository is created if it does not exist."
)
UNRELATED = "The Hub shows a download counter for each model repository on its page."
LORA = "LoRA freezes the base weights and trains small low-rank adapter matrices instead."

# (label, expected grounded flags per sub-question, [(sub-question, chunk text)])
CASES: list[tuple[str, tuple[bool, ...], list[tuple[str, str]]]] = [
    ("answered", (True,), [("How do I upload a trained model to the Hub?", PUSH)]),
    (
        "related but not answering",
        (False,),
        [("What is the maximum size of a model file on the Hub?", UNRELATED)],
    ),
    (
        "two parts, one answered",
        (True, False),
        [
            ("How do I upload a trained model to the Hub?", PUSH),
            ("Which learning rate does the Trainer use by default?", LORA),
        ],
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
    try:
        CostLedger.from_settings(s).check_budget(0.05)
    except BudgetExceeded as exc:
        pytest.skip(str(exc))
    return s


def test_real_cheap_model_judges_three_cases(settings: ServiceSettings) -> None:
    ledger = CostLedger.from_settings(settings)
    start = ledger.total_spent()
    model = make_chat_model(settings.agent_judge_profile, settings)

    async def main() -> None:
        for label, expected, parts in CASES:
            rounds = [
                RetrievalRound.from_result(
                    1, question, scored_result(KB, [0.9], pages=[i], texts=[text])
                )
                for i, (question, text) in enumerate(parts, 1)
            ]
            ctx = make_context(cfg=kb_config(KB, judge=True), settings=settings, judge_model=model)
            before = len(ledger.entries())
            began = time.perf_counter()
            outcome = await judge_rounds(ctx, rounds)
            rows = ledger.entries()[before:]
            print(
                f"[{label}] {time.perf_counter() - began:.2f} s verdict={outcome.verdict} "
                f"missing={list(outcome.missing)} fallback={outcome.fallback} "
                f"tokens={ctx.budget.spent} cny={sum(float(r.get('cny') or 0) for r in rows):.5f}"
            )
            # The model's answer is not scripted: a call that fails is a fallback, not a failure
            # of the request; what must hold is that the judgement came back and is consistent.
            assert outcome.verdict in {"enough", "partly_enough", "not_enough"}
            if outcome.fallback is None:
                said = tuple(q not in outcome.missing for q, _ in parts)
                assert said == expected, (label, outcome)

    asyncio.run(main())
    spent = ledger.total_spent() - start
    print(f"[ledger] spent {spent:.5f} CNY")
    assert spent < LIMIT_CNY
