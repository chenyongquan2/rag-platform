"""Fake chat models that write one cost-ledger line per call, like the real gateway (spec P3-05).

The eval harness reads the ledger lines a question caused (`collect_ledger_rows`): its tests need
models that write them, with a `profile` so calls can be told apart (`online` answers, `cheap`
rewrite and judge calls).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from rag.config import ModelPrice
from rag.cost import CostLedger
from rag.testing.fakes import FakeChatModel, StreamingFakeChatModel

PRICES = {"fake-chat": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0)}


def write_ledger_row(
    path: str, profile: str, prompt_tokens: int, completion_tokens: int, reasoning: int
) -> None:
    CostLedger(path, prices=PRICES, budget_stop_cny=28.0).record(
        kind="chat",
        model="fake-chat",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning,
        profile=profile,
    )


class LedgerStreamingChat(StreamingFakeChatModel):
    """Streams scripted replies (an answer model) and writes a ledger line per call."""

    ledger_path: str = ""
    profile: str = "online"
    prompt_tokens: int = 1000
    completion_tokens: int = 300

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        message = super()._next_message(messages)
        write_ledger_row(
            self.ledger_path, self.profile, self.prompt_tokens, self.completion_tokens, 100
        )
        return message


class LedgerChat(FakeChatModel):
    """Returns scripted replies (structured ones too: a rewrite or judge model) and writes a
    ledger line per call."""

    ledger_path: str = ""
    profile: str = "cheap"
    prompt_tokens: int = 500
    completion_tokens: int = 100

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        message = super()._next_message(messages)
        write_ledger_row(
            self.ledger_path, self.profile, self.prompt_tokens, self.completion_tokens, 20
        )
        return message
