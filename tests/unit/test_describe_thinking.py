"""Spec P6-01: the gateway's Qwen model thinks by default and then answers nothing within the
output cap, so the PDF clients ask it not to (measured in P6-01, see docs/decisions.md)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from rag.config import ModelPrice, ServiceSettings
from rag.cost import CostLedger
from rag.ingest.describe import (
    NO_THINKING,
    GatewayTableSummarizer,
    GatewayVisionDescriber,
    reset_shared_services,
    shared_service,
)
from rag.testing.fakes import FakeChatModel
from rag.testing.pdf import tiny_png

PRICES = {"m": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0)}


def ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(tmp_path / "ledger.jsonl", prices=PRICES, budget_stop_cny=28.0)


def test_gateway_clients_turn_the_thinking_mode_off_when_asked(tmp_path: Path) -> None:
    chat = FakeChatModel(responses=[AIMessage(content="x")])
    vision = GatewayVisionDescriber(
        chat, model="m", ledger=ledger(tmp_path), extra_body=NO_THINKING
    )
    assert vision.chat.kwargs == {"extra_body": {"enable_thinking": False}}
    assert vision.describe(tiny_png(), caption="", language="en").text == "x"
    table = GatewayTableSummarizer(
        FakeChatModel(responses=[AIMessage(content="y")]),
        model="m",
        ledger=ledger(tmp_path),
        extra_body=NO_THINKING,
    )
    assert table.chat.kwargs == {"extra_body": {"enable_thinking": False}}
    assert table.summarize("| a |\n|---|\n| 1 |", caption="").text == "y"
    plain = GatewayVisionDescriber(chat, model="m", ledger=ledger(tmp_path))
    assert plain.chat is chat


def test_shared_service_gateway_clients_have_the_thinking_mode_off(tmp_path: Path) -> None:
    reset_shared_services()
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        chat_backend="fake",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    try:
        service = shared_service(
            settings,
            vision_model="qwen/qwen3.6-flash",
            text_model="qwen/qwen3.6-flash",
            cache_path=tmp_path / "t.sqlite3",
        )
        assert service.describer is not None and service.summarizer is not None
        expected = {"extra_body": {"enable_thinking": False}}
        assert service.describer.chat.kwargs == expected  # type: ignore[attr-defined]
        assert service.summarizer.chat.kwargs == expected  # type: ignore[attr-defined]
    finally:
        reset_shared_services()
