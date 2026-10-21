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


def test_the_thinking_switch_is_sent_to_qwen_models_only() -> None:
    from rag.ingest.describe import extra_body_for

    assert extra_body_for("qwen/qwen3.6-flash") == {"enable_thinking": False}
    assert extra_body_for("Qwen3.8-max") == {"enable_thinking": False}
    for other in ("deepseek-v4-flash", "google/gemini-3.5-flash-lite", "gpt-5.5"):
        assert extra_body_for(other) is None


def test_shared_service_does_not_send_the_switch_to_another_model(tmp_path: Path) -> None:
    reset_shared_services()
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        chat_backend="fake",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    try:
        service = shared_service(
            settings,
            vision_model="deepseek-v4-flash",  # a model with a price that is not a Qwen model
            text_model="deepseek-v4-flash",
            cache_path=tmp_path / "t.sqlite3",
        )
        assert service.describer is not None and service.summarizer is not None
        assert not hasattr(service.describer.chat, "kwargs")  # not wrapped: no extra_body
        assert not hasattr(service.summarizer.chat, "kwargs")
    finally:
        reset_shared_services()


def test_shared_service_refuses_a_second_model_for_the_same_role(tmp_path: Path) -> None:
    import pytest

    reset_shared_services()
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        chat_backend="fake",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    cache = tmp_path / "t.sqlite3"
    try:
        shared_service(settings, vision_model="qwen/qwen3.6-flash", cache_path=cache)
        with pytest.raises(ValueError, match="already uses"):
            shared_service(settings, vision_model="google/gemini-3.5-flash-lite", cache_path=cache)
        shared_service(settings, text_model="deepseek-v4-flash", cache_path=cache)
        with pytest.raises(ValueError, match="already uses"):
            shared_service(settings, text_model="qwen/qwen3.6-flash", cache_path=cache)
        shared_service(settings, vision_model="qwen/qwen3.6-flash", cache_path=cache)  # same: fine
    finally:
        reset_shared_services()


def test_a_description_in_another_language_is_not_taken_from_the_cache(tmp_path: Path) -> None:
    from rag.ingest.describe import ModelTextCache, ModelTextService
    from rag.testing.pdf import FakeVision

    vision = FakeVision()
    service = ModelTextService(cache=ModelTextCache(tmp_path / "c.sqlite3"), describer=vision)
    png = tiny_png()
    service.describe_image(png, caption="", language="en")
    service.describe_image(png, caption="", language="en")
    assert vision.calls == 1
    service.describe_image(png, caption="", language="zh")
    assert vision.calls == 2  # another language is another text
    service.describe_image(png, caption="", language="zh")
    assert vision.calls == 2
