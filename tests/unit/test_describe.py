from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from rag.config import ModelPrice
from rag.cost import CostLedger
from rag.ingest.describe import (
    DescribeBudgetError,
    GatewayTableSummarizer,
    GatewayVisionDescriber,
    ModelReply,
    ModelTextCache,
    ModelTextService,
    SpendGuard,
    estimate_image_cost,
    estimate_table_cost,
    prepare_image,
)
from rag.testing.fakes import FakeChatModel
from rag.testing.pdf import FakeSummarizer, FakeVision, tiny_png

PRICES = {"m": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0)}


def ledger(tmp_path: Path, stop: float = 28.0) -> CostLedger:
    return CostLedger(tmp_path / "ledger.jsonl", prices=PRICES, budget_stop_cny=stop)


# ---- the cache (AC6) -------------------------------------------------------------------------


def test_describe_cache_round_trip_survives_a_new_connection(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "model_text.sqlite3"
    first = ModelTextCache(path)
    first.put("image", "m", "v1", "abc", "a bar chart", prompt_tokens=10, completion_tokens=5)
    first.close()
    second = ModelTextCache(path)
    hit = second.get("image", "m", "v1", "abc")
    assert hit is not None
    assert (hit.text, hit.prompt_tokens, hit.completion_tokens) == ("a bar chart", 10, 5)


def test_describe_cache_key_has_kind_model_prompt_version_and_content() -> None:
    cache = ModelTextCache(":memory:")
    cache.put("image", "m", "v1", "abc", "text", prompt_tokens=1, completion_tokens=1)
    assert cache.get("table", "m", "v1", "abc") is None
    assert cache.get("image", "other", "v1", "abc") is None
    assert cache.get("image", "m", "v2", "abc") is None
    assert cache.get("image", "m", "v1", "abd") is None
    assert cache.get("image", "m", "v1", "abc") is not None


def test_describe_cache_put_overwrites() -> None:
    cache = ModelTextCache(":memory:")
    cache.put("image", "m", "v1", "abc", "old", prompt_tokens=1, completion_tokens=1)
    cache.put("image", "m", "v1", "abc", "new", prompt_tokens=1, completion_tokens=1)
    hit = cache.get("image", "m", "v1", "abc")
    assert hit is not None and hit.text == "new"


# ---- the service: cache, guard, stats --------------------------------------------------------


def service(
    tmp_path: Path, *, vision: FakeVision | None = None, budget: float | None = None
) -> tuple[ModelTextService, FakeVision, FakeSummarizer]:
    vision = vision or FakeVision()
    summarizer = FakeSummarizer()
    guard = SpendGuard(budget)
    svc = ModelTextService(
        cache=ModelTextCache(tmp_path / "c.sqlite3"),
        describer=vision,
        summarizer=summarizer,
        guard=guard,
        image_estimate_cny=0.003,
        table_estimate_cny=0.002,
    )
    return svc, vision, summarizer


def test_describe_cache_second_read_makes_no_model_call(tmp_path: Path) -> None:
    svc, vision, summarizer = service(tmp_path)
    png = tiny_png()
    first = svc.describe_image(png, caption="Figure 1", language="en")
    second = svc.describe_image(png, caption="Figure 1", language="en")
    assert first == second
    assert vision.calls == 1
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert svc.summarize_table(table, caption="") == svc.summarize_table(table, caption="")
    assert summarizer.calls == 1
    stats = svc.stats
    assert (stats.images_described, stats.image_cache_hits) == (1, 1)
    assert (stats.tables_summarised, stats.table_cache_hits) == (1, 1)


def test_describe_cache_is_shared_between_service_instances(tmp_path: Path) -> None:
    svc, vision, _ = service(tmp_path)
    svc.describe_image(tiny_png(), caption="", language="en")
    again, vision2, _ = service(tmp_path)
    again.describe_image(tiny_png(), caption="", language="en")
    assert (vision.calls, vision2.calls) == (1, 0)
    assert again.stats.image_cache_hits == 1 and again.stats.spend_cny == 0.0


def test_describe_different_images_are_different_calls(tmp_path: Path) -> None:
    svc, vision, _ = service(tmp_path)
    svc.describe_image(tiny_png(color=(1, 2, 3)), caption="", language="en")
    svc.describe_image(tiny_png(color=(3, 2, 1)), caption="", language="en")
    assert vision.calls == 2


def test_spend_guard_stops_before_the_call_that_would_pass_the_budget(tmp_path: Path) -> None:
    svc, vision, _ = service(tmp_path, budget=0.005)
    svc.describe_image(tiny_png(color=(1, 1, 1)), caption="", language="en")  # 0.003 spent
    with pytest.raises(DescribeBudgetError, match=r"0\.005"):
        svc.describe_image(tiny_png(color=(2, 2, 2)), caption="", language="en")
    assert vision.calls == 1  # the second call was never made


def test_spend_guard_counts_real_cost_and_ignores_cache_hits(tmp_path: Path) -> None:
    guard = SpendGuard(0.01)
    guard.check(0.004)
    guard.add(0.004)
    guard.add(0.004)
    assert guard.spent == pytest.approx(0.008)
    with pytest.raises(DescribeBudgetError):
        guard.check(0.003)
    SpendGuard(None).check(1e9)  # no budget: nothing to stop


def test_describe_model_error_is_counted_and_raised_to_the_caller(tmp_path: Path) -> None:
    class Broken(FakeVision):
        def describe(self, png: bytes, *, caption: str, language: str) -> ModelReply:
            raise TimeoutError("gateway too slow")

    svc, _, _ = service(tmp_path, vision=Broken())
    with pytest.raises(TimeoutError):
        svc.describe_image(tiny_png(), caption="", language="en")
    assert svc.stats.errors == ["image: TimeoutError: gateway too slow"]


# ---- estimates (AC6) -------------------------------------------------------------------------


def test_estimates_are_upper_bounds_from_the_price_table(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    one = estimate_image_cost(led, "m", 1)
    assert one == pytest.approx(led.estimate("m", 1700, 300))
    assert estimate_image_cost(led, "m", 10) == pytest.approx(10 * one)
    table = estimate_table_cost(led, "m", [300, 900])
    assert table == pytest.approx(led.estimate("m", 300 + 900 + 2 * 200, 2 * 150))
    assert estimate_image_cost(led, "m", 0) == 0.0


def test_estimate_needs_a_price(tmp_path: Path) -> None:
    from rag.cost import UnknownModelPriceError

    with pytest.raises(UnknownModelPriceError):
        estimate_image_cost(ledger(tmp_path), "no-such-model", 1)


# ---- the gateway clients ---------------------------------------------------------------------


def reply(text: str, tokens_in: int = 120, tokens_out: int = 40) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    )


def test_gateway_vision_sends_a_small_jpeg_and_reads_usage(tmp_path: Path) -> None:
    chat = FakeChatModel(responses=[reply("A line chart of loss.")])
    client = GatewayVisionDescriber(chat, model="m", ledger=ledger(tmp_path), max_px=64)
    out = client.describe(tiny_png(400, 300), caption="Figure 2: loss", language="en")
    assert out.text == "A line chart of loss."
    assert (out.prompt_tokens, out.completion_tokens) == (120, 40)
    assert out.cny == pytest.approx((120 * 2.0 + 40 * 8.0) / 1e6)
    (message,) = chat.calls[0]
    assert isinstance(message, HumanMessage)
    parts = message.content
    assert isinstance(parts, list)
    text_part = next(p for p in parts if p["type"] == "text")["text"]
    assert "Figure 2: loss" in text_part and "English" in text_part
    url = next(p for p in parts if p["type"] == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    sent = base64.b64decode(url.split(",", 1)[1])
    assert prepare_image(tiny_png(400, 300), max_px=64) == sent
    assert max(Image.open(io.BytesIO(sent)).size) <= 64


def test_gateway_vision_empty_answer_is_an_error(tmp_path: Path) -> None:
    chat = FakeChatModel(responses=[reply("   ")])
    client = GatewayVisionDescriber(chat, model="m", ledger=ledger(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        client.describe(tiny_png(), caption="", language="en")


def test_gateway_table_summarizer_sends_the_table_text(tmp_path: Path) -> None:
    chat = FakeChatModel(responses=[reply("Flow and power of three pumps.")])
    client = GatewayTableSummarizer(chat, model="m", ledger=ledger(tmp_path), max_table_chars=50)
    out = client.summarize("| Pump | Flow |\n|---|---|\n" + "| Zephyr | 120 |\n" * 20, caption="T1")
    assert out.text == "Flow and power of three pumps."
    (message,) = chat.calls[0]
    body = str(message.content)
    assert "| Pump | Flow |" in body and "T1" in body
    assert len(body) < 1500  # the table text is cut at max_table_chars


# ---- the shared service ------------------------------------------------------------------------


def test_shared_service_is_one_object_that_each_caller_completes(tmp_path: Path) -> None:
    from rag.config import ServiceSettings
    from rag.ingest.describe import reset_shared_services, shared_service

    reset_shared_services()
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        chat_backend="fake",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    cache = tmp_path / "text.sqlite3"
    try:
        first = shared_service(
            settings, vision_model="qwen/qwen3.6-flash", cache_path=cache, budget_cny=0.5
        )
        assert first.describer is not None and first.summarizer is None
        second = shared_service(
            settings, text_model="deepseek-v4-flash", cache_path=cache, budget_cny=0.5
        )
        assert second is first and first.summarizer is not None
        assert first.image_estimate_cny > 0 and first.table_estimate_cny > 0
        assert first.guard.budget_cny == 0.5
        other = shared_service(
            settings, vision_model="qwen/qwen3.6-flash", cache_path=cache, budget_cny=0.1
        )
        assert other is not first  # another budget, another guard
    finally:
        reset_shared_services()
