"""Spec P8-01 AC5: the daily model budget per caller and for the whole service (design s16-4).

The budget reads the cost ledger (a file in tmp_path here): rows of the current UTC day that
carry a `caller` (written inside a service request). Over the budget, ask_docs is refused before
any model call; search_docs still works.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig, ModelPrice
from rag.cost import CostLedger
from rag.observe import shutdown_observability
from rag.observe.context import bind_request
from rag.security.access import Credentials
from rag.serve.errors import ServiceError
from rag.serve.http_app import build_http_app
from rag.serve.limits import DailyBudget, make_daily_budget
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel, observe_settings
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder
from rag.types import ErrorCategory

NOW = datetime(2026, 10, 21, 18, 0, 0, tzinfo=UTC)  # 6 hours to the next UTC day


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def row(time: str, cny: float, caller: str | None) -> str:
    data: dict[str, Any] = {"time": time, "kind": "chat", "model": "m", "cny": cny}
    if caller is not None:
        data["caller"] = caller
    return json.dumps(data) + "\n"


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        row("2026-10-21T01:00:00.000+00:00", 0.5, "local")
        + row("2026-10-21T02:00:00.000+00:00", 0.1, "guest")
        + row("2026-10-20T23:59:59.000+00:00", 5.0, "local")  # yesterday
        + row("2026-10-21T03:00:00.000+00:00", 10.0, None)  # an offline job, no caller
        + "not json\n",
        encoding="utf-8",
        newline="",
    )
    return path


def budget(path: Path, **kwargs: Any) -> DailyBudget:
    kwargs.setdefault("total_cny", 0.0)
    kwargs.setdefault("per_caller_cny", 0.0)
    kwargs.setdefault("callers", {})
    kwargs.setdefault("estimate_cny", 0.02)
    return DailyBudget(path, clock=lambda: NOW, **kwargs)


def test_today_is_counted_per_caller_and_in_total(ledger: Path) -> None:
    b = budget(ledger)
    assert b.spent_today("local") == pytest.approx(0.5)
    assert b.spent_today("guest") == pytest.approx(0.1)
    assert b.spent_today() == pytest.approx(0.6)  # the job without a caller is not the service's


def test_a_caller_over_its_budget_is_refused_until_the_next_day(ledger: Path) -> None:
    b = budget(ledger, per_caller_cny=0.51)
    refusal = b.refusal("local")
    assert refusal is not None and refusal.scope == "caller"
    assert refusal.retry_after_s == pytest.approx(6 * 3600)
    assert b.refusal("guest") is None


def test_the_service_budget_covers_all_callers(ledger: Path) -> None:
    b = budget(ledger, total_cny=0.61)
    assert b.refusal("guest") is not None and b.refusal("guest").scope == "service"
    assert b.refusal("someone_new") is not None


def test_per_caller_overrides_and_zero_turns_a_limit_off(ledger: Path) -> None:
    b = budget(ledger, per_caller_cny=0.2, callers={"local": 2.0, "guest": 0.0})
    assert b.refusal("local") is None
    assert b.refusal("guest") is None  # 0: no limit for this caller
    assert budget(ledger).refusal("local") is None  # every limit 0: off


def test_new_rows_are_seen_and_yesterday_drops_out(ledger: Path) -> None:
    b = budget(ledger, per_caller_cny=1.0)
    assert b.refusal("local") is None
    with ledger.open("a", encoding="utf-8", newline="") as fh:
        fh.write(row("2026-10-21T05:00:00.000+00:00", 0.5, "local"))
    assert b.refusal("local") is not None
    later = DailyBudget(
        ledger,
        total_cny=0,
        per_caller_cny=1.0,
        callers={},
        estimate_cny=0.02,
        clock=lambda: datetime(2026, 10, 22, 0, 0, 1, tzinfo=UTC),
    )
    assert later.refusal("local") is None and later.spent_today("local") == 0.0


def test_a_missing_ledger_counts_as_nothing_spent(tmp_path: Path) -> None:
    assert budget(tmp_path / "none.jsonl", per_caller_cny=0.05).refusal("local") is None


def test_ledger_rows_carry_the_caller_and_request_of_the_current_request(tmp_path: Path) -> None:
    led = CostLedger(
        tmp_path / "l.jsonl", prices={"m": ModelPrice(input_cny_per_m=1.0)}, budget_stop_cny=1.0
    )
    with bind_request("req-9", caller="alice", kb="hf_docs", tool="ask_docs"):
        inside = led.record(kind="chat", model="m", prompt_tokens=1)
    outside = led.record(kind="chat", model="m", prompt_tokens=1)
    assert (inside["caller"], inside["request_id"]) == ("alice", "req-9")
    assert "caller" not in outside and "request_id" not in outside


def test_make_daily_budget_is_off_when_every_limit_is_zero(tmp_path: Path) -> None:
    off = observe_settings(budget_daily_cny=0, budget_daily_caller_cny=0)
    assert make_daily_budget(off) is None
    on = observe_settings(cost_ledger_path=str(tmp_path / "l.jsonl"))
    assert isinstance(make_daily_budget(on), DailyBudget)


# ---- in the service --------------------------------------------------------------------------


class CountingModel(StreamingFakeChatModel):
    invocations: int = 0

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        type(self).invocations += 1
        return super()._generate(*args, **kwargs)

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        type(self).invocations += 1
        return super()._stream(*args, **kwargs)


def _service(ledger: Path) -> tuple[SearchService, ScriptedSearchBackend]:
    backend = ScriptedSearchBackend("hf_docs")
    CountingModel.invocations = 0
    service = SearchService(
        agent_settings(service_allowed_hosts=["testserver"]),
        configs={"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=CountingModel(default_response="Push it [1]."),
        helper_model=CountingModel(default_response="{}"),
        budget=budget(ledger, per_caller_cny=0.4),  # local spent 0.5 today
    )
    return service, backend


def test_ask_docs_is_refused_before_any_model_call_and_search_still_works(ledger: Path) -> None:
    service, backend = _service(ledger)
    service.load()
    creds = Credentials(transport="http")
    with pytest.raises(ServiceError) as info:
        asyncio.run(service.ask_docs(creds, kb="hf_docs", question="push a model?"))
    assert info.value.category == ErrorCategory.BUDGET_EXCEEDED
    assert info.value.retry_after_s == pytest.approx(6 * 3600)
    assert CountingModel.invocations == 0 and backend.calls == []
    response = service.search_docs(creds, kb="hf_docs", query="push a model")
    assert response.kb == "hf_docs"


def test_http_budget_refusal_is_429_with_retry_after(ledger: Path) -> None:
    service, _ = _service(ledger)
    with TestClient(build_http_app(service)) as client:
        assert service.wait_ready(5)
        r = client.post("/v1/ask_docs", json={"kb": "hf_docs", "question": "push?"})
        assert r.status_code == 429 and r.json()["error"] == "budget_exceeded"
        assert r.headers["retry-after"] == str(6 * 3600)
        r = client.post("/v1/ask_docs/stream", json={"kb": "hf_docs", "question": "push?"})
        assert r.status_code == 429


def test_concurrent_asks_reserve_their_estimate(ledger: Path) -> None:
    """Review P8-01 round 1: ten asks at once with 0.03 CNY left must not all pass."""
    b = budget(ledger, per_caller_cny=0.53, estimate_cny=0.02)  # local spent 0.5: room for 1
    first = b.hold("local")
    assert first.refusal is None
    second = b.hold("local")
    assert second.refusal is not None and second.refusal.scope == "caller"
    first.release()
    second.release()  # a refused hold releases nothing
    third = b.hold("local")
    assert third.refusal is None
    third.release()
    third.release()  # twice is harmless


def test_the_service_releases_the_reservation_after_the_ask(ledger: Path) -> None:
    service, _ = _service(ledger)
    service._budget = budget(ledger, per_caller_cny=0.53, estimate_cny=0.02)
    service.load()
    creds = Credentials(transport="http")

    async def two_at_once() -> list[Any]:
        return await asyncio.gather(
            service.ask_docs(creds, kb="hf_docs", question="push a model?"),
            service.ask_docs(creds, kb="hf_docs", question="push a dataset?"),
            return_exceptions=True,
        )

    results = asyncio.run(two_at_once())
    refused = [r for r in results if isinstance(r, ServiceError)]
    assert len(refused) == 1 and refused[0].category == ErrorCategory.BUDGET_EXCEEDED
    # Both finished: nothing is held any more, so one more ask fits again.
    assert asyncio.run(service.ask_docs(creds, kb="hf_docs", question="again?")).kb == "hf_docs"


def test_an_ask_cancelled_during_the_budget_check_holds_nothing(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review P8-01 round 2: the ledger is read in a worker thread; a cancel while it reads must
    not leave a hold behind (it would block the caller's asks for good)."""
    import time as _time

    service, _ = _service(ledger)
    b = budget(ledger, per_caller_cny=0.53, estimate_cny=0.02)  # room for exactly one ask
    service._budget = b
    service.load()
    creds = Credentials(transport="http")
    read = DailyBudget._read

    def slow_read(self: DailyBudget) -> None:
        _time.sleep(0.3)
        read(self)

    monkeypatch.setattr(DailyBudget, "_read", slow_read)

    async def cancel_mid_check() -> None:
        task = asyncio.create_task(service.ask_docs(creds, kb="hf_docs", question="push?"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.5)  # the worker thread finishes its read

    asyncio.run(cancel_mid_check())
    assert b.held_today("local") == 0.0
    monkeypatch.setattr(DailyBudget, "_read", read)
    assert asyncio.run(service.ask_docs(creds, kb="hf_docs", question="again?")).kb == "hf_docs"


def test_holds_are_dropped_at_the_next_utc_day(tmp_path: Path) -> None:
    now = {"t": NOW}
    b = DailyBudget(
        tmp_path / "empty.jsonl",
        total_cny=0,
        per_caller_cny=0.03,
        callers={},
        estimate_cny=0.02,
        clock=lambda: now["t"],
    )
    held = b.hold("local")
    assert held.refusal is None and b.hold("local").refusal is not None
    now["t"] = datetime(2026, 10, 22, 0, 0, 1, tzinfo=UTC)
    assert b.hold("local").refusal is None  # yesterday's hold no longer counts
    held.release()  # releasing yesterday's hold later does not go below zero
    assert b.held_today("local") == pytest.approx(0.02)
