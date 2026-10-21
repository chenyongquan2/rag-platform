"""Spec P5-02 AC3, AC4: the scheduled probe (search every N seconds, ask rarely and capped)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.monitor.probe import ProbeConfig, Prober
from rag.observe import shutdown_observability
from rag.testing import memory_observability, metric_points


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class Clock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _chunk(source_rel: str = "docs/hub/push.md") -> dict[str, Any]:
    return {"chunk_id": "hf_docs:x#0", "source_rel": source_rel, "title_path": ["Hub", "Push"]}


def _search_body(**over: Any) -> dict[str, Any]:
    body = {
        "request_id": "r1",
        "degraded": False,
        "rerank_timeout": False,
        "not_found": False,
        "chunks": [_chunk(), _chunk("docs/other.md")],
    }
    body.update(over)
    return body


def _ask_body(**over: Any) -> dict[str, Any]:
    body = {
        "request_id": "r2",
        "status": "answered",
        "found": True,
        "degraded": False,
        "answer": "Use push_to_hub.",
        "sources": [_chunk()],
    }
    body.update(over)
    return body


def _config(tmp_path: Path, **over: Any) -> ProbeConfig:
    values: dict[str, Any] = {
        "base_url": "http://svc",
        "kb": "hf_docs",
        "query": "how to push a model",
        "expected": "push.md",
        "question": "how do I push a model to the Hub?",
        "interval_s": 60.0,
        "timeout_s": 5.0,
        "ask_enabled": False,
        "ask_interval_s": 1800.0,
        "ask_daily_cap": 2,
        "state_path": tmp_path / "state.json",
    }
    values.update(over)
    return ProbeConfig(**values)


def _prober(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    clock: Clock | None = None,
    **over: Any,
) -> Prober:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Prober(_config(tmp_path, **over), client=client, clock=clock or Clock())


def _ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/search_docs":
        return httpx.Response(200, json=_search_body())
    return httpx.Response(200, json=_ask_body())


def _gauge(obs: Any, name: str, kind: str) -> float | None:
    for attrs, point in metric_points(obs, name):
        if attrs.get("rag.probe.kind") == kind:
            return float(point.value)
    return None


def _runs(obs: Any) -> dict[tuple[str, str], int]:
    return {
        (a["rag.probe.kind"], a["rag.probe.outcome"]): p.value
        for a, p in metric_points(obs, "rag.probe.runs")
    }


def test_a_good_search_sets_success_latency_and_last_success(tmp_path: Path) -> None:
    clock = Clock()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        clock.now += 0.25  # the request takes a quarter of a second
        return _ok(request)

    with memory_observability() as obs:
        prober = _prober(tmp_path, handler, clock)
        result = prober.probe_search()
        assert result.outcome == "ok" and 0.2 < result.duration_s < 0.3
        assert _gauge(obs, "rag.probe.success", "search") == 1
        assert _gauge(obs, "rag.probe.last_success_timestamp", "search") == clock.now
        assert _runs(obs) == {("search", "ok"): 1}
        (point,) = [p for _, p in metric_points(obs, "rag.probe.duration")]
        assert point.count == 1 and 0.2 < point.sum < 0.3
    assert seen == [{"kb": "hf_docs", "query": "how to push a model", "top_k": 3}]


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (httpx.Response(503, json={"error": "not_ready"}), "error"),
        (httpx.Response(200, text="not json"), "error"),
        (httpx.Response(200, json=_search_body(chunks=[])), "insane"),
        (httpx.Response(200, json=_search_body(chunks=[_chunk("docs/other.md")])), "insane"),
        (httpx.Response(200, json=_search_body(degraded=True)), "insane"),
        (httpx.Response(200, json=_search_body(rerank_timeout=True)), "insane"),
        (httpx.Response(200, json=_search_body(not_found=True, chunks=[])), "insane"),
    ],
)
def test_a_bad_search_is_not_a_success(
    tmp_path: Path, response: httpx.Response, outcome: str
) -> None:
    with memory_observability() as obs:
        prober = _prober(tmp_path, lambda request: response)
        assert prober.probe_search().outcome == outcome
        assert _gauge(obs, "rag.probe.success", "search") == 0
        assert _gauge(obs, "rag.probe.last_success_timestamp", "search") is None
        assert _runs(obs) == {("search", outcome): 1}


def test_top_one_must_be_the_expected_page(tmp_path: Path) -> None:
    """The expected page in second place is not enough: the probe checks the first result."""
    body = _search_body(chunks=[_chunk("docs/other.md"), _chunk("docs/hub/push.md")])
    with memory_observability():
        prober = _prober(tmp_path, lambda request: httpx.Response(200, json=body))
        assert prober.probe_search().outcome == "insane"


def test_unreachable_service_and_timeout_are_errors_and_keep_the_last_success(
    tmp_path: Path,
) -> None:
    clock = Clock()
    state = {"mode": "ok"}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["mode"] == "refused":
            raise httpx.ConnectError("refused")
        if state["mode"] == "timeout":
            raise httpx.ReadTimeout("slow")
        return _ok(request)

    with memory_observability() as obs:
        prober = _prober(tmp_path, handler, clock)
        prober.probe_search()
        first_success = clock.now
        clock.now += 60
        state["mode"] = "refused"
        assert prober.probe_search().outcome == "error"
        clock.now += 60
        state["mode"] = "timeout"
        assert prober.probe_search().outcome == "error"
        assert _gauge(obs, "rag.probe.success", "search") == 0
        # The timestamp stays at the last good run: the alert measures how long ago it was.
        assert _gauge(obs, "rag.probe.last_success_timestamp", "search") == first_success
        assert _runs(obs) == {("search", "ok"): 1, ("search", "error"): 2}


def test_run_due_searches_every_interval(tmp_path: Path) -> None:
    clock = Clock()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _ok(request)

    with memory_observability():
        prober = _prober(tmp_path, handler, clock, interval_s=60.0)
        assert [r.kind for r in prober.run_due()] == ["search"]
        clock.now += 30
        assert prober.run_due() == []
        clock.now += 30
        assert [r.kind for r in prober.run_due()] == ["search"]
    assert calls == ["/v1/search_docs", "/v1/search_docs"]


def test_ask_probe_is_off_by_default(tmp_path: Path) -> None:
    clock = Clock()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _ok(request)

    with memory_observability():
        prober = _prober(tmp_path, handler, clock)  # ask_enabled False
        for _ in range(5):
            prober.run_due()
            clock.now += 3600
    assert "/v1/ask_docs" not in calls
    assert not (tmp_path / "state.json").exists()


def test_ask_probe_runs_at_its_interval_and_never_above_the_daily_cap(tmp_path: Path) -> None:
    clock = Clock(1_700_000_000.0)  # 2023-11-14 22:13 UTC
    asked: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/ask_docs":
            asked.append(clock.now)
        return _ok(request)

    with memory_observability() as obs:
        prober = _prober(tmp_path, handler, clock, ask_enabled=True, ask_daily_cap=2)
        for _ in range(8):  # 8 half-hour steps = 4 hours, across midnight UTC (at +6400 s)
            prober.run_due()
            clock.now += 1800
        assert _runs(obs)[("ask", "ok")] == len(asked)
    midnight = 1_700_006_400.0  # 2023-11-15 00:00 UTC
    # Day one: the first call and one half hour later, then the cap. Day two: the same again.
    assert [t - 1_700_000_000.0 for t in asked if t < midnight] == [0.0, 1800.0]
    assert [t - 1_700_000_000.0 for t in asked if t >= midnight] == [7200.0, 9000.0]


def test_the_cap_survives_a_restart_and_resets_the_next_day(tmp_path: Path) -> None:
    clock = Clock(1_700_000_000.0)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _ok(request)

    with memory_observability():
        first = _prober(tmp_path, handler, clock, ask_enabled=True, ask_daily_cap=1)
        first.run_due()
        assert calls.count("/v1/ask_docs") == 1
        clock.now += 1800
        restarted = _prober(tmp_path, handler, clock, ask_enabled=True, ask_daily_cap=1)
        restarted.run_due()
        assert calls.count("/v1/ask_docs") == 1  # the counter was read back from the state file
        clock.now += 86400
        restarted.run_due()
        assert calls.count("/v1/ask_docs") == 2  # a new UTC day


def test_the_cap_counts_a_failed_ask_too(tmp_path: Path) -> None:
    """A request that timed out may still have been paid for: it uses up one of the day's."""
    clock = Clock()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/ask_docs":
            raise httpx.ReadTimeout("slow")
        return _ok(request)

    with memory_observability() as obs:
        prober = _prober(tmp_path, handler, clock, ask_enabled=True, ask_daily_cap=1)
        prober.run_due()
        clock.now += 1800
        prober.run_due()
        assert calls.count("/v1/ask_docs") == 1
        assert _runs(obs)[("ask", "error")] == 1


@pytest.mark.parametrize(
    "body",
    [
        _ask_body(status="not_found", found=False, sources=[]),
        _ask_body(degraded=True),
        _ask_body(answer=""),
        _ask_body(sources=[_chunk("docs/other.md")]),
    ],
)
def test_ask_answers_that_are_empty_degraded_or_off_topic_are_insane(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    with memory_observability():
        prober = _prober(tmp_path, lambda request: httpx.Response(200, json=body), ask_enabled=True)
        assert prober.probe_ask().outcome == "insane"


def test_config_comes_from_settings(tmp_path: Path) -> None:
    from rag.testing import observe_settings

    settings = observe_settings(service_port=29001, probe_kb="hf_docs", probe_interval_s=30)
    config = ProbeConfig.from_settings(settings)
    assert config.base_url == "http://127.0.0.1:29001"
    assert config.interval_s == 30 and config.ask_enabled is False
    assert config.ask_daily_cap == settings.probe_ask_daily_cap
