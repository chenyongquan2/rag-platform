"""Spec P0-11: metric instruments and the cost-ledger hook."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.config import ModelPrice
from rag.cost import CostLedger
from rag.observe import attributes as attr
from rag.observe import layer_span, request_span, shutdown_observability
from rag.observe.metrics import STEP_DURATION_BUCKETS_S
from rag.testing import memory_observability, metric_points
from rag.types import RetrievalResult


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _sums(points: list[tuple[dict, object]], *keys: str) -> dict[tuple, float]:
    return {tuple(a.get(k) for k in keys): p.value for a, p in points}  # type: ignore[attr-defined]


def test_request_metrics_from_result() -> None:
    with memory_observability() as obs:
        with request_span("search_docs", kb="hf_docs") as scope:
            scope.set_result(RetrievalResult(request_id=scope.request_id, kb="hf_docs"))
        with request_span("search_docs", kb="hf_docs") as scope:
            scope.set_result(
                RetrievalResult(
                    request_id=scope.request_id,
                    kb="hf_docs",
                    index_version="kb_hf_docs_v1",
                    data_generation=3,
                    degraded=True,
                    rerank_timeout=True,
                )
            )
        with (
            pytest.raises(ConnectionError),
            request_span("search_docs", kb="hf_docs", classify=lambda e: "es_unavailable"),
        ):
            raise ConnectionError("es down")
        with request_span("ask_docs", kb="other"):
            pass

        requests = _sums(metric_points(obs, attr.METRIC_REQUESTS), attr.KB, attr.TOOL, attr.OUTCOME)
        errors = _sums(metric_points(obs, attr.METRIC_ERRORS), attr.KB, attr.ERROR_CATEGORY)
        degraded = _sums(metric_points(obs, attr.METRIC_DEGRADED), attr.KB, attr.TOOL)
        timeouts = _sums(metric_points(obs, attr.METRIC_RERANK_TIMEOUTS), attr.KB, attr.TOOL)
        spans = obs.span_exporter.get_finished_spans()

    assert requests == {
        ("hf_docs", "search_docs", "ok"): 2,
        ("hf_docs", "search_docs", "error"): 1,
        ("other", "ask_docs", "ok"): 1,
    }
    assert errors == {
        ("hf_docs", "degraded"): 1,
        ("hf_docs", "rerank_timeout"): 1,
        ("hf_docs", "es_unavailable"): 1,
    }
    assert degraded == {("hf_docs", "search_docs"): 1}
    assert timeouts == {("hf_docs", "search_docs"): 1}

    flagged = spans[1]
    assert flagged.attributes[attr.DEGRADED] is True
    assert flagged.attributes[attr.RERANK_TIMEOUT] is True
    assert tuple(flagged.attributes[attr.ERROR_CATEGORIES]) == ("degraded", "rerank_timeout")
    assert flagged.attributes[attr.INDEX_VERSION] == "kb_hf_docs_v1"
    assert flagged.attributes[attr.DATA_GENERATION] == 3


def test_step_latency_histogram() -> None:
    with memory_observability() as obs:
        with request_span("search_docs", kb="k"):
            with layer_span("rerank"):
                time.sleep(0.02)
            with pytest.raises(TimeoutError), layer_span("rerank"):
                raise TimeoutError
        points = metric_points(obs, attr.METRIC_STEP_DURATION)

    by_key = {(a[attr.STEP], a[attr.OUTCOME]): p for a, p in points}
    assert set(by_key) == {("rerank", "ok"), ("rerank", "error"), ("search_docs", "ok")}
    ok = by_key[("rerank", "ok")]
    assert ok.count == 1
    assert ok.sum >= 0.015  # seconds
    assert tuple(ok.explicit_bounds) == STEP_DURATION_BUCKETS_S
    assert all(a[attr.KB] == "k" for a, _ in points)


def _ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(
        tmp_path / "ledger.jsonl",
        prices={"m": ModelPrice(input_cny_per_m=1.0, output_cny_per_m=2.0)},
        budget_stop_cny=10.0,
    )


def test_ledger_rows_feed_token_and_cost_counters(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with memory_observability() as obs:
        ledger.record(
            kind="chat",
            model="m",
            prompt_tokens=1000,
            completion_tokens=500,
            profile="online",
            usage_estimated=False,
        )
        ledger.record(
            kind="embedding", model="m", prompt_tokens=2000, profile="query", usage_estimated=True
        )
        tokens = _sums(
            metric_points(obs, attr.METRIC_MODEL_TOKENS),
            attr.MODEL_KIND,
            attr.GEN_AI_REQUEST_MODEL,
            attr.GEN_AI_TOKEN_TYPE,
            attr.USAGE_ESTIMATED,
        )
        cost = _sums(
            metric_points(obs, attr.METRIC_MODEL_COST),
            attr.MODEL_KIND,
            attr.GEN_AI_REQUEST_MODEL,
            attr.MODEL_PROFILE,
        )
    assert tokens == {
        ("chat", "m", "input", False): 1000,
        ("chat", "m", "output", False): 500,
        ("embedding", "m", "input", True): 2000,
    }
    assert cost == pytest.approx(
        {("chat", "m", "online"): 0.002, ("embedding", "m", "query"): 0.002}
    )
    assert len(ledger.entries()) == 2
