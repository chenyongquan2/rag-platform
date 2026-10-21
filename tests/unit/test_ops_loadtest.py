"""Spec P8-01 AC8: the load-test runner (percentiles, counting window, refusal of real asks)."""

from __future__ import annotations

import asyncio
import math

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rag.ops.loadtest import main, markdown, percentile, run_level


def test_percentile_interpolates() -> None:
    assert percentile([], 50) != percentile([], 50)  # NaN
    assert percentile([10.0], 95) == 10.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def _app() -> Starlette:
    seen = {"n": 0}

    async def search(request: Request) -> JSONResponse:
        seen["n"] += 1
        await asyncio.sleep(0.01)
        if seen["n"] % 10 == 0:
            return JSONResponse({"error": "rate_limited"}, status_code=429)
        return JSONResponse({"degraded": seen["n"] % 5 == 0, "rerank_timeout": False})

    return Starlette(routes=[Route("/v1/search_docs", search, methods=["POST"])])


def test_run_level_counts_statuses_flags_and_throughput() -> None:
    async def go() -> object:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await run_level(
                client,
                tool="search_docs",
                kb="demo_docs",
                questions=["a", "b"],
                concurrency=4,
                duration_s=0.5,
                warmup_s=0.1,
                measure_cpu=False,
            )

    result = asyncio.run(go())
    assert result.requests > 20 and result.ok < result.requests
    assert result.statuses.get("429", 0) > 0
    assert result.degraded > 0
    assert result.throughput_rps == pytest.approx(result.ok / 0.5)
    assert 10 <= result.p50_ms <= result.p95_ms <= result.max_ms
    table = markdown([result], target_p95_ms=1000)
    assert "| 4 |" in table and "Highest concurrency with P95 <= 1000 ms: 4" in table
    assert not math.isnan(result.p99_ms)


def test_ask_docs_is_refused_for_a_real_kb(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--url", "http://127.0.0.1:9", "--tool", "ask_docs", "--kb", "hf_docs"]) == 2
    assert "dev-fake" in capsys.readouterr().err
