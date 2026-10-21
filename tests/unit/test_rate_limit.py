"""Spec P8-01 AC4: requests per caller per minute (design s16-4).

A fixed one-minute window: the counter of `caller` in minute `floor(t / 60)`. The memory counter
and a fake Redis client stand in for Redis here; tests/integration/test_rate_limit_redis.py uses
the real one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.security.access import Credentials
from rag.serve.errors import ServiceError
from rag.serve.http_app import build_http_app
from rag.serve.limits import (
    MemoryWindowCounter,
    RateLimiter,
    RedisWindowCounter,
    make_rate_limiter,
)
from rag.serve.mcp_server import build_mcp_server
from rag.serve.service import SearchService
from rag.testing import memory_observability, observe_settings
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline
from rag.types import ErrorCategory


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def limiter(per_minute: int = 3, callers: dict[str, int] | None = None, t: float = 6030.0):
    clock = Clock(t)  # 6030 s = minute 100, 30 s in
    return RateLimiter(
        MemoryWindowCounter(), per_minute=per_minute, per_caller=callers or {}, clock=clock
    ), clock


def test_up_to_the_limit_then_refused_with_the_seconds_left_in_the_minute() -> None:
    rl, _ = limiter(per_minute=3)
    results = [rl.admit("alice") for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].retry_after_s == pytest.approx(30.0)
    assert results[-1].limit == 3 and results[-1].count == 4


def test_the_window_resets_in_the_next_minute() -> None:
    rl, clock = limiter(per_minute=1)
    assert rl.admit("alice").allowed and not rl.admit("alice").allowed
    clock.t += 30.0  # minute 101
    assert rl.admit("alice").allowed


def test_other_callers_are_not_affected() -> None:
    rl, _ = limiter(per_minute=1)
    assert rl.admit("alice").allowed and not rl.admit("alice").allowed
    assert rl.admit("bob").allowed


def test_per_caller_limits_and_zero_means_no_limit() -> None:
    rl, _ = limiter(per_minute=1, callers={"batch": 5, "probe": 0})
    assert sum(rl.admit("batch").allowed for _ in range(7)) == 5
    assert all(rl.admit("probe").allowed for _ in range(50))


class BrokenCounter:
    def incr(self, caller: str, window: int, ttl_s: int) -> int:
        raise ConnectionError("redis is down")


def test_a_counter_error_lets_the_request_through_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rl = RateLimiter(BrokenCounter(), per_minute=1, per_caller={}, clock=Clock(0.0))
    with caplog.at_level(logging.WARNING):
        results = [rl.admit("alice") for _ in range(5)]
    assert all(r.allowed for r in results) and all(r.counter_error for r in results)
    assert "rate limit counter failed" in caplog.text
    assert caplog.text.count("rate limit counter failed") == 1  # one line, not one per request


class FakePipe:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self.ops: list[tuple[str, Any]] = []

    def incr(self, key: str) -> FakePipe:
        self.ops.append(("incr", key))
        return self

    def expire(self, key: str, ttl: int) -> FakePipe:
        self.ops.append(("expire", (key, ttl)))
        return self

    def execute(self) -> list[Any]:
        out: list[Any] = []
        for op, arg in self.ops:
            if op == "incr":
                self.client.values[arg] = self.client.values.get(arg, 0) + 1
                out.append(self.client.values[arg])
            else:
                self.client.ttls[arg[0]] = arg[1]
                out.append(True)
        return out


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> FakePipe:
        return FakePipe(self)


def test_redis_counter_uses_the_prefix_and_expires_its_keys() -> None:
    redis = FakeRedis()
    rl = RateLimiter(
        RedisWindowCounter(redis, prefix="p801:rl:"), per_minute=2, per_caller={}, clock=Clock(6030)
    )
    assert [rl.admit("alice").allowed for _ in range(3)] == [True, True, False]
    assert redis.values == {"p801:rl:alice:100": 3}
    assert redis.ttls == {"p801:rl:alice:100": 120}


def test_make_rate_limiter_follows_the_settings() -> None:
    assert make_rate_limiter(observe_settings(rate_limit_mode="off")) is None
    assert isinstance(
        make_rate_limiter(observe_settings(rate_limit_mode="memory")).counter, MemoryWindowCounter
    )
    redis = make_rate_limiter(observe_settings(rate_limit_mode="redis", rate_limit_key_prefix="x:"))
    assert isinstance(redis.counter, RedisWindowCounter) and redis.counter.prefix == "x:"


# ---- in the service, over HTTP and MCP -------------------------------------------------------


def _service(tmp_path: Path, per_minute: int = 2) -> tuple[SearchService, FakeSearchPipeline]:
    configs = {"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))}
    pipeline = FakeSearchPipeline("hf_docs")
    settings = observe_settings(business_record_path=str(tmp_path / "records.jsonl"))
    rl, _ = limiter(per_minute=per_minute)
    service = SearchService(
        settings, configs=configs, builder=FakePipelineBuilder({"hf_docs": pipeline}), limiter=rl
    )
    return service, pipeline


def test_http_answers_429_with_retry_after_and_records_it(tmp_path: Path) -> None:
    service, pipeline = _service(tmp_path)
    records_path = str(tmp_path / "records.jsonl")
    with (
        memory_observability(business_record_path=records_path),
        TestClient(build_http_app(service)) as client,
    ):
        assert service.wait_ready(5)
        body = {"kb": "hf_docs", "query": "q"}
        assert [client.post("/v1/search_docs", json=body).status_code for _ in range(2)] == [
            200,
            200,
        ]
        r = client.post("/v1/search_docs", json=body)
        assert r.status_code == 429 and r.json()["error"] == "rate_limited"
        assert r.headers["retry-after"] == "30"
        # get_context and the stream count against the same minute.
        r = client.post("/v1/get_context", json={"chunk_id": "hf_docs:p1.md#0"})
        assert r.status_code == 429
        r = client.post("/v1/ask_docs/stream", json={"kb": "hf_docs", "question": "q"})
        assert r.status_code == 429 and r.headers["retry-after"] == "30"
        # list_knowledge_bases is free.
        assert client.get("/v1/knowledge_bases").status_code == 200
    assert len(pipeline.calls) == 2
    records = [json.loads(x) for x in (tmp_path / "records.jsonl").read_text("utf-8").splitlines()]
    assert sum(r["flags"] == ["rate_limited"] for r in records) == 3


def test_mcp_tool_gets_an_error_result(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, per_minute=1)
    service.load()

    async def main() -> list[Any]:
        async with Client(build_mcp_server(service)) as client:
            return [
                await client.call_tool("search_docs", {"kb": "hf_docs", "query": "q"})
                for _ in range(2)
            ]

    ok, refused = anyio.run(main)
    assert not ok.is_error
    assert refused.is_error and refused.structured_content["error"] == "rate_limited"


def test_ask_docs_is_limited_before_the_graph_runs(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend("hf_docs")
    rl, _ = limiter(per_minute=0, callers={"local": 1})
    service = SearchService(
        agent_settings(),
        configs={"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        limiter=rl,
    )
    service.load()
    creds = Credentials(transport="http")
    assert rl.admit("local").allowed  # the caller's one request of this minute
    with pytest.raises(ServiceError) as info:
        asyncio.run(service.ask_docs(creds, kb="hf_docs", question="push a model?"))
    assert info.value.category == ErrorCategory.RATE_LIMITED
    assert info.value.retry_after_s == pytest.approx(30.0)
    assert backend.calls == []
