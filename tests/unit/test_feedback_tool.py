"""Spec P5-02 AC2: the `feedback` MCP tool and the feedback counters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.serve.mcp_server import build_mcp_server
from rag.serve.service import SearchService
from rag.testing import memory_observability, metric_points
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _service(records: Path) -> SearchService:
    from rag.testing import observe_settings

    configs = {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
        )
    }
    builder = FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")})
    service = SearchService(
        observe_settings(business_record_path=str(records)), configs=configs, builder=builder
    )
    service.load()
    return service


def _run(service: SearchService, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    async def main() -> list[Any]:
        out: list[Any] = []
        async with Client(build_mcp_server(service)) as client:
            out.append(await client.list_tools())
            for name, arguments in calls:
                out.append(await client.call_tool(name, arguments))
        return out

    return anyio.run(main)


def test_feedback_tool_is_listed_with_its_arguments(tmp_path: Path) -> None:
    (listing,) = _run(_service(tmp_path / "r.jsonl"), [])
    tool = {t.name: t for t in listing.tools}["feedback"]
    assert set(tool.input_schema["properties"]) == {"request_id", "useful", "comment"}
    assert set(tool.input_schema["required"]) == {"request_id", "useful"}
    assert "request_id" in tool.description and "search_docs" in tool.description
    assert tool.annotations is not None and tool.annotations.read_only_hint is False


def test_feedback_tool_writes_one_record_keyed_by_the_rated_request(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        _, ok = _run(
            _service(path),
            [("feedback", {"request_id": "req-42", "useful": False, "comment": "  wrong  "})],
        )
    assert not ok.is_error
    assert ok.structured_content == {"request_id": "req-42", "recorded": True}
    (line,) = path.read_text("utf-8").splitlines()
    row = json.loads(line)
    assert row["request_id"] == "req-42" and row["tool"] == "feedback"
    assert row["extra"]["useful"] is False and row["extra"]["comment"] == "wrong"
    assert row["extra"]["feedback_request_id"] != "req-42"  # the feedback call has its own id


BAD: list[dict[str, Any]] = [
    {"request_id": "has space", "useful": True},
    {"request_id": "a" * 65, "useful": True},
    {"request_id": "abc\n", "useful": True},
    {"request_id": "", "useful": True},
    {"request_id": "r1", "useful": True, "comment": "x" * 501},
    {"request_id": "r1", "useful": "yes"},  # a string is not a verdict
    {"request_id": "r1", "useful": 1},  # nor is a number
    {"request_id": "r1"},
    {"useful": True},
    {"request_id": "r1", "useful": True, "comment": 5},
]


@pytest.mark.parametrize("arguments", BAD)
def test_feedback_tool_rejects_what_the_http_endpoint_rejects(
    arguments: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)) as obs:
        _, result = _run(_service(path), [("feedback", arguments)])
        counted = metric_points(obs, "rag.feedback")
    assert result.is_error
    assert counted == []  # a refusal is not a verdict
    rows = (
        [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []
    )
    assert not any(row["extra"].get("useful") is not None for row in rows)


def test_feedback_comment_of_the_limit_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)):
        _, ok = _run(
            _service(path),
            [("feedback", {"request_id": "r1", "useful": True, "comment": "x" * 500})],
        )
    assert not ok.is_error
    assert len(json.loads(path.read_text("utf-8"))["extra"]["comment"]) == 500


def test_feedback_counts_verdicts(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with memory_observability(business_record_path=str(path)) as obs:
        service = _service(path)
        _run(
            service,
            [
                ("feedback", {"request_id": "a", "useful": True}),
                ("feedback", {"request_id": "b", "useful": True}),
                ("feedback", {"request_id": "c", "useful": False}),
            ],
        )
        points = {
            p[0]["rag.feedback.verdict"]: p[1].value for p in metric_points(obs, "rag.feedback")
        }
    assert points == {"useful": 2, "not_useful": 1}
