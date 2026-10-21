"""Spec P0-09 AC12: the MCP tools (mcp 2.3 `MCPServer`) through the SDK's in-process client."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from mcp import Client

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.security.access import Credentials
from rag.serve.mcp_server import build_mcp_server
from rag.serve.schema import ContextResponse, KnowledgeBaseList, SearchResponse
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

TOOLS = {"search_docs", "ask_docs", "get_context", "list_knowledge_bases", "feedback"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _service() -> SearchService:
    configs = {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs",
            description="Hugging Face library documentation (Transformers, Datasets, Hub).",
            language="en",
            access=AccessConfig(callers=["*"]),
        ),
        "secret_kb": KnowledgeBaseConfig(
            kb="secret_kb", description="Someone else's pages", access=AccessConfig(callers=["bob"])
        ),
    }
    builder = FakePipelineBuilder({kb: FakeSearchPipeline(kb) for kb in configs})
    service = SearchService(observe_settings(), configs=configs, builder=builder)
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


def _payload(result: Any) -> dict[str, Any]:
    """Structured content; the text block carries the same JSON for text-only clients."""
    (block,) = result.content
    assert json.loads(block.text) == result.structured_content
    return result.structured_content


def test_tools_listed_with_kb_descriptions() -> None:
    (listing,) = _run(_service(), [])
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == TOOLS
    search = tools["search_docs"]
    assert "hf_docs" in search.description
    assert "Hugging Face library documentation" in search.description
    assert "secret_kb" not in search.description  # only KBs the server's caller may use
    assert "untrusted" in search.description.lower()
    assert set(search.input_schema["properties"]) == {
        "kb",
        "query",
        "filters",
        "top_k",
        "deadline_ms",
    }
    assert set(search.input_schema["required"]) == {"kb", "query"}
    assert "doc_type" in search.input_schema["properties"]["filters"]["description"]
    assert "chunks" in search.output_schema["properties"]
    assert "routes" not in search.output_schema["properties"]
    context = tools["get_context"]
    assert set(context.input_schema["properties"]) == {"chunk_id", "mode"}
    assert "updated" in context.description
    assert "hf_docs" in tools["list_knowledge_bases"].description


def test_mcp_tools_return_structured_results() -> None:
    service = _service()
    _, search, context, listing = _run(
        service,
        [
            ("search_docs", {"kb": "hf_docs", "query": "how to log?", "top_k": 2}),
            ("get_context", {"chunk_id": "hf_docs:p1.md#0", "mode": "neighbors"}),
            ("list_knowledge_bases", {}),
        ],
    )
    assert not search.is_error and not context.is_error and not listing.is_error
    via_mcp = SearchResponse.model_validate(_payload(search))
    direct = service.search_docs(
        Credentials(transport="http"), kb="hf_docs", query="how to log?", top_k=2
    )
    ignore = {"request_id", "timings_ms"}
    assert via_mcp.model_dump(exclude=ignore) == direct.model_dump(exclude=ignore)
    assert ContextResponse.model_validate(_payload(context)).mode == "neighbors"
    kbs = KnowledgeBaseList.model_validate(_payload(listing)).knowledge_bases
    assert [k.kb for k in kbs] == ["hf_docs"]
    pipeline = service.pipeline("hf_docs")
    assert isinstance(pipeline, FakeSearchPipeline)
    assert pipeline.calls[0]["acl_groups"] == ("public",)


def test_mcp_errors_are_category_and_request_id() -> None:
    service = _service()
    pipeline = service.pipeline("secret_kb")
    assert isinstance(pipeline, FakeSearchPipeline)
    hf = service.pipeline("hf_docs")
    assert isinstance(hf, FakeSearchPipeline)
    hf.context_raises = RuntimeError("secret detail C:/x")
    _, bad_filter, hidden, crashed = _run(
        service,
        [
            ("search_docs", {"kb": "hf_docs", "query": "q", "filters": {"acl_groups": ["x"]}}),
            ("search_docs", {"kb": "secret_kb", "query": "q"}),
            ("get_context", {"chunk_id": "hf_docs:p1.md#0"}),
        ],
    )
    for result, category in (
        (bad_filter, "invalid_filter"),
        (hidden, "kb_not_found"),
        (crashed, "internal_error"),
    ):
        assert result.is_error
        payload = _payload(result)
        assert set(payload) == {"error", "request_id", "message"}
        assert payload["error"] == category and len(payload["request_id"]) == 32
    assert "secret" not in crashed.content[0].text
    assert pipeline.calls == []
