"""Spec P3-01 AC16, AC17: the `ask_docs` MCP tool and `POST /v1/ask_docs`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from mcp import Client
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.serve.http_app import build_http_app
from rag.serve.mcp_server import build_mcp_server
from rag.serve.schema import AskResponse
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder

REPLY = "Use push_to_hub [1]. Or upload_folder [2]."
ERROR_KEYS = {"error", "request_id", "message"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _service() -> tuple[SearchService, ScriptedSearchBackend]:
    configs = {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs",
            description="Hugging Face library documentation.",
            language="en",
            access=AccessConfig(callers=["*"]),
        ),
        "secret_kb": KnowledgeBaseConfig(
            kb="secret_kb", description="Someone else's pages", access=AccessConfig(callers=["bob"])
        ),
    }
    backend = ScriptedSearchBackend("hf_docs")
    service = SearchService(
        agent_settings(),
        configs=configs,
        builder=FakePipelineBuilder({"hf_docs": backend, "secret_kb": ScriptedSearchBackend("s")}),
        answer_model=StreamingFakeChatModel(default_response=REPLY),
    )
    service.load()
    return service, backend


def _payload(result: Any) -> dict[str, Any]:
    (block,) = result.content
    assert json.loads(block.text) == result.structured_content
    return result.structured_content


def test_mcp_ask_docs() -> None:
    service, backend = _service()

    async def main() -> list[Any]:
        out: list[Any] = []
        async with Client(build_mcp_server(service)) as client:
            out.append(await client.list_tools())
            out.append(
                await client.call_tool(
                    "ask_docs",
                    {"kb": "hf_docs", "question": "How do I push a model?", "session_id": "s1"},
                )
            )
            out.append(await client.call_tool("ask_docs", {"kb": "secret_kb", "question": "q"}))
            out.append(await client.call_tool("ask_docs", {"kb": "hf_docs", "question": " "}))
        return out

    listing, answered, hidden, empty = anyio.run(main)
    tool = {t.name: t for t in listing.tools}["ask_docs"]
    assert set(tool.input_schema["properties"]) == {
        "kb",
        "question",
        "session_id",
        "deadline_ms",
        "resume_token",
        "answer",
    }
    assert set(tool.input_schema["required"]) == {"kb", "question"}
    assert "hf_docs" in tool.description and "secret_kb" not in tool.description
    assert "untrusted" in tool.description.lower()
    assert "no answer was written" in tool.description  # not_found with a flag or a category
    assert {"answer", "sources", "status", "rounds", "coverage_unconfirmed"} <= set(
        tool.output_schema["properties"]
    )
    # A read-only hint would be wrong: the call stores the turn in the session's history.
    assert tool.annotations is not None and tool.annotations.read_only_hint is False

    assert not answered.is_error
    response = AskResponse.model_validate(_payload(answered))
    assert response.status == "answered" and response.answer == "Use push_to_hub. Or upload_folder."
    assert response.session_id == "s1" and response.rounds == 1 and response.found
    assert backend.calls[0]["acl_groups"] == ("public",)  # the same caller handling as search_docs
    for result, category in ((hidden, "kb_not_found"), (empty, "invalid_request")):
        assert result.is_error
        body = _payload(result)
        assert set(body) == ERROR_KEYS and body["error"] == category


def test_http_ask_docs() -> None:
    service, backend = _service()
    with TestClient(build_http_app(service)) as client:
        assert service.wait_ready(5)
        r = client.post(
            "/v1/ask_docs",
            json={"kb": "hf_docs", "question": "How do I push a model?", "session_id": "s1"},
            headers={"X-Request-Id": "abc-1"},
        )
        assert r.status_code == 200, r.text
        response = AskResponse.model_validate(r.json())
        assert response.request_id == "abc-1" and r.headers["x-request-id"] == "abc-1"
        assert response.answer == "Use push_to_hub. Or upload_folder."
        assert [c.untrusted for c in response.sources] == [True, True, True]
        assert response.thread_id.startswith("s1:") and response.cache_allowed
        assert "routes" not in r.json() and "chunks" not in r.json()
        r2 = client.post(
            "/v1/ask_docs",
            json={"kb": "hf_docs", "question": "and the second?", "session_id": "s1"},
        )
        assert r2.json()["used_history"] is True and r2.json()["cache_allowed"] is False
        assert len(backend.calls) == 2

        def error(r: Any, status: int, category: str) -> None:
            assert r.status_code == status, r.text
            body = r.json()
            assert set(body) == ERROR_KEYS and body["error"] == category
            assert r.headers["x-request-id"] == body["request_id"]

        error(
            client.post("/v1/ask_docs", json={"kb": "hf_docs", "question": "q", "top_k": 3}),
            400,
            "invalid_request",
        )  # unknown body field
        error(
            client.post(
                "/v1/ask_docs", content=b"{not json", headers={"Content-Type": "application/json"}
            ),
            400,
            "invalid_request",
        )
        error(client.post("/v1/ask_docs", json={"kb": "hf_docs"}), 400, "invalid_request")
        error(
            client.post(
                "/v1/ask_docs", json={"kb": "hf_docs", "question": "q", "session_id": "a:b"}
            ),
            400,
            "invalid_request",
        )
        error(
            client.post("/v1/ask_docs", json={"kb": "secret_kb", "question": "q"}),
            404,
            "kb_not_found",
        )
        backend.raises = RuntimeError("secret detail C:/x")
        r = client.post("/v1/ask_docs", json={"kb": "hf_docs", "question": "q"})
        error(r, 500, "internal_error")
        assert "secret" not in r.text


# ---- P3-02: the question asked back, over both transports -------------------------------------

VAGUE = "How do I fix this error?"
REPLY_TO_QUESTION = "transformers raises KeyError 'llama'"


def _asking_service(caller: str = "local", agent: Any = None) -> SearchService:
    from rag.config import AgentConfig

    backend = ScriptedSearchBackend("hf_docs")
    service = SearchService(
        agent_settings(service_caller=caller),
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs",
                description="Hugging Face library documentation.",
                language="en",
                access=AccessConfig(callers=["*"]),
                agent=AgentConfig(clarify=True),
            )
        },
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=StreamingFakeChatModel(default_response=REPLY),
        agent=agent,
    )
    service.load()
    return service


def test_clarification_over_mcp_and_http() -> None:
    service = _asking_service()

    async def mcp() -> tuple[Any, Any, Any]:
        async with Client(build_mcp_server(service)) as client:
            asked = await client.call_tool(
                "ask_docs", {"kb": "hf_docs", "question": VAGUE, "session_id": "m1"}
            )
            token = _payload(asked)["resume_token"]
            args = {"kb": "hf_docs", "question": VAGUE, "session_id": "m1"}
            answered = await client.call_tool(
                "ask_docs", {**args, "resume_token": token, "answer": REPLY_TO_QUESTION}
            )
            reused = await client.call_tool(
                "ask_docs", {**args, "resume_token": token, "answer": REPLY_TO_QUESTION}
            )
            return asked, answered, reused

    asked, answered, reused = anyio.run(mcp)
    first = AskResponse.model_validate(_payload(asked))
    assert first.status == "need_clarification" and first.resume_token and first.sources == []
    assert "Could you be more specific?" in first.answer
    second = AskResponse.model_validate(_payload(answered))
    assert (second.status, second.clarified, second.resume_token) == ("answered", True, None)
    assert second.answer == "Use push_to_hub. Or upload_folder." and second.rounds == 1
    assert reused.is_error and _payload(reused)["error"] == "invalid_resume_token"
    assert set(_payload(reused)) == ERROR_KEYS
    assert first.resume_token not in json.dumps(_payload(reused))

    service = _asking_service(agent=service.agent)  # the same process, a fresh service object
    other = _asking_service(caller="mallory", agent=service.agent)
    with TestClient(build_http_app(service)) as client, TestClient(build_http_app(other)) as evil:
        assert service.wait_ready(5) and other.wait_ready(5)
        body = {"kb": "hf_docs", "question": VAGUE, "session_id": "h1"}
        r = client.post("/v1/ask_docs", json=body, headers={"X-Request-Id": "h-1"})
        assert r.status_code == 200 and r.json()["status"] == "need_clarification"
        token = r.json()["resume_token"]
        assert token and r.json()["answer"] and r.json()["sources"] == []
        resume = {**body, "resume_token": token, "answer": REPLY_TO_QUESTION}

        stolen = evil.post("/v1/ask_docs", json=resume)  # another caller's token is no use
        assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_resume_token"
        assert set(stolen.json()) == ERROR_KEYS and token not in stolen.text
        wrong = client.post("/v1/ask_docs", json={**resume, "session_id": "h2"})
        assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_resume_token"
        no_answer = client.post("/v1/ask_docs", json={**body, "resume_token": token})
        assert no_answer.status_code == 400 and no_answer.json()["error"] == "invalid_request"

        done = client.post("/v1/ask_docs", json=resume, headers={"X-Request-Id": "h-2"})
        assert done.status_code == 200, done.text
        response = AskResponse.model_validate(done.json())
        assert response.status == "answered" and response.clarified and response.request_id == "h-2"
        again = client.post("/v1/ask_docs", json=resume)
        assert again.status_code == 400 and again.json()["error"] == "invalid_resume_token"
