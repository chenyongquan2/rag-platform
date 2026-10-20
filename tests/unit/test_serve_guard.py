"""Spec P3-04 review round 1: the HTTP routes accept only JSON bodies and only the Host names
the service is meant to answer to, so a page on another site cannot make a developer's browser
call the local service (cross-site requests send `text/plain` or form types without a preflight;
a rebound DNS name arrives with its own Host header)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder

BODY = {"kb": "hf_docs", "question": "How do I push a model?"}
ERROR_KEYS = {"error", "request_id", "message"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _client(*, host: str = "127.0.0.1", **settings: Any) -> TestClient:
    service = SearchService(
        agent_settings(**settings),
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", description="d", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": ScriptedSearchBackend("hf_docs")}),
        answer_model=StreamingFakeChatModel(default_response="Fine [1]."),
    )
    service.load()
    return TestClient(build_http_app(service, host=host), base_url="http://localhost:28080")


POST_ROUTES = ["/v1/ask_docs", "/v1/ask_docs/stream", "/v1/search_docs", "/v1/get_context"]


@pytest.mark.parametrize("route", [*POST_ROUTES, "/v1/feedback"])
def test_a_post_without_a_json_content_type_is_refused(route: str) -> None:
    with _client() as client:
        for headers in (
            {},
            {"Content-Type": "text/plain"},
            {"Content-Type": "multipart/form-data"},
            {"Content-Type": "application/x-www-form-urlencoded"},
            {"Content-Type": "application/jsonx"},
            {"Content-Type": "text/json"},
            {"Content-Type": "application/json-seq"},
        ):
            r = client.post(route, content=b'{"kb": "hf_docs", "question": "q"}', headers=headers)
            assert r.status_code == 415, (route, headers)
            assert set(r.json()) == ERROR_KEYS and r.json()["error"] == "invalid_request"
            assert r.headers["x-request-id"] == r.json()["request_id"]
        ok = client.post(
            route, content=b"{}", headers={"Content-Type": "application/json; charset=utf-8"}
        )
        assert ok.status_code != 415  # reaches the route (which then judges the body)


def test_a_stream_request_without_a_content_type_runs_nothing() -> None:
    backend = ScriptedSearchBackend("hf_docs")
    service = SearchService(
        agent_settings(),
        configs={"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=StreamingFakeChatModel(default_response="Fine [1]."),
    )
    service.load()
    with TestClient(build_http_app(service), base_url="http://localhost:1") as client:
        r = client.post("/v1/ask_docs/stream", content=b'{"kb":"hf_docs","question":"q"}')
    assert r.status_code == 415 and backend.calls == []


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:28080",
        "LOCALHOST:1",
        "127.0.0.1",
        "127.0.0.1:28080",
        "[::1]",
        "[::1]:28080",
        "localhost.",
    ],
)
def test_the_local_host_names_are_answered(host: str) -> None:
    with _client() as client:
        for path in ("/healthz", "/", "/v1/knowledge_bases", "/ui/app.js"):
            r = client.get(path, headers={"Host": host})
            assert r.status_code == 200, (path, host)
        r = client.post("/v1/ask_docs", json=BODY, headers={"Host": host})
        assert r.status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:28080",
        "localhost.evil.example",
        "127.0.0.1.evil.example",
        "127.0.0.2",
        "[::2]",
        "",
        "localhost:80:80",
        "localhost@evil.example",
        "0.0.0.0",
    ],
)
def test_another_host_name_is_refused_on_every_route(host: str) -> None:
    with _client() as client:
        for method, path in (
            ("GET", "/"),
            ("GET", "/healthz"),
            ("GET", "/readyz"),
            ("GET", "/ui/app.js"),
            ("GET", "/v1/knowledge_bases"),
            ("POST", "/v1/ask_docs"),
            ("POST", "/v1/ask_docs/stream"),
            ("POST", "/v1/feedback"),
        ):
            r = client.request(
                method, path, json=BODY if method == "POST" else None, headers={"Host": host}
            )
            assert r.status_code == 421, (method, path, host)
            assert set(r.json()) == ERROR_KEYS and r.json()["error"] == "invalid_request"


def test_configured_host_names_and_the_listen_address_are_answered() -> None:
    with _client(service_allowed_hosts=["docs.example.com"]) as client:
        assert client.get("/healthz", headers={"Host": "docs.example.com"}).status_code == 200
        assert client.get("/healthz", headers={"Host": "DOCS.example.com:8443"}).status_code == 200
        assert client.get("/healthz", headers={"Host": "other.example.com"}).status_code == 421
    # Listening on a named interface: that address is a name the service answers to.
    with _client(host="10.1.2.3") as client:
        assert client.get("/healthz", headers={"Host": "10.1.2.3:28080"}).status_code == 200
    # Listening on every interface says nothing about names: only the configured ones.
    with _client(host="0.0.0.0") as client:
        assert client.get("/healthz", headers={"Host": "0.0.0.0:28080"}).status_code == 421
        assert client.get("/healthz", headers={"Host": "localhost"}).status_code == 200


def test_the_allowed_hosts_setting_is_checked() -> None:
    from rag.config import ServiceSettings

    assert ServiceSettings(_env_file=None).service_allowed_hosts == []
    for bad in ([""], ["has space"], ["a/b"], ["http://x"]):
        with pytest.raises(ValueError):
            agent_settings(service_allowed_hosts=bad)
