"""Spec P8-01 AC1, AC2: OAuth bearer tokens on the HTTP API and on MCP streamable-http, scopes
mapped to callers, stdio and local mode unchanged (design s16-4, s16-1).

The MCP tests run a real uvicorn server on a free port (the SDK's client speaks HTTP), with the
SDK's own `TokenVerifier` hook; the HTTP tests use Starlette's test client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.testclient import TestClient

from rag.config import AccessConfig, AuthCallerConfig, KnowledgeBaseConfig, ServiceSettings
from rag.observe import shutdown_observability
from rag.security.access import Caller, Credentials
from rag.security.tokens import make_token_verifier
from rag.serve import cli
from rag.serve.http_app import build_http_app
from rag.serve.mcp_server import build_mcp_server
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.auth import AUDIENCE, ISSUER, FakeIssuer, new_rsa_key
from rag.testing.live import LiveServer
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

LOCAL = AuthCallerConfig(scope="rag.local", caller="local", acl_groups=["public", "internal"])
GUEST = AuthCallerConfig(scope="rag.guest", caller="guest", acl_groups=["public"])


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture(scope="module")
def issuer() -> FakeIssuer:
    return FakeIssuer()


def _configs() -> dict[str, KnowledgeBaseConfig]:
    return {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs", description="public pages", access=AccessConfig(callers=["*"])
        ),
        "mt5": KnowledgeBaseConfig(
            kb="mt5", description="private manual", access=AccessConfig(callers=["local"])
        ),
    }


def oauth_settings(tmp_path: Path, issuer: FakeIssuer, **overrides: Any) -> ServiceSettings:
    return observe_settings(
        auth_mode="oauth",
        auth_issuer=ISSUER,
        auth_audience=AUDIENCE,
        auth_jwks_path=str(issuer.write_jwks(tmp_path / "jwks.json")),
        auth_callers=[LOCAL, GUEST],
        service_allowed_hosts=["testserver"],
        **overrides,
    )


def _service(settings: ServiceSettings) -> tuple[SearchService, dict[str, FakeSearchPipeline]]:
    pipelines = {kb: FakeSearchPipeline(kb) for kb in _configs()}
    service = SearchService(settings, configs=_configs(), builder=FakePipelineBuilder(pipelines))
    return service, pipelines


def _app(settings: ServiceSettings, service: SearchService) -> Any:
    verifier = make_token_verifier(settings)
    mcp = build_mcp_server(service, transport="streamable-http", token_verifier=verifier)
    return build_http_app(service, mcp=mcp, token_verifier=verifier)


@pytest.fixture
def oauth(tmp_path: Path, issuer: FakeIssuer) -> Iterator[tuple[TestClient, dict[str, Any]]]:
    settings = oauth_settings(tmp_path, issuer)
    service, pipelines = _service(settings)
    with TestClient(_app(settings, service)) as client:
        assert service.wait_ready(5)
        yield client, pipelines


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


SEARCH_MT5 = {"kb": "mt5", "query": "OrderSend parameters"}


def test_http_without_a_token_is_401_with_the_bearer_challenge(oauth: Any) -> None:
    client, pipelines = oauth
    r = client.post("/v1/search_docs", json=SEARCH_MT5)
    assert r.status_code == 401, r.text
    assert r.json()["error"] == "unauthorized" and set(r.json()) == {
        "error",
        "request_id",
        "message",
    }
    assert r.headers["www-authenticate"].startswith("Bearer ")
    assert "resource_metadata=" in r.headers["www-authenticate"]
    assert pipelines["mt5"].calls == []
    r = client.get("/v1/knowledge_bases", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401


@pytest.mark.parametrize("case", ["expired", "audience", "issuer", "signature"])
def test_http_bad_tokens_are_401(oauth: Any, issuer: FakeIssuer, case: str) -> None:
    client, pipelines = oauth
    token = {
        "expired": lambda: issuer.token(expires_in_s=-120),
        "audience": lambda: issuer.token(audience="http://other/mcp"),
        "issuer": lambda: issuer.token(issuer="https://evil.example/"),
        "signature": lambda: issuer.token(key=new_rsa_key()),
    }[case]()
    r = client.post("/v1/search_docs", json=SEARCH_MT5, headers=bearer(token))
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert 'error="invalid_token"' in r.headers["www-authenticate"]
    assert pipelines["mt5"].calls == []


def test_http_valid_token_reaches_the_tool_with_the_callers_groups(
    oauth: Any, issuer: FakeIssuer
) -> None:
    client, pipelines = oauth
    r = client.post("/v1/search_docs", json=SEARCH_MT5, headers=bearer(issuer.token()))
    assert r.status_code == 200, r.text
    assert pipelines["mt5"].calls[-1]["acl_groups"] == ("public", "internal")
    r = client.get("/v1/knowledge_bases", headers=bearer(issuer.token()))
    assert [k["kb"] for k in r.json()["knowledge_bases"]] == ["hf_docs", "mt5"]


def test_scope_without_the_kb_gets_kb_not_found_like_today(oauth: Any, issuer: FakeIssuer) -> None:
    client, pipelines = oauth
    guest = bearer(issuer.token(scopes=("rag.guest",), client_id="guest-app"))
    r = client.post("/v1/search_docs", json=SEARCH_MT5, headers=guest)
    assert r.status_code == 404 and r.json()["error"] == "kb_not_found"
    r = client.post("/v1/get_context", json={"chunk_id": "mt5:p1.md#0"}, headers=guest)
    assert r.status_code == 404 and r.json()["error"] == "kb_not_found"
    r = client.post(
        "/v1/ask_docs/stream", json={"kb": "mt5", "question": "OrderSend?"}, headers=guest
    )
    assert r.status_code == 404 and r.json()["error"] == "kb_not_found"
    r = client.get("/v1/knowledge_bases", headers=guest)
    assert [k["kb"] for k in r.json()["knowledge_bases"]] == ["hf_docs"]
    r = client.post("/v1/search_docs", json={"kb": "hf_docs", "query": "q"}, headers=guest)
    assert r.status_code == 200
    assert pipelines["hf_docs"].calls[-1]["acl_groups"] == ("public",)
    assert pipelines["mt5"].calls == [] and pipelines["mt5"].context_calls == []


@pytest.mark.parametrize("scopes", [("openid",), ("rag.local", "rag.guest"), ()])
def test_scope_mapping_to_no_or_several_callers_is_forbidden(
    oauth: Any, issuer: FakeIssuer, scopes: tuple[str, ...]
) -> None:
    client, pipelines = oauth
    r = client.post("/v1/search_docs", json=SEARCH_MT5, headers=bearer(issuer.token(scopes=scopes)))
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert pipelines["mt5"].calls == []


def test_health_and_the_page_need_no_token(oauth: Any) -> None:
    client, _ = oauth
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/").status_code == 200


def test_scope_resolver_maps_one_scope_to_one_caller(tmp_path: Path, issuer: FakeIssuer) -> None:
    service, _ = _service(oauth_settings(tmp_path, issuer))
    caller = service.resolver.resolve(
        Credentials(transport="http", client_id="c", subject="s", scopes=("x", "rag.guest"))
    )
    assert caller == Caller(caller_id="guest", acl_groups=("public",))


def test_stdio_stays_the_local_caller_in_oauth_mode(tmp_path: Path, issuer: FakeIssuer) -> None:
    settings = oauth_settings(tmp_path, issuer, service_acl_groups=["public", "team"])
    service, _ = _service(settings)
    caller = service.resolver.resolve(Credentials(transport="stdio"))
    assert caller == Caller(caller_id="local", acl_groups=("public", "team"))


def test_local_mode_is_unchanged_and_ignores_bearer_headers() -> None:
    settings = observe_settings(service_allowed_hosts=["testserver"])
    service, pipelines = _service(settings)
    with TestClient(build_http_app(service)) as client:
        assert service.wait_ready(5)
        r = client.post("/v1/search_docs", json=SEARCH_MT5)
        assert r.status_code == 200
        r = client.post("/v1/search_docs", json=SEARCH_MT5, headers=bearer("garbage"))
        assert r.status_code == 200
    assert pipelines["mt5"].calls[-1]["acl_groups"] == ("public",)


def test_mcp_tool_descriptions_name_only_open_kbs_in_oauth_mode(
    tmp_path: Path, issuer: FakeIssuer
) -> None:
    settings = oauth_settings(tmp_path, issuer)
    service, _ = _service(settings)
    mcp = build_mcp_server(
        service, transport="streamable-http", token_verifier=make_token_verifier(settings)
    )
    text = json.dumps([t.description for t in mcp._tool_manager.list_tools()])
    assert "hf_docs" in text and "mt5" not in text and "private manual" not in text


# ---- MCP over streamable-http, a real server ------------------------------------------------


@pytest.fixture
def live(tmp_path: Path, issuer: FakeIssuer) -> Iterator[tuple[LiveServer, Any]]:
    settings = oauth_settings(tmp_path, issuer)
    service, pipelines = _service(settings)
    with LiveServer(_app(settings, service), service=service) as server:
        assert service.wait_ready(5)
        yield server, pipelines


def _mcp_calls(url: str, token: str, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    async def main() -> list[Any]:
        out = []
        async with (
            httpx2.AsyncClient(headers=bearer(token), timeout=20) as http,
            Client(streamable_http_client(url, http_client=http)) as client,
        ):
            for name, arguments in calls:
                out.append(await client.call_tool(name, arguments))
        return out

    return anyio.run(main)


def test_mcp_without_or_with_a_bad_token_is_401(live: Any, issuer: FakeIssuer) -> None:
    server, pipelines = live
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    headers = {"Accept": "application/json, text/event-stream"}
    r = httpx.post(server.url + "/mcp", json=body, headers=headers)
    assert r.status_code == 401
    assert "resource_metadata=" in r.headers["www-authenticate"]
    for token in (issuer.token(expires_in_s=-120), issuer.token(audience="http://other/mcp")):
        r = httpx.post(server.url + "/mcp", json=body, headers={**headers, **bearer(token)})
        assert r.status_code == 401
    assert pipelines["mt5"].calls == []


def test_mcp_protected_resource_metadata_names_the_issuer(live: Any) -> None:
    server, _ = live
    r = httpx.get(server.url + "/.well-known/oauth-protected-resource/mcp")
    assert r.status_code == 200, r.text
    assert r.json()["authorization_servers"] == [ISSUER]


def test_mcp_valid_tokens_reach_the_tools_as_their_callers(live: Any, issuer: FakeIssuer) -> None:
    server, pipelines = live
    url = server.url + "/mcp"
    search = ("search_docs", {"kb": "mt5", "query": "OrderSend"})
    listing = ("list_knowledge_bases", {})
    ok, kbs = _mcp_calls(url, issuer.token(), [search, listing])
    assert not ok.is_error and ok.structured_content["kb"] == "mt5"
    assert [k["kb"] for k in kbs.structured_content["knowledge_bases"]] == ["hf_docs", "mt5"]
    assert pipelines["mt5"].calls[-1]["acl_groups"] == ("public", "internal")
    calls_before = len(pipelines["mt5"].calls)

    denied, kbs = _mcp_calls(url, issuer.token(scopes=("rag.guest",)), [search, listing])
    assert denied.is_error and denied.structured_content["error"] == "kb_not_found"
    assert [k["kb"] for k in kbs.structured_content["knowledge_bases"]] == ["hf_docs"]
    (forbidden,) = _mcp_calls(url, issuer.token(scopes=("openid",)), [search])
    assert forbidden.is_error and forbidden.structured_content["error"] == "forbidden"
    assert len(pipelines["mt5"].calls) == calls_before


# ---- the command line ------------------------------------------------------------------------


def _factory(settings: ServiceSettings) -> SearchService:
    return _service(settings)[0]


def test_cli_refuses_a_public_address_in_local_mode(capsys: pytest.CaptureFixture[str]) -> None:
    ran: list[Any] = []
    code = cli.main(
        ["http", "--host", "0.0.0.0"],
        settings=observe_settings(),
        service_factory=_factory,
        run_http=lambda app, host, port: ran.append(host),
    )
    assert code == 2 and ran == []
    assert "AUTH_MODE=oauth" in capsys.readouterr().err


def test_cli_in_oauth_mode_serves_with_the_verifier(
    tmp_path: Path, issuer: FakeIssuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "init_observability", lambda s, name: None)
    monkeypatch.setattr(cli, "shutdown_observability", lambda: None)
    statuses: list[int] = []

    def run_http(app: Any, host: str, port: int) -> None:
        # No lifespan: only the guards in front of the routes are under test here.
        client = TestClient(app)
        statuses.append(client.post("/v1/search_docs", json=SEARCH_MT5).status_code)

    code = cli.main(
        ["http", "--host", "0.0.0.0"],
        settings=oauth_settings(tmp_path, issuer),
        service_factory=_factory,
        run_http=run_http,
    )
    assert code == 0 and statuses == [401]
