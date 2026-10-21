"""Spec P0-09: `python -m rag.serve stdio|http` builds the service and runs its server."""

from __future__ import annotations

from typing import Any

import pytest

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.serve import cli
from rag.serve.service import SearchService
from rag.testing import observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline


def _service_factory(created: list[SearchService]) -> Any:
    def factory(settings: Any) -> SearchService:
        configs = {"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))}
        builder = FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")})
        service = SearchService(settings, configs=configs, builder=builder)
        created.append(service)
        return service

    return factory


def test_parser_defaults_from_settings() -> None:
    settings = observe_settings(service_host="127.0.0.2", service_port=29999)
    args = cli.build_parser(settings).parse_args(["http"])
    assert (args.command, args.host, args.port) == ("http", "127.0.0.2", 29999)
    args = cli.build_parser(settings).parse_args(["http", "--host", "0.0.0.0", "--port", "1234"])
    assert (args.host, args.port) == ("0.0.0.0", 1234)
    assert cli.build_parser(settings).parse_args(["stdio"]).command == "stdio"
    with pytest.raises(SystemExit):
        cli.build_parser(settings).parse_args([])


def test_stdio_and_http_start_the_load_and_run(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[SearchService] = []
    ran: list[tuple[str, Any]] = []
    settings = observe_settings()
    monkeypatch.setattr(cli, "init_observability", lambda s, name: ran.append(("obs", name)))
    monkeypatch.setattr(cli, "shutdown_observability", lambda: ran.append(("shutdown", None)))

    def run_stdio(mcp: Any) -> None:
        ran.append(("stdio", sorted(t.name for t in mcp._tool_manager.list_tools())))

    def run_http(app: Any, host: str, port: int) -> None:
        ran.append(("http", (host, port, sorted(r.path for r in app.routes))))

    common = {
        "settings": settings,
        "service_factory": _service_factory(created),
        "run_stdio": run_stdio,
        "run_http": run_http,
    }
    assert cli.main(["stdio"], **common) == 0
    # stdio starts the load itself (in the background); main closes the service on exit.
    created[0].wait_ready(5)
    assert created[0].status()["status"] == "closed"
    assert cli.main(["http", "--port", "1234"], **common) == 0

    assert ran[0] == ("obs", cli.SERVICE_NAME)
    assert ran[1] == (
        "stdio",
        ["ask_docs", "feedback", "get_context", "list_knowledge_bases", "search_docs"],
    )
    assert ran[2] == ("shutdown", None)
    host, port, paths = ran[4][1]
    assert (ran[4][0], host, port) == ("http", settings.service_host, 1234)
    assert {"/healthz", "/readyz", "/v1/search_docs", "/v1/ask_docs", "/v1/get_context"} <= set(
        paths
    )
    assert ran[5] == ("shutdown", None)


# ---- spec P3-04 review round 1: the dev-fake mode, the import, the uvicorn settings ---------


def test_dev_fake_is_not_imported_unless_asked_for() -> None:
    import subprocess
    import sys

    code = "import rag.serve.cli, sys; print('rag.serve.devfake' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_dev_fake_refuses_other_addresses_and_writes_its_records_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(cli, "init_observability", lambda s, name: seen.append(s))
    monkeypatch.setattr(cli, "shutdown_observability", lambda: None)
    started: list[tuple[str, int]] = []
    settings = observe_settings(business_record_path="data/observe/records/business.jsonl")

    def run_http(app: Any, host: str, port: int) -> None:
        started.append((host, port))

    assert (
        cli.main(["http", "--dev-fake", "--host", "0.0.0.0"], settings=settings, run_http=run_http)
        == 2
    )
    assert started == [] and seen == []  # refused before anything was set up
    assert (
        cli.main(["http", "--dev-fake", "--port", "1"], settings=settings, run_http=run_http) == 0
    )
    assert started == [("127.0.0.1", 1)]
    # Records of fake questions do not go into the file of real ones.
    assert seen[0].business_record_path == "data/observe/dev-fake/business.jsonl"
    assert settings.business_record_path == "data/observe/records/business.jsonl"


def test_the_server_waits_a_few_seconds_for_open_streams_then_cancels_them() -> None:
    from rag.serve.server import make_server

    settings = observe_settings(service_shutdown_timeout_s=7)
    server = make_server(object(), "127.0.0.1", 1, service=None, settings=settings)  # type: ignore[arg-type]
    assert server.config.timeout_graceful_shutdown == 7
    assert (server.config.host, server.config.port) == ("127.0.0.1", 1)
    with pytest.raises(ValueError):
        observe_settings(service_shutdown_timeout_s=0)
