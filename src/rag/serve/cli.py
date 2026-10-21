"""Command line of the search service (spec P0-09, design s16-1).

    uv run python -m rag.serve stdio                       # MCP over stdio, one process per client
    uv run python -m rag.serve http [--host H] [--port P]  # HTTP API + MCP streamable-http at /mcp
                                                           # and the Q&A page at /
    uv run python -m rag.serve http --dev-fake             # the same over fake data (no gateway)

`AUTH_MODE=local` (default) answers only on a local address; `AUTH_MODE=oauth` checks bearer
tokens on `/v1/` and `/mcp` (spec P8-01) and may listen anywhere (stdio is unchanged).

Both set up observability (service name `rag-serve`), read the KB configs and start loading the
reranker in a background thread: over stdio the MCP handshake and `list_knowledge_bases` answer
at once and a search waits for the load within its deadline; over HTTP `/readyz` turns 200 when
the load ends. Logs go to stderr and the log file, never to stdout (stdout is the MCP channel).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from rag.config import ServiceSettings
from rag.observe import init_observability, shutdown_observability
from rag.observe.health import DependencyChecker, register_dependency_gauge
from rag.security.tokens import make_token_verifier
from rag.serve.http_app import build_http_app
from rag.serve.mcp_server import build_mcp_server
from rag.serve.server import make_server
from rag.serve.service import SearchService

SERVICE_NAME = "rag-serve"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DEV_FAKE_RECORDS = "data/observe/dev-fake/business.jsonl"


def build_parser(settings: ServiceSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.serve", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stdio", help="MCP server over stdio (for coding assistants)")
    p_http = sub.add_parser("http", help="HTTP API and MCP streamable-http at /mcp")
    p_http.add_argument("--host", default=settings.service_host, help="default: SERVICE_HOST")
    p_http.add_argument("--port", type=int, default=settings.service_port, help="SERVICE_PORT")
    p_http.add_argument(
        "--dev-fake",
        action="store_true",
        help="serve the fake knowledge base demo_docs (fake search, fake model, memory stores); "
        "only on a local address",
    )
    p_http.add_argument(
        "--fake-search-ms", type=int, default=400, help="with --dev-fake: time of one search"
    )
    p_http.add_argument(
        "--fake-token-ms",
        type=int,
        default=30,
        help="with --dev-fake: pause between two pieces of the answer",
    )
    return parser


def _run_stdio(mcp: Any) -> None:
    mcp.run("stdio")


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: ServiceSettings | None = None,
    service_factory: Callable[[ServiceSettings], SearchService] = SearchService.from_settings,
    run_stdio: Callable[[Any], None] = _run_stdio,
    run_http: Callable[[Any, str, int], None] | None = None,
) -> int:
    settings = settings if settings is not None else ServiceSettings()
    args = build_parser(settings).parse_args(argv)
    local_only = settings.auth_mode == "local" and not getattr(args, "dev_fake", False)
    if args.command == "http" and local_only and args.host not in LOCAL_HOSTS:
        # Local mode serves everyone as the configured caller, private KBs included: only the
        # local machine may reach it (spec P8-01).
        print(
            f"AUTH_MODE=local serves every request as caller {settings.service_caller!r}: listen "
            f"on {' or '.join(LOCAL_HOSTS)}, or set AUTH_MODE=oauth (bearer tokens) to listen "
            "on another address",
            file=sys.stderr,
        )
        return 2
    if args.command == "http" and args.dev_fake:
        if args.host not in LOCAL_HOSTS:
            hosts = " or ".join(LOCAL_HOSTS)
            print(f"--dev-fake serves fake data: use --host {hosts}", file=sys.stderr)
            return 2

        # Only now: the fake knowledge base and its fakes are not imported by a normal server.
        from rag.serve.devfake import build_dev_service

        def service_factory(s: ServiceSettings) -> SearchService:
            return build_dev_service(s, search_ms=args.fake_search_ms, token_ms=args.fake_token_ms)

        # Records of fake questions stay out of the file of real ones.
        settings = settings.model_copy(update={"business_record_path": DEV_FAKE_RECORDS})

    obs = init_observability(settings, SERVICE_NAME)
    if obs is not None and obs.mode != "off":
        # Spec P5-02: can this process reach Elasticsearch and Redis? Read at every export.
        register_dependency_gauge(obs.meter, DependencyChecker.from_settings(settings))
    service = service_factory(settings)
    try:
        if args.command == "stdio":
            service.start()
            run_stdio(build_mcp_server(service, transport="stdio"))
        else:
            verifier = make_token_verifier(settings) if settings.auth_mode == "oauth" else None
            mcp = build_mcp_server(service, transport="streamable-http", token_verifier=verifier)
            if run_http is None:

                def run_http(app: Any, host: str, port: int) -> None:
                    make_server(app, host, port, service=service, settings=settings).run()

            # The app's lifespan starts the load once uvicorn runs.
            app = build_http_app(service, mcp=mcp, host=args.host, token_verifier=verifier)
            run_http(app, args.host, args.port)
    finally:
        service.close()
        shutdown_observability()
    return 0
