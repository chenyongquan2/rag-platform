"""Command line of the search service (spec P0-09, design s16-1).

    uv run python -m rag.serve stdio                       # MCP over stdio, one process per client
    uv run python -m rag.serve http [--host H] [--port P]  # HTTP API + MCP streamable-http at /mcp
                                                           # and the Q&A page at /
    uv run python -m rag.serve http --dev-fake             # the same over fake data (no gateway)

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
from rag.serve.devfake import LOCAL_HOSTS, build_dev_service
from rag.serve.http_app import build_http_app
from rag.serve.mcp_server import build_mcp_server
from rag.serve.service import SearchService

SERVICE_NAME = "rag-serve"


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


def _run_http(app: Any, host: str, port: int) -> None:
    import uvicorn

    # log_config=None keeps our logging setup (stderr + log file) instead of uvicorn's.
    uvicorn.run(app, host=host, port=port, log_config=None, log_level="warning")


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: ServiceSettings | None = None,
    service_factory: Callable[[ServiceSettings], SearchService] = SearchService,
    run_stdio: Callable[[Any], None] = _run_stdio,
    run_http: Callable[[Any, str, int], None] = _run_http,
) -> int:
    settings = settings if settings is not None else ServiceSettings()
    args = build_parser(settings).parse_args(argv)
    if args.command == "http" and args.dev_fake:
        if args.host not in LOCAL_HOSTS:
            hosts = " or ".join(LOCAL_HOSTS)
            print(f"--dev-fake serves fake data: use --host {hosts}", file=sys.stderr)
            return 2

        def service_factory(s: ServiceSettings) -> SearchService:
            return build_dev_service(s, search_ms=args.fake_search_ms, token_ms=args.fake_token_ms)

    init_observability(settings, SERVICE_NAME)
    service = service_factory(settings)
    try:
        if args.command == "stdio":
            service.start()
            run_stdio(build_mcp_server(service, transport="stdio"))
        else:
            mcp = build_mcp_server(service, transport="streamable-http")
            # The app's lifespan starts the load once uvicorn runs.
            run_http(build_http_app(service, mcp=mcp, host=args.host), args.host, args.port)
    finally:
        service.close()
        shutdown_observability()
    return 0
