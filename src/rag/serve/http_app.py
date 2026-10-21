"""HTTP API of the search service (design s16-1) and the MCP streamable-http endpoint.

One Starlette app, so one process holds one reranker for every way in:

    GET  /                     the Q&A page (one HTML file, a script for the SSE parser, one for
                               drawing, one for the page, one style sheet: `web/`)
    GET  /ui/<file>            those files, nothing else
    GET  /healthz              200 {"status": "ok"} while the process runs
    GET  /readyz               200 when the reranker and the chains are loaded, else 503
    POST /v1/search_docs       {kb, query, filters?, top_k?, deadline_ms?} -> SearchResponse
    POST /v1/ask_docs          {kb, question, session_id?, deadline_ms?, resume_token?, answer?}
                               -> AskResponse
    POST /v1/ask_docs/stream   the same body -> Server-Sent Events (`rag.serve.stream`)
    POST /v1/feedback          {request_id, useful, comment?} -> FeedbackResponse
    POST /v1/get_context       {chunk_id, mode?}                           -> ContextResponse
    GET  /v1/knowledge_bases                                               -> KnowledgeBaseList
    /mcp                       MCP over streamable-http (when an MCPServer is given)

JSON bodies are checked against fixed models (unknown fields are `invalid_request`). Errors are
`{error, request_id, message}` with the category's status (`rag.serve.errors.http_status`);
`not_ready`, `rate_limited` and `budget_exceeded` add `Retry-After`. With a token verifier
(`AUTH_MODE=oauth`, spec P8-01) every `/v1/` route needs `Authorization: Bearer <access token>`:
a missing or refused token is `401 unauthorized` with a `WWW-Authenticate: Bearer` challenge
that points to the protected resource metadata; the page, `/ui/`, `/healthz` and `/readyz` stay
open. `/mcp` checks its tokens itself (the SDK's `RequireAuthMiddleware`, same verifier). A
caller may send `X-Request-Id` (letters, digits, `._:-`, at
most 64); every answer carries the request id in that header. Starlette rather than FastAPI:
the MCP SDK's streamable-http app is already a Starlette app (mounted here as it is), Starlette
comes with the SDK, and four routes need no generated docs or dependency injection.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from mcp.server import MCPServer
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser, BearerAuthBackend
from mcp.server.auth.provider import TokenVerifier
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.observe import new_request_id
from rag.security.access import Credentials
from rag.security.tokens import protected_resource_metadata_url
from rag.serve.errors import ServiceError, http_status
from rag.serve.schema import ErrorResponse
from rag.serve.service import SearchService
from rag.serve.stream import encode
from rag.types import ErrorCategory

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"
WEB_DIR = Path(__file__).parent / "web"
_UI_FILES = {
    "sse.js": "text/javascript",
    "render.js": "text/javascript",
    "app.js": "text/javascript",
    "app.css": "text/css",
}
# The page loads only its own files and runs no inline code, so a script that got into the
# page through an answer or a document could not run and could not send anything out.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-cache",
}
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
RETRY_AFTER_S = 10
TRANSPORT = "http"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kb: str
    query: str
    filters: dict[str, Any] | None = None
    top_k: int | None = None
    deadline_ms: int | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kb: str
    question: str
    session_id: str | None = None
    deadline_ms: int | None = None
    resume_token: str | None = None
    answer: str | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str
    useful: bool
    comment: str | None = None


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    mode: str | None = None


LOCAL_NAMES = frozenset({"localhost", "127.0.0.1", "[::1]"})
_HOST = re.compile(r"(?P<name>\[[0-9a-f:]+\]|[a-z0-9.-]+)(?::[0-9]{1,5})?")
_MCP_PATH = "/mcp"  # the MCP mount checks Host and Origin itself (the SDK's rebinding protection)


def host_name(header: str) -> str | None:
    """The host name of a `Host` header, lower case, without port and trailing dot; None when the
    header is not a plain `name`, `name:port`, `ipv4:port` or `[ipv6]:port`."""
    match = _HOST.fullmatch(header.strip().lower())
    if match is None:
        return None
    name = match["name"]
    return name if name.startswith("[") else name.rstrip(".")


def allowed_host_names(listen_host: str, extra: list[str]) -> frozenset[str]:
    """The names the HTTP routes answer to: localhost, 127.0.0.1, [::1], the address the server
    listens on (unless it listens on every interface) and `service_allowed_hosts`."""
    names = set(LOCAL_NAMES) | {name.lower().rstrip(".") for name in extra}
    if listen_host not in ("", "0.0.0.0", "::"):
        names.add(f"[{listen_host}]" if ":" in listen_host else listen_host.lower())
    return frozenset(names)


class RequestGuard:
    """Refuses, before any route runs, a request whose `Host` is not one of the names the service
    answers to (421) and a POST whose body is not declared as JSON (415), with the usual error
    body. A page on another site can make a browser send a POST with a form or `text/plain`
    type to a service on the developer's machine without a preflight, and a DNS name that was
    rebound to 127.0.0.1 reaches it with its own Host header; neither is a request this service
    was meant to answer. The MCP mount is left to the SDK's own checks."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: frozenset[str]) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].startswith(_MCP_PATH):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        refusal = None
        if host_name(headers.get("host", "")) not in self.allowed_hosts:
            refusal = 421
        elif scope["method"] == "POST" and not _is_json(headers.get("content-type", "")):
            refusal = 415
        if refusal is None:
            await self.app(scope, receive, send)
            return
        given = headers.get(REQUEST_ID_HEADER.lower(), "")
        request_id = given if _REQUEST_ID.fullmatch(given) else new_request_id()
        logger.warning(
            "request %s refused with %s (Host %r)", request_id, refusal, headers.get("host")
        )
        response = JSONResponse(
            ErrorResponse.of(ErrorCategory.INVALID_REQUEST, request_id).model_dump(mode="json"),
            status_code=refusal,
            headers={REQUEST_ID_HEADER: request_id},
        )
        await response(scope, receive, send)


_API_PATH = "/v1/"


class BearerGuard:
    """Refuses a `/v1/` request without a valid bearer token (401, `unauthorized`, with the
    RFC 6750 challenge) before any route runs; a request with one gets the SDK's
    `AuthenticatedUser` in `scope["user"]`, which the routes turn into `Credentials`. The token
    check is the SDK's `BearerAuthBackend` over our verifier, so `/v1/` and `/mcp` accept the
    same tokens."""

    def __init__(
        self, app: ASGIApp, *, verifier: TokenVerifier, resource_metadata_url: str | None
    ) -> None:
        self.app = app
        self.backend = BearerAuthBackend(verifier)
        self.resource_metadata_url = resource_metadata_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_API_PATH):
            await self.app(scope, receive, send)
            return
        conn = HTTPConnection(scope)
        result = await self.backend.authenticate(conn)
        if result is not None:
            _, user = result
            scope["user"] = user
            await self.app(scope, receive, send)
            return
        given = conn.headers.get(REQUEST_ID_HEADER, "")
        request_id = given if _REQUEST_ID.fullmatch(given) else new_request_id()
        has_token = conn.headers.get("authorization", "").lower().startswith("bearer ")
        parts = ['error="invalid_token"'] if has_token else []
        if self.resource_metadata_url:
            parts.append(f'resource_metadata="{self.resource_metadata_url}"')
        challenge = "Bearer " + ", ".join(parts) if parts else "Bearer"
        logger.warning(
            "request %s refused: %s bearer token on %s",
            request_id,
            "invalid" if has_token else "no",
            scope["path"],
        )
        response = JSONResponse(
            ErrorResponse.of(ErrorCategory.UNAUTHORIZED, request_id).model_dump(mode="json"),
            status_code=401,
            headers={REQUEST_ID_HEADER: request_id, "WWW-Authenticate": challenge},
        )
        await response(scope, receive, send)


def credentials_of(request: Request) -> Credentials:
    """What the transport knows about the sender: the verified token's client, subject and
    scopes when `BearerGuard` accepted one; nothing else (local mode)."""
    user = request.scope.get("user")
    if isinstance(user, AuthenticatedUser):
        token = user.access_token
        return Credentials(
            transport=TRANSPORT,
            client_id=token.client_id,
            subject=token.subject,
            scopes=tuple(token.scopes),
        )
    return Credentials(transport=TRANSPORT)


def _is_json(content_type: str) -> bool:
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _request_id(request: Request) -> str:
    given = request.headers.get(REQUEST_ID_HEADER)
    if given and _REQUEST_ID.fullmatch(given):
        return given
    return new_request_id()


def _error(category: str, request_id: str, retry_after_s: float | None = None) -> JSONResponse:
    headers = {REQUEST_ID_HEADER: request_id}
    if category == ErrorCategory.NOT_READY:
        headers["Retry-After"] = str(RETRY_AFTER_S)
    elif retry_after_s is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after_s)))
    return JSONResponse(
        ErrorResponse.of(category, request_id).model_dump(mode="json"),
        status_code=http_status(category),
        headers=headers,
    )


async def _body[M: BaseModel](request: Request, model: type[M]) -> tuple[M | None, str]:
    """The JSON body as `model`, or None and why (for the log) when it is not valid JSON or
    does not fit."""
    try:
        data = await request.json()
        return model.model_validate(data), ""
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
        return None, f"body rejected, fields {fields}"
    except (ValueError, UnicodeDecodeError):
        return None, "body is not JSON"


async def _call(
    fn: Callable[..., BaseModel], credentials: Credentials, request_id: str, **kwargs: Any
) -> JSONResponse:
    try:
        response = await anyio.to_thread.run_sync(
            functools.partial(fn, credentials, request_id=request_id, **kwargs)
        )
    except ServiceError as err:
        return _error(err.category, err.request_id, err.retry_after_s)
    except Exception:
        logger.exception("request %s failed outside the service", request_id)
        return _error(ErrorCategory.INTERNAL_ERROR, request_id)
    return JSONResponse(response.model_dump(mode="json"), headers={REQUEST_ID_HEADER: request_id})


async def _acall(
    fn: Callable[..., Awaitable[BaseModel]],
    credentials: Credentials,
    request_id: str,
    **kwargs: Any,
) -> JSONResponse:
    """Like `_call` for a service method that is async (it runs on this event loop)."""
    try:
        response = await fn(credentials, request_id=request_id, **kwargs)
    except ServiceError as err:
        return _error(err.category, err.request_id, err.retry_after_s)
    except Exception:
        logger.exception("request %s failed outside the service", request_id)
        return _error(ErrorCategory.INTERNAL_ERROR, request_id)
    return JSONResponse(response.model_dump(mode="json"), headers={REQUEST_ID_HEADER: request_id})


def build_http_app(
    service: SearchService,
    *,
    mcp: MCPServer | None = None,
    host: str = "127.0.0.1",
    token_verifier: TokenVerifier | None = None,
) -> Starlette:
    """The app over `service`; its lifespan starts the service's load (and the MCP session
    manager when `mcp` is given, mounted at `/mcp`). `host` is where it listens: on 127.0.0.1
    the MCP endpoint checks Host and Origin headers (DNS rebinding protection of the SDK).
    `token_verifier` (OAuth mode): `/v1/` routes need a bearer token it accepts."""

    def web_file(name: str, media_type: str) -> Response:
        return Response(
            (WEB_DIR / name).read_text("utf-8"), media_type=media_type, headers=_PAGE_HEADERS
        )

    async def page(request: Request) -> Response:
        return web_file("index.html", "text/html")

    async def ui_file(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in _UI_FILES:
            return Response("Not Found", status_code=404, media_type="text/plain")
        return web_file(name, _UI_FILES[name])

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def readyz(request: Request) -> JSONResponse:
        # Counts, not names: a KB name would tell callers about KBs they may not see (s16-4).
        status = service.status()
        states = list(status["knowledge_bases"].values())
        body = {
            "status": status["status"],
            "load_s": status["load_s"],
            "ready_kbs": states.count("ready"),
            "unavailable_kbs": states.count("unavailable"),
        }
        return JSONResponse(body, status_code=200 if service.ready else 503)

    async def reject(request: Request, tool: str, request_id: str, detail: str) -> JSONResponse:
        return await _call(
            functools.partial(
                service.reject, tool, category=ErrorCategory.INVALID_REQUEST, detail=detail
            ),
            credentials_of(request),
            request_id,
        )

    async def search_docs(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, SearchRequest)
        if body is None:
            return await reject(request, "search_docs", request_id, problem)
        return await _call(
            service.search_docs,
            credentials_of(request),
            request_id,
            kb=body.kb,
            query=body.query,
            filters=body.filters,
            top_k=body.top_k,
            deadline_ms=body.deadline_ms,
        )

    async def ask_docs(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, AskRequest)
        if body is None:
            return await reject(request, "ask_docs", request_id, problem)
        return await _acall(
            service.ask_docs,
            credentials_of(request),
            request_id,
            kb=body.kb,
            question=body.question,
            session_id=body.session_id,
            deadline_ms=body.deadline_ms,
            resume_token=body.resume_token,
            answer=body.answer,
        )

    async def ask_docs_stream(request: Request) -> JSONResponse | StreamingResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, AskRequest)
        if body is None:
            return await reject(request, "ask_docs", request_id, problem)
        events = service.ask_docs_stream(
            credentials_of(request),
            request_id=request_id,
            kb=body.kb,
            question=body.question,
            session_id=body.session_id,
            deadline_ms=body.deadline_ms,
            resume_token=body.resume_token,
            answer=body.answer,
        )
        # The first event decides: a failure before it is a plain error response (s16-9), the
        # same one `POST /v1/ask_docs` gives; once the first event is out the status is 200.
        try:
            first = await anext(events)
        except ServiceError as err:
            return _error(err.category, err.request_id, err.retry_after_s)
        except Exception:
            logger.exception("request %s failed outside the service", request_id)
            return _error(ErrorCategory.INTERNAL_ERROR, request_id)

        async def body_bytes() -> AsyncIterator[bytes]:
            # `aclosing`: when the response is closed at a write, the event generator is
            # closed with it, and that cancels the run.
            async with contextlib.aclosing(events):
                yield encode(first)
                async for item in events:
                    yield encode(item)

        return StreamingResponse(
            body_bytes(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # nginx and kin: do not hold the stream back
                REQUEST_ID_HEADER: request_id,
            },
        )

    async def feedback(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, FeedbackRequest)
        if body is None:
            return await reject(request, "feedback", request_id, problem)
        return await _call(
            service.feedback,
            credentials_of(request),
            request_id,
            rated_request_id=body.request_id,
            useful=body.useful,
            comment=body.comment,
        )

    async def get_context(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, ContextRequest)
        if body is None:
            return await reject(request, "get_context", request_id, problem)
        return await _call(
            service.get_context,
            credentials_of(request),
            request_id,
            chunk_id=body.chunk_id,
            mode=body.mode,
        )

    async def knowledge_bases(request: Request) -> JSONResponse:
        return await _call(
            service.list_knowledge_bases, credentials_of(request), _request_id(request)
        )

    routes: list[BaseRoute] = [
        Route("/", page, methods=["GET"]),
        Route("/ui/{name}", ui_file, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/readyz", readyz, methods=["GET"]),
        Route("/v1/search_docs", search_docs, methods=["POST"]),
        Route("/v1/ask_docs", ask_docs, methods=["POST"]),
        Route("/v1/ask_docs/stream", ask_docs_stream, methods=["POST"]),
        Route("/v1/feedback", feedback, methods=["POST"]),
        Route("/v1/get_context", get_context, methods=["POST"]),
        Route("/v1/knowledge_bases", knowledge_bases, methods=["GET"]),
    ]
    if mcp is not None:
        routes.append(
            Mount("/", app=mcp.streamable_http_app(streamable_http_path="/mcp", host=host))
        )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        service.start()
        try:
            if mcp is None:
                yield
            else:
                async with mcp.session_manager.run():
                    yield
        finally:
            await service.cancel_streams()  # runs of streams still open end cleanly

    middleware = [
        Middleware(
            RequestGuard,
            allowed_hosts=allowed_host_names(host, service.settings.service_allowed_hosts),
        )
    ]
    if token_verifier is not None:
        middleware.append(
            Middleware(
                BearerGuard,
                verifier=token_verifier,
                resource_metadata_url=protected_resource_metadata_url(service.settings),
            )
        )
    return Starlette(routes=routes, lifespan=lifespan, middleware=middleware)
