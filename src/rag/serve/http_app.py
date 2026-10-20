"""HTTP API of the search service (design s16-1) and the MCP streamable-http endpoint.

One Starlette app, so one process holds one reranker for every way in:

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
`not_ready` adds `Retry-After`. A caller may send `X-Request-Id` (letters, digits, `._:-`, at
most 64); every answer carries the request id in that header. Starlette rather than FastAPI:
the MCP SDK's streamable-http app is already a Starlette app (mounted here as it is), Starlette
comes with the SDK, and four routes need no generated docs or dependency injection.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import BaseRoute, Mount, Route

from rag.observe import new_request_id
from rag.security.access import Credentials
from rag.serve.errors import ServiceError, http_status
from rag.serve.schema import ErrorResponse
from rag.serve.service import SearchService
from rag.serve.stream import encode
from rag.types import ErrorCategory

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"
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


def _request_id(request: Request) -> str:
    given = request.headers.get(REQUEST_ID_HEADER)
    if given and _REQUEST_ID.match(given):
        return given
    return new_request_id()


def _error(category: str, request_id: str) -> JSONResponse:
    headers = {REQUEST_ID_HEADER: request_id}
    if category == ErrorCategory.NOT_READY:
        headers["Retry-After"] = str(RETRY_AFTER_S)
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


async def _call(fn: Callable[..., BaseModel], request_id: str, **kwargs: Any) -> JSONResponse:
    try:
        response = await anyio.to_thread.run_sync(
            functools.partial(fn, Credentials(transport=TRANSPORT), request_id=request_id, **kwargs)
        )
    except ServiceError as err:
        return _error(err.category, err.request_id)
    except Exception:
        logger.exception("request %s failed outside the service", request_id)
        return _error(ErrorCategory.INTERNAL_ERROR, request_id)
    return JSONResponse(response.model_dump(mode="json"), headers={REQUEST_ID_HEADER: request_id})


async def _acall(
    fn: Callable[..., Awaitable[BaseModel]], request_id: str, **kwargs: Any
) -> JSONResponse:
    """Like `_call` for a service method that is async (it runs on this event loop)."""
    try:
        response = await fn(Credentials(transport=TRANSPORT), request_id=request_id, **kwargs)
    except ServiceError as err:
        return _error(err.category, err.request_id)
    except Exception:
        logger.exception("request %s failed outside the service", request_id)
        return _error(ErrorCategory.INTERNAL_ERROR, request_id)
    return JSONResponse(response.model_dump(mode="json"), headers={REQUEST_ID_HEADER: request_id})


def build_http_app(
    service: SearchService, *, mcp: MCPServer | None = None, host: str = "127.0.0.1"
) -> Starlette:
    """The app over `service`; its lifespan starts the service's load (and the MCP session
    manager when `mcp` is given, mounted at `/mcp`). `host` is where it listens: on 127.0.0.1
    the MCP endpoint checks Host and Origin headers (DNS rebinding protection of the SDK)."""

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

    async def reject(tool: str, request_id: str, detail: str) -> JSONResponse:
        return await _call(
            functools.partial(
                service.reject, tool, category=ErrorCategory.INVALID_REQUEST, detail=detail
            ),
            request_id,
        )

    async def search_docs(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, SearchRequest)
        if body is None:
            return await reject("search_docs", request_id, problem)
        return await _call(
            service.search_docs,
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
            return await reject("ask_docs", request_id, problem)
        return await _acall(
            service.ask_docs,
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
            return await reject("ask_docs", request_id, problem)
        events = service.ask_docs_stream(
            Credentials(transport=TRANSPORT),
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
            return _error(err.category, err.request_id)
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
            return await reject("feedback", request_id, problem)
        return await _call(
            service.feedback,
            request_id,
            rated_request_id=body.request_id,
            useful=body.useful,
            comment=body.comment,
        )

    async def get_context(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        body, problem = await _body(request, ContextRequest)
        if body is None:
            return await reject("get_context", request_id, problem)
        return await _call(service.get_context, request_id, chunk_id=body.chunk_id, mode=body.mode)

    async def knowledge_bases(request: Request) -> JSONResponse:
        return await _call(service.list_knowledge_bases, _request_id(request))

    routes: list[BaseRoute] = [
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

    return Starlette(routes=routes, lifespan=lifespan)
