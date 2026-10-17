"""The current request (id, kb, caller, tool) in a context variable (spec P0-11, design s15-1).

Context variables follow the code that handles one request: nested calls and asyncio tasks
created inside it see the same value. A plain `threading.Thread` or a thread pool does not; run
the work through `contextvars.copy_context().run(...)` to carry the request (and the current
OTel span, which lives in context variables too) into the thread.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry import trace


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    kb: str | None = None
    caller: str | None = None
    tool: str | None = None


_CURRENT: ContextVar[RequestContext | None] = ContextVar("rag_request", default=None)


def new_request_id() -> str:
    """32 hex characters, random (uuid4)."""
    return uuid.uuid4().hex


def current_request() -> RequestContext | None:
    return _CURRENT.get()


def current_request_id() -> str | None:
    request = _CURRENT.get()
    return request.request_id if request is not None else None


@contextmanager
def bind_request(
    request_id: str | None = None,
    *,
    kb: str | None = None,
    caller: str | None = None,
    tool: str | None = None,
) -> Iterator[RequestContext]:
    """Make a request current for the block; a new id is generated when none is given."""
    request = RequestContext(request_id or new_request_id(), kb=kb, caller=caller, tool=tool)
    token = _CURRENT.set(request)
    try:
        yield request
    finally:
        _CURRENT.reset(token)


def current_trace_id() -> str | None:
    """Trace id of the current span as 32 hex characters, or None without a recording span."""
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def current_span_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.span_id, "016x") if context.is_valid else None
