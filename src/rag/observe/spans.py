"""Request and layer-boundary spans (spec P0-11, design s15-1, s16-8).

    with request_span("search_docs", kb=kb, caller=caller, config_version=cfg_hash) as scope:
        with layer_span("retrieve", attributes={attr.ROUTE: "bm25"}):
            ...
        scope.set_result(result)

`request_span` opens the root span of one request, makes the request current (its id goes on
every span and log line inside) and, when it ends, counts the request and its error categories
and records its duration. `layer_span` wraps one step (as `with` block or decorator of a sync or
async function) and records the step's duration. LlamaIndex and LangChain spans started inside a
block become its children: OTel keeps the current span in a context variable.

An exception leaving a span marks it as an error with `error.type` (the class name) and
`rag.error.category`: the `classify` callback's answer, else the `error_category` attribute of
the exception, else `unclassified`. A category found by an inner span is stored on the
exception, so the request span reports the same one.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from rag.observe import attributes as attr
from rag.observe.context import bind_request, current_request
from rag.observe.setup import get_observability
from rag.types import Answer, RetrievalResult

logger = logging.getLogger(__name__)

Classifier = Callable[[BaseException], str | None]
F = TypeVar("F", bound=Callable[..., Any])

# Python attribute on an exception that carries its error category (design s16-8).
EXCEPTION_CATEGORY_ATTR = "error_category"


def _clean(attributes: Mapping[str, AttributeValue | None]) -> dict[str, AttributeValue]:
    return {k: v for k, v in attributes.items() if v is not None}


def _error_category(exc: BaseException, classify: Classifier | None) -> str:
    category: Any = None
    if classify is not None:
        try:
            category = classify(exc)
        except Exception:
            logger.warning("error classifier failed for %r", exc, exc_info=True)
    if category is None:
        category = getattr(exc, EXCEPTION_CATEGORY_ATTR, None)
    if category is None:
        return attr.UNCLASSIFIED
    category = str(category)
    # Exceptions with __slots__ or a read-only property refuse the attribute.
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, EXCEPTION_CATEGORY_ATTR, category)
    return category


def _mark_error(span: Span, exc: BaseException, classify: Classifier | None) -> str:
    category = _error_category(exc, classify)
    span.record_exception(exc)
    span.set_attribute(attr.ERROR_TYPE, type(exc).__name__)
    span.set_attribute(attr.ERROR_CATEGORY, category)
    span.set_status(Status(StatusCode.ERROR, category))
    return category


def _trace_id(span: Span) -> str | None:
    context = span.get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def activate_span(span: Span) -> contextlib.AbstractContextManager[Span]:
    """Make `span` current inside the block without ending it or marking errors on it: for a
    `layer_span(..., activate=False)` whose work runs in pieces between yields."""
    return trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


def set_error_category(category: str, span: Span | None = None) -> None:
    """Note a category on a span (default: the current one) without failing it, e.g. `degraded`
    when one retrieval route failed. Request metrics count the categories of the result passed to
    `RequestScope.set_result` or added with `RequestScope.add_error_category`."""
    (span or trace.get_current_span()).set_attribute(attr.ERROR_CATEGORY, str(category))


@dataclass
class RequestScope:
    """What a `request_span` block gets: the request and trace ids and the root span."""

    request_id: str
    trace_id: str | None
    span: Span
    tool: str
    kb: str | None
    error_categories: list[str] = field(default_factory=list)

    def add_error_category(self, category: str) -> None:
        """Count a category for this request (flags not carried by a result object)."""
        if category not in self.error_categories:
            self.error_categories.append(str(category))
        self.span.set_attribute(attr.ERROR_CATEGORIES, list(self.error_categories))

    def set_result(self, result: RetrievalResult | Answer) -> None:
        """Index version, data generation, flags and hit count of the result on the root span;
        its error categories are counted when the request ends."""
        self.span.set_attributes(
            _clean(
                {
                    attr.INDEX_VERSION: result.index_version or None,
                    attr.DATA_GENERATION: result.data_generation,
                    attr.DEGRADED: result.degraded,
                    attr.RERANK_TIMEOUT: result.rerank_timeout,
                    attr.HITS: len(result.chunks),
                }
            )
        )
        for category in result.error_categories:
            self.add_error_category(str(category))
        if not result.error_categories:
            self.span.set_attribute(attr.ERROR_CATEGORIES, list(self.error_categories))


@contextmanager
def request_span(
    name: str,
    *,
    kb: str | None,
    request_id: str | None = None,
    caller: str | None = None,
    config_version: str | None = None,
    data_generation: int | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
    classify: Classifier | None = None,
) -> Iterator[RequestScope]:
    """Root span of one request named after the tool (`search_docs`, `ask_docs`); the request
    id is the caller's or a new one."""
    obs = get_observability()
    start = time.perf_counter()
    with bind_request(request_id, kb=kb, caller=caller, tool=name) as request:
        span_attributes = _clean(
            {
                attr.TOOL: name,
                attr.KB: kb,
                attr.REQUEST_ID: request.request_id,
                attr.CALLER: caller,
                attr.CONFIG_VERSION: config_version,
                attr.DATA_GENERATION: data_generation,
                **(attributes or {}),
            }
        )
        with obs.tracer.start_as_current_span(
            name,
            kind=SpanKind.SERVER,
            attributes=span_attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            scope = RequestScope(
                request_id=request.request_id,
                trace_id=_trace_id(span),
                span=span,
                tool=name,
                kb=kb,
            )
            outcome = attr.OUTCOME_OK
            try:
                yield scope
            except Exception as exc:
                outcome = attr.OUTCOME_ERROR
                scope.add_error_category(_mark_error(span, exc, classify))
                raise
            except BaseException as exc:  # cancelled or interrupted: not a failure of ours
                outcome = attr.OUTCOME_CANCELLED
                span.set_attribute(attr.ERROR_TYPE, type(exc).__name__)
                raise
            finally:
                span.set_attribute(attr.OUTCOME, outcome)
                obs.metrics.record_request(
                    kb=kb, tool=name, outcome=outcome, error_categories=scope.error_categories
                )
                obs.metrics.record_step(name, time.perf_counter() - start, outcome=outcome, kb=kb)


class layer_span:  # noqa: N801 - used like a function: `with layer_span(...)`, `@layer_span(...)`
    """Span of one layer step under the current request; records the step's duration.

    As a context manager it yields the span; as a decorator it opens a new span for each call
    (for async functions around the awaited body). One instance is not reentrant as a context
    manager; the decorator form is.

    `activate=False` (context manager only) starts the span without making it current: an async
    generator that yields to its caller in the middle of the step makes it current only around
    its own work with `activate_span(span)`, so the caller's code between two items does not run
    inside the step's span (spec P1-02).
    """

    def __init__(
        self,
        step: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        classify: Classifier | None = None,
        activate: bool = True,
    ) -> None:
        self.step = step
        self.attributes = dict(attributes or {})
        self.classify = classify
        self.activate = activate
        self._cm: Any = None
        self._span: Span | None = None
        self._start = 0.0
        self._kb: str | None = None

    def __enter__(self) -> Span:
        obs = get_observability()
        request = current_request()
        self._kb = request.kb if request is not None else None
        span_attributes = _clean(
            {
                attr.STEP: self.step,
                attr.REQUEST_ID: request.request_id if request is not None else None,
                attr.KB: self._kb,
                **self.attributes,
            }
        )
        if self.activate:
            self._cm = obs.tracer.start_as_current_span(
                self.step,
                attributes=span_attributes,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._span = self._cm.__enter__()
        else:
            self._cm = None
            self._span = obs.tracer.start_span(self.step, attributes=span_attributes)
        self._start = time.perf_counter()
        return self._span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        assert self._span is not None and (self._cm is not None or not self.activate)
        outcome = attr.OUTCOME_OK
        if isinstance(exc, Exception):
            outcome = attr.OUTCOME_ERROR
            _mark_error(self._span, exc, self.classify)
        elif exc is not None:
            outcome = attr.OUTCOME_CANCELLED
            self._span.set_attribute(attr.ERROR_TYPE, type(exc).__name__)
        get_observability().metrics.record_step(
            self.step, time.perf_counter() - self._start, outcome=outcome, kb=self._kb
        )
        cm, span, self._cm, self._span = self._cm, self._span, None, None
        if cm is None:
            span.end()
            return None
        return cm.__exit__(exc_type, exc, tb)

    def __call__(self, fn: F) -> F:
        step, attributes, classify = self.step, self.attributes, self.classify
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with layer_span(step, attributes=attributes, classify=classify):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with layer_span(step, attributes=attributes, classify=classify):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
