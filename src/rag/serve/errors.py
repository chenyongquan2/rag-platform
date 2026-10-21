"""Error categories of the search service and what callers see of them (design s16-4, s16-8).

Callers get only the category, the request id and a fixed sentence per category; the exception
and its text go to the log line with the same request id. `error_category` is also the
classifier passed to `request_span`, so the span, the error metric, the business record and the
response name the same category.
"""

from __future__ import annotations

from elastic_transport import ConnectionTimeout, TransportError
from elasticsearch import ApiError

from rag.observe.spans import EXCEPTION_CATEGORY_ATTR
from rag.retrieve.filters import CALLER_FILTER_KEYS, FilterError
from rag.retrieve.hybrid import RetrievalError
from rag.types import ErrorCategory

_KNOWN = {category.value for category in ErrorCategory}

_STATUS: dict[str, int] = {
    ErrorCategory.INVALID_REQUEST: 400,
    ErrorCategory.INVALID_FILTER: 400,
    ErrorCategory.KB_NOT_FOUND: 404,
    ErrorCategory.NOT_READY: 503,
    ErrorCategory.RETRIEVAL_FAILED: 502,
    ErrorCategory.RETRIEVAL_TIMEOUT: 504,
    ErrorCategory.INTERNAL_ERROR: 500,
    ErrorCategory.AGENT_DEADLINE: 504,
    ErrorCategory.INVALID_RESUME_TOKEN: 400,
    ErrorCategory.SERVER_SHUTDOWN: 503,
    ErrorCategory.UNAUTHORIZED: 401,
    ErrorCategory.FORBIDDEN: 403,
    ErrorCategory.RATE_LIMITED: 429,
    ErrorCategory.BUDGET_EXCEEDED: 429,
}

_MESSAGES: dict[str, str] = {
    ErrorCategory.INVALID_REQUEST: (
        "The request is malformed or an argument is out of range: query must be non-empty "
        "text, top_k between 1 and the knowledge base's maximum, deadline_ms positive and "
        "within the service maximum, question (ask_docs) non-empty text, session_id (ask_docs) "
        "1 to 64 letters, digits, dot, dash or underscore, answer (ask_docs) non-empty text "
        "and only together with resume_token, chunk_id as returned by "
        "search_docs, mode one of chunk, neighbors, parent, page."
    ),
    ErrorCategory.INVALID_FILTER: (
        "filters may only use the keys "
        + ", ".join(CALLER_FILTER_KEYS)
        + " (doc_type and language: a string or a list of strings; source_rel_prefix: one "
        "path prefix)."
    ),
    ErrorCategory.KB_NOT_FOUND: (
        "No knowledge base with this name is available to you; call list_knowledge_bases."
    ),
    ErrorCategory.NOT_READY: "The service is still loading its models; retry in a minute.",
    ErrorCategory.RETRIEVAL_FAILED: "The search backend failed; retry later.",
    ErrorCategory.RETRIEVAL_TIMEOUT: (
        "The search did not finish within the deadline; retry, perhaps with a larger deadline_ms."
    ),
    ErrorCategory.INTERNAL_ERROR: "Internal error; quote the request id when reporting it.",
    ErrorCategory.AGENT_DEADLINE: (
        "The request waited for an earlier request of the same session until its deadline_ms "
        "ran out; retry, perhaps with a larger deadline_ms."
    ),
    ErrorCategory.INVALID_RESUME_TOKEN: (
        "The resume_token is not valid: it is unknown, expired, already used, or was issued "
        "for another caller, knowledge base or session. A token can answer one question "
        "once; ask the question again to get a new one."
    ),
}


_MESSAGES[ErrorCategory.SERVER_SHUTDOWN] = (
    "The server is shutting down and stopped this request; ask again in a minute."
)
_MESSAGES[ErrorCategory.UNAUTHORIZED] = (
    "A valid bearer access token is required (Authorization: Bearer <token>); it is missing, "
    "expired or not issued for this service."
)
_MESSAGES[ErrorCategory.FORBIDDEN] = (
    "The access token is valid but its scopes grant no access to this service, or name more "
    "than one caller; ask for a token with exactly one of the service's scopes."
)
_MESSAGES[ErrorCategory.RATE_LIMITED] = (
    "Too many requests from this caller in the current minute; retry after the number of "
    "seconds in Retry-After (HTTP) or in a minute."
)
_MESSAGES[ErrorCategory.BUDGET_EXCEEDED] = (
    "The daily model budget of this caller or of the service is used up; ask_docs is refused "
    "until the next day (UTC), search_docs still works."
)


class ServiceError(Exception):
    """A failed request as callers see it: category and request id, nothing else (and, for
    `rate_limited` and `budget_exceeded`, when to retry). `detail`
    (what was wrong, for the log line) and the exception that caused it (`__cause__`) are
    logged, never returned; `str()` is the category alone."""

    def __init__(
        self,
        category: str,
        request_id: str,
        *,
        detail: str = "",
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(str(category))
        self.category = str(category)
        self.request_id = request_id
        self.detail = detail
        self.retry_after_s = retry_after_s  # HTTP sends it as Retry-After (whole seconds)

    def __repr__(self) -> str:
        return f"ServiceError({self.category!r}, request_id={self.request_id!r})"


def error_category(exc: BaseException) -> str:
    """The s16-8 category of an exception raised while serving a request."""
    if isinstance(exc, ServiceError):
        return exc.category
    if isinstance(exc, FilterError):
        return ErrorCategory.INVALID_FILTER
    if isinstance(exc, RetrievalError):
        return str(exc.category)
    if isinstance(exc, ConnectionTimeout):
        return ErrorCategory.RETRIEVAL_TIMEOUT
    if isinstance(exc, (ApiError, TransportError)):
        return ErrorCategory.RETRIEVAL_FAILED
    noted = getattr(exc, EXCEPTION_CATEGORY_ATTR, None)
    if isinstance(noted, str) and noted in _KNOWN:
        return noted
    return ErrorCategory.INTERNAL_ERROR


def http_status(category: str) -> int:
    return _STATUS.get(str(category), 500)


def error_message(category: str) -> str:
    """Fixed text for the category (empty for an unknown one); never built from an exception."""
    return _MESSAGES.get(str(category), "")
