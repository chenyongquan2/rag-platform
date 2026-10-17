"""Spec P0-09 AC8: exceptions map to error categories and HTTP statuses (design s16-4, s16-8)."""

from __future__ import annotations

from elastic_transport import ApiResponseMeta, ConnectionError, ConnectionTimeout, HttpHeaders
from elasticsearch import ApiError, NotFoundError

from rag.retrieve.filters import FilterError
from rag.retrieve.hybrid import RetrievalError
from rag.serve.errors import ServiceError, error_category, error_message, http_status
from rag.types import ErrorCategory


def _api_error(cls: type[ApiError], status: int) -> ApiError:
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=None,  # type: ignore[arg-type]
    )
    return cls("error", meta, {"error": {"type": "x"}})


def test_error_category_mapping() -> None:
    cases: list[tuple[BaseException, str]] = [
        (ServiceError(ErrorCategory.KB_NOT_FOUND, "r1"), "kb_not_found"),
        (FilterError("unknown filter keys ['x']"), "invalid_filter"),
        (
            RetrievalError("both failed", category=ErrorCategory.RETRIEVAL_FAILED),
            "retrieval_failed",
        ),
        (
            RetrievalError("too late", category=ErrorCategory.RETRIEVAL_TIMEOUT),
            "retrieval_timeout",
        ),
        (ConnectionTimeout("timed out"), "retrieval_timeout"),
        (ConnectionError("refused"), "retrieval_failed"),
        (_api_error(ApiError, 500), "retrieval_failed"),
        (_api_error(NotFoundError, 404), "retrieval_failed"),
        (RuntimeError("bug with secret path C:/x"), "internal_error"),
        (ValueError("bug"), "internal_error"),
    ]
    for exc, expected in cases:
        assert error_category(exc) == expected, exc

    # A category noted on the exception by a span (P0-11) is used when it is one of ours.
    noted = RuntimeError("x")
    noted.error_category = "retrieval_timeout"  # type: ignore[attr-defined]
    assert error_category(noted) == "retrieval_timeout"
    odd = RuntimeError("x")
    odd.error_category = "unclassified"  # type: ignore[attr-defined]
    assert error_category(odd) == "internal_error"


def test_http_status_per_category() -> None:
    assert http_status("invalid_request") == 400
    assert http_status("invalid_filter") == 400
    assert http_status("kb_not_found") == 404
    assert http_status("not_ready") == 503
    assert http_status("retrieval_failed") == 502
    assert http_status("retrieval_timeout") == 504
    assert http_status("internal_error") == 500
    assert http_status("something_new") == 500

    err = ServiceError(ErrorCategory.INVALID_FILTER, "req-9")
    assert err.category == "invalid_filter" and err.request_id == "req-9"
    assert str(err) == "invalid_filter"  # no details in the exception text either
    # Fixed text per category, never built from an exception.
    assert "doc_type" in error_message("invalid_filter")
    assert error_message("internal_error")
    assert error_message("unknown_category") == ""
