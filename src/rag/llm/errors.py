"""Error categories of chat model calls (spec P1-01, design s16-8).

Callers catch the exception, put the category into the result, the business log and metrics,
and show only the category and the request id to the caller; the text goes to the log.
"""

from __future__ import annotations

import httpx
import openai
from langchain_core.exceptions import OutputParserException

from rag.cost import BudgetExceeded
from rag.llm.structured import StructuredOutputError
from rag.types import ErrorCategory


def chat_error_category(exc: BaseException) -> ErrorCategory | None:
    """Category of an exception raised by a chat model call, or None if it is not one of ours
    (a bug, which callers should not hide behind a category).

    - `budget_exceeded`: the call was refused before sending (spent + estimate > stop).
    - `structured_output_failed`: the answer did not parse into the requested schema.
    - `llm_timeout`: connect or read timeout, after the profile's retries.
    - `llm_error`: any other gateway failure (connection refused, 4xx, 429, 5xx).
    """
    if isinstance(exc, BudgetExceeded):
        return ErrorCategory.BUDGET_EXCEEDED
    if isinstance(exc, StructuredOutputError | OutputParserException):
        return ErrorCategory.STRUCTURED_OUTPUT_FAILED
    if isinstance(exc, openai.APITimeoutError | httpx.TimeoutException | TimeoutError):
        return ErrorCategory.LLM_TIMEOUT
    if isinstance(exc, openai.APIError | httpx.TransportError | httpx.HTTPStatusError):
        return ErrorCategory.LLM_ERROR
    return None
