"""Chat model factory, structured output and error categories (spec P1-01, design s11-1)."""

from rag.config import ChatProfile, ChatProfileName
from rag.llm.errors import chat_error_category
from rag.llm.factory import PROFILES, make_chat_model, resolve_chat_profile
from rag.llm.structured import StructuredOutputError, StructuredOutputMethod, structured_output
from rag.llm.transport import AsyncLedgerTransport, LedgerTransport, parse_usage

__all__ = [
    "PROFILES",
    "AsyncLedgerTransport",
    "ChatProfile",
    "ChatProfileName",
    "LedgerTransport",
    "StructuredOutputError",
    "StructuredOutputMethod",
    "chat_error_category",
    "make_chat_model",
    "parse_usage",
    "resolve_chat_profile",
    "structured_output",
]
