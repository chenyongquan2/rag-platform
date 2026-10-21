"""What may be cached, and what a cached value holds (design s12-1, s16-8).

Never written: a result that is not complete or not a plain answer to the question alone.
`search_cacheable` and `ask_cacheable` return the reason a result is refused (one of the
error category names of design s16-8, or a name for the other cases), which goes to the
metric and the span so a drop of the hit rate can be explained.

Written: a complete result. That includes a complete `not_found` result: the index really has
no chunk above the threshold for this question, and the entry is dropped with the next data
generation like any other; it lives for the shorter `cache_not_found_ttl_s` anyway.

A cached value is the result without anything that belongs to the request that made it: no
request id, session id, thread id, resume token, token count, timings and (for a search) no
route lists and no reranker list. `*_from_payload` builds a result for the new request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from rag.cache.keys import CacheKeyFields
from rag.types import AnswerStatus, AskResult, RetrievalResult

ENTRY_VERSION = 1


class Verdict(NamedTuple):
    ok: bool
    reason: str = ""


def search_cacheable(result: RetrievalResult) -> Verdict:
    """A search result is written only when it is complete: both routes answered and the rerank
    finished (the flags are error categories of s16-8)."""
    for category in result.error_categories:
        return Verdict(False, str(category))
    return Verdict(True)


def ask_cacheable(result: AskResult) -> Verdict:
    """An answer is written only when it answers the question asked on its own: no clarification,
    no history, no flag of s16-8 (degraded, rerank_timeout, answer_parse_failed, a model error, a
    stop at a limit), no step that fell back, and the graph itself allows it (`cache_allowed`)."""
    if result.status == AnswerStatus.NEED_CLARIFICATION:
        return Verdict(False, "need_clarification")
    if result.used_history:
        return Verdict(False, "used_history")
    if result.clarified:
        return Verdict(False, "clarified")
    for category in result.error_categories:
        return Verdict(False, str(category))
    if result.coverage_unconfirmed:
        return Verdict(False, "coverage_unconfirmed")
    if result.fallbacks:
        return Verdict(False, "fallback")
    if not result.cache_allowed:
        return Verdict(False, "cache_not_allowed")
    return Verdict(True)


def is_not_found(result: RetrievalResult | AskResult) -> bool:
    if isinstance(result, AskResult):
        return result.status == AnswerStatus.NOT_FOUND
    return result.not_found


# ---- values ----------------------------------------------------------------------------------

_SEARCH_LEFT_OUT = {"request_id", "routes", "rerank", "timings_ms", "error_categories"}
_ASK_LEFT_OUT = {
    "request_id",
    "session_id",
    "thread_id",
    "timings_ms",
    "tokens_used",
    "resume_token",
    "error_categories",
}


def search_payload(result: RetrievalResult) -> dict[str, Any]:
    return result.model_dump(mode="json", exclude=_SEARCH_LEFT_OUT)


def search_from_payload(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    data_generation: int,
    timings_ms: Mapping[str, float],
) -> RetrievalResult:
    return RetrievalResult.model_validate(
        {
            **payload,
            "request_id": request_id,
            "data_generation": data_generation,
            "timings_ms": dict(timings_ms),
        }
    )


def ask_payload(result: AskResult) -> dict[str, Any]:
    return result.model_dump(mode="json", exclude=_ASK_LEFT_OUT)


def ask_from_payload(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    question: str,
    timings_ms: Mapping[str, float],
    data_generation: int,
) -> AskResult:
    return AskResult.model_validate(
        {
            **payload,
            "request_id": request_id,
            "session_id": None,
            "thread_id": "",
            "question": question,
            "tokens_used": 0,
            "resume_token": None,
            "data_generation": data_generation,
            "timings_ms": dict(timings_ms),
        }
    )


@dataclass(frozen=True)
class Entry:
    kind: str
    created_at: float
    key_fields: dict[str, Any]
    payload: dict[str, Any]


def encode_entry(fields: CacheKeyFields, payload: Mapping[str, Any], *, created_at: float) -> str:
    """The stored text: version, kind, `created_at` (Unix seconds) and the key fields for
    debugging, then the payload."""
    return json.dumps(
        {
            "v": ENTRY_VERSION,
            "kind": fields.kind,
            "created_at": created_at,
            "key_fields": fields.as_dict(),
            "payload": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_entry(text: str) -> Entry | None:
    """The entry, or None for anything that is not a current entry (a value another version
    wrote, a cut-off write, junk): the caller treats it as a miss."""
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("v") != ENTRY_VERSION:
            return None
        kind, created_at = data["kind"], data["created_at"]
        key_fields, payload = data["key_fields"], data["payload"]
        if not (
            isinstance(kind, str)
            and isinstance(created_at, (int, float))
            and isinstance(key_fields, dict)
            and isinstance(payload, dict)
        ):
            return None
        return Entry(kind, float(created_at), key_fields, payload)
    except (ValueError, KeyError, TypeError):
        return None
