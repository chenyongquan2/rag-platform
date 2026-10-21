"""Spec P5-01 AC3: what is never written to the cache (s12-1, s16-8), what is, and what a
cached value holds."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag.cache.keys import make_fields
from rag.cache.rules import (
    ask_cacheable,
    ask_from_payload,
    ask_payload,
    decode_entry,
    encode_entry,
    search_cacheable,
    search_from_payload,
    search_payload,
)
from rag.testing.agent import pages_result
from rag.testing.serve import fake_chunk
from rag.types import (
    AnswerSentence,
    AnswerStatus,
    AskResult,
    Citation,
    ErrorCategory,
    RetrievalResult,
)


def _search(**over: Any) -> RetrievalResult:
    return pages_result("hf_docs", [1, 2, 3]).model_copy(update={"request_id": "r1", **over})


def _ask(**over: Any) -> AskResult:
    chunks = [fake_chunk("hf_docs", n, rerank_score=0.9) for n in (1, 2)]
    base: dict[str, Any] = {
        "request_id": "r1",
        "kb": "hf_docs",
        "session_id": None,
        "thread_id": "t:1",
        "status": AnswerStatus.ANSWERED,
        "found": True,
        "answer": "Use push_to_hub.",
        "sentences": [
            AnswerSentence(
                text="Use push_to_hub.", citations=[Citation(marker=1, chunk_id=chunks[0].chunk_id)]
            )
        ],
        "sources": chunks,
        "rounds": 1,
        "tokens_used": 321,
        "data_generation": 7,
        "index_version": "kb_hf_docs_v1",
        "config_version": "cfg",
        "question": "How do I push a model?",
        "timings_ms": {"retrieve": 5.0, "total": 9.0},
        "cache_allowed": True,
    }
    base.update(over)
    return AskResult(**base)


# ---- one test per never-cache rule -----------------------------------------------------------


def test_a_clean_search_result_is_cacheable() -> None:
    verdict = search_cacheable(_search())
    assert verdict.ok and verdict.reason == ""


def test_degraded_search_is_never_cached() -> None:
    verdict = search_cacheable(_search(degraded=True))
    assert not verdict.ok and verdict.reason == "degraded"


def test_rerank_timeout_search_is_never_cached() -> None:
    verdict = search_cacheable(_search(rerank_timeout=True))
    assert not verdict.ok and verdict.reason == "rerank_timeout"


def test_a_not_found_search_is_cached_when_complete() -> None:
    """Decided (decisions.md): a complete not-found result is a real answer of the index."""
    result = pages_result("hf_docs", [1, 2, 3], not_found=True)
    assert search_cacheable(result).ok


def test_a_clean_answer_is_cacheable() -> None:
    assert ask_cacheable(_ask()).ok


def test_answer_with_degraded_flag_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(degraded=True, cache_allowed=True))
    assert not verdict.ok and verdict.reason == "degraded"


def test_answer_with_rerank_timeout_is_never_cached() -> None:
    assert ask_cacheable(_ask(rerank_timeout=True)).reason == "rerank_timeout"


def test_answer_parse_failed_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(answer_parse_failed=True, status=AnswerStatus.NOT_FOUND))
    assert not verdict.ok and verdict.reason == "answer_parse_failed"


def test_coverage_unconfirmed_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(coverage_unconfirmed=True, stop_reason="deadline"))
    assert not verdict.ok and verdict.reason in {"coverage_unconfirmed", "agent_deadline"}


@pytest.mark.parametrize(
    "category",
    [
        ErrorCategory.LLM_TIMEOUT,
        ErrorCategory.LLM_ERROR,
        ErrorCategory.BUDGET_EXCEEDED,
        ErrorCategory.STRUCTURED_OUTPUT_FAILED,
    ],
)
def test_a_generation_error_is_never_cached(category: ErrorCategory) -> None:
    verdict = ask_cacheable(_ask(generation_error=category, status=AnswerStatus.NOT_FOUND))
    assert not verdict.ok and verdict.reason == str(category)


def test_a_retrieval_error_in_a_later_round_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(retrieval_error=ErrorCategory.RETRIEVAL_TIMEOUT, degraded=True))
    assert not verdict.ok


def test_token_cap_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(stop_reason="token_cap", coverage_unconfirmed=True))
    assert not verdict.ok


def test_need_clarification_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(status=AnswerStatus.NEED_CLARIFICATION, resume_token="x" * 20))
    assert not verdict.ok and verdict.reason == "need_clarification"


def test_a_call_that_used_history_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(used_history=True))
    assert not verdict.ok and verdict.reason == "used_history"


def test_cache_allowed_false_from_the_graph_is_never_cached() -> None:
    verdict = ask_cacheable(_ask(cache_allowed=False))
    assert not verdict.ok and verdict.reason == "cache_not_allowed"


def test_a_clarified_run_is_never_cached() -> None:
    assert not ask_cacheable(_ask(clarified=True)).ok


def test_a_fallback_step_is_never_cached() -> None:
    assert not ask_cacheable(_ask(fallbacks=["rewrite:timeout"], degraded=True)).ok


def test_a_clean_not_found_answer_is_cached() -> None:
    result = _ask(status=AnswerStatus.NOT_FOUND, found=False, answer="Not found.", sources=[])
    assert ask_cacheable(result).ok


# ---- what a cached value holds ---------------------------------------------------------------


def test_search_payload_round_trip_without_request_specific_data() -> None:
    result = _search(request_id="secret-request-id")
    payload = search_payload(result)
    text = json.dumps(payload)
    assert "secret-request-id" not in text
    assert "routes" not in payload and "rerank" not in payload and "timings_ms" not in payload
    assert "error_categories" not in payload
    back = search_from_payload(
        payload, request_id="new-id", data_generation=9, timings_ms={"cache": 0.4}
    )
    assert back.request_id == "new-id" and back.data_generation == 9
    assert back.timings_ms == {"cache": 0.4}
    assert [c.model_dump() for c in back.chunks] == [c.model_dump() for c in result.chunks]
    assert back.routes == {} and back.rerank == []
    assert back.not_found is False and back.config_version == result.config_version


def test_ask_payload_round_trip_without_request_specific_data() -> None:
    result = _ask(request_id="req-A", session_id="sess", thread_id="sess:9", resume_token=None)
    payload = ask_payload(result)
    text = json.dumps(payload)
    for private in ("req-A", "sess:9", "sess"):
        assert private not in text
    assert "tokens_used" not in payload and "timings_ms" not in payload
    back = ask_from_payload(
        payload,
        request_id="req-B",
        question="how do i push a model",
        timings_ms={"cache": 0.5},
        data_generation=7,
    )
    assert back.request_id == "req-B" and back.session_id is None and back.thread_id == ""
    assert back.tokens_used == 0 and back.timings_ms == {"cache": 0.5}
    assert back.question == "how do i push a model"
    assert back.answer == result.answer and back.sources == result.sources
    assert back.sentences == result.sentences and back.rounds == 1
    assert back.cache_allowed is True and back.resume_token is None


def test_entry_envelope_carries_created_at_and_the_key_fields() -> None:
    fields = make_fields("search", kb="hf_docs", question="Q?", acl_groups=["public"], top_k=3)
    text = encode_entry(fields, {"chunks": []}, created_at=1234.5)
    entry = decode_entry(text)
    assert entry is not None
    assert entry.created_at == 1234.5 and entry.kind == "search"
    assert entry.key_fields["kb"] == "hf_docs" and entry.key_fields["acl"] == ["public"]
    assert entry.payload == {"chunks": []}


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[]", '{"v": 99, "kind": "search", "payload": {}}', '{"v": 1}'],
)
def test_a_value_that_cannot_be_read_is_a_miss_not_an_error(text: str) -> None:
    assert decode_entry(text) is None
