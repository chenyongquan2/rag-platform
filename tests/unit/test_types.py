from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag.types import (
    Answer,
    AnswerSentence,
    AnswerStatus,
    Chunk,
    Citation,
    ErrorCategory,
    RetrievalResult,
    RouteHit,
)


def _chunk(chunk_id: str = "hf_docs:a.md:0#0", score: float | None = 0.9) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="hf_docs:a.md:0",
        kb="hf_docs",
        text="Use `push_to_hub` to upload.",
        title_path=["Hub", "Upload"],
        source_rel="a.md",
        source_url="https://example.org/a",
        source_version="abc123",
        score=score,
    )


def test_retrieval_result_fields_and_flags() -> None:
    chunk = _chunk()
    assert chunk.untrusted is True
    assert chunk.suspicious_instruction is False
    assert chunk.metadata == {}

    result = RetrievalResult(
        request_id="req-1",
        kb="hf_docs",
        index_version="v1",
        data_generation=3,
        chunks=[chunk],
        timings_ms={"bm25": 12.5, "rerank": 80.0},
    )
    assert result.degraded is False
    assert result.rerank_timeout is False
    assert result.error_categories == []

    flagged = result.model_copy(update={"degraded": True, "rerank_timeout": True})
    assert flagged.error_categories == [ErrorCategory.DEGRADED, ErrorCategory.RERANK_TIMEOUT]
    # error_categories is part of the JSON that callers receive.
    dumped = json.loads(flagged.model_dump_json())
    assert dumped["error_categories"] == ["degraded", "rerank_timeout"]
    assert RetrievalResult.model_validate(dumped) == flagged
    assert ErrorCategory.DEGRADED == "degraded"
    assert ErrorCategory.ANSWER_PARSE_FAILED == "answer_parse_failed"


def test_answer_roundtrip_json() -> None:
    chunk = _chunk()
    answer = Answer(
        request_id="req-2",
        kb="hf_docs",
        index_version="v1",
        data_generation=3,
        status=AnswerStatus.ANSWERED,
        found=True,
        sentences=[
            AnswerSentence(
                text="Call push_to_hub.",
                citations=[Citation(marker=1, chunk_id=chunk.chunk_id)],
            ),
            AnswerSentence(text="Unsupported claim.", citations=[]),
        ],
        chunks=[chunk],
    )
    assert answer.answer_parse_failed is False
    assert answer.resume_token is None
    assert answer.text == "Call push_to_hub. Unsupported claim."

    restored = Answer.model_validate_json(answer.model_dump_json())
    assert restored == answer

    failed = answer.model_copy(update={"answer_parse_failed": True, "degraded": True})
    assert failed.error_categories == [ErrorCategory.DEGRADED, ErrorCategory.ANSWER_PARSE_FAILED]
    assert json.loads(failed.model_dump_json())["error_categories"] == [
        "degraded",
        "answer_parse_failed",
    ]


def test_retrieval_result_routes_kept_out_of_json() -> None:
    """Each route's own ranked list rides on the result for eval and trace (s9-1, spec P0-10),
    but is not part of the JSON returned to tool callers."""
    bm25 = [RouteHit(chunk_id="c1", doc_id="hf_docs:a.md", rank=1, score=7.5)]
    vector = [
        RouteHit(chunk_id="c2", doc_id="hf_docs:b.md", rank=1),
        RouteHit(chunk_id="c1", doc_id="hf_docs:a.md", rank=2, score=0.8),
    ]
    result = RetrievalResult(
        request_id="req-3",
        kb="hf_docs",
        chunks=[_chunk("c1")],
        routes={"bm25": bm25, "vector": vector},
    )
    assert result.routes["vector"][1].rank == 2
    assert result.routes["vector"][0].score is None
    assert RetrievalResult(request_id="r", kb="hf_docs").routes == {}

    dumped = json.loads(result.model_dump_json())
    assert "routes" not in dumped
    assert "routes" not in result.model_dump()
    with pytest.raises(ValidationError):
        RouteHit(chunk_id="c1", doc_id="hf_docs:a.md", rank=0)
