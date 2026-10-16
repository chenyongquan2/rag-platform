from __future__ import annotations

import json

from rag.types import (
    Answer,
    AnswerSentence,
    AnswerStatus,
    Chunk,
    Citation,
    ErrorCategory,
    RetrievalResult,
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
