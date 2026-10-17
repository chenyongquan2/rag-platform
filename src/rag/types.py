"""Data classes passed between layers (design s16-1, s16-2, s16-8).

Framework types (LlamaIndex `NodeWithScore`, LangChain messages) stay inside their layer;
everything that crosses a layer boundary is one of these models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ErrorCategory(StrEnum):
    """Error category names written to the result, the business log and metrics (design s16-8)."""

    DEGRADED = "degraded"
    RERANK_TIMEOUT = "rerank_timeout"
    ANSWER_PARSE_FAILED = "answer_parse_failed"
    # Chat model calls (spec P1-01); `rag.llm.chat_error_category` maps exceptions to these.
    LLM_TIMEOUT = "llm_timeout"
    LLM_ERROR = "llm_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    STRUCTURED_OUTPUT_FAILED = "structured_output_failed"
    # Retrieval (spec P0-07): both routes failed, or the deadline passed before an answer;
    # a caller filter outside the whitelist (s16-1).
    RETRIEVAL_FAILED = "retrieval_failed"
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    INVALID_FILTER = "invalid_filter"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEED_CLARIFICATION = "need_clarification"
    NOT_FOUND = "not_found"


class Chunk(BaseModel):
    """One retrieved segment. `text` is source content and is always untrusted (design s16-1)."""

    chunk_id: str
    doc_id: str
    kb: str
    text: str
    untrusted: bool = True
    suspicious_instruction: bool = False
    title_path: list[str] = Field(default_factory=list)
    source_rel: str = ""
    source_url: str | None = None
    source_version: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _ResultMeta(BaseModel):
    """Fields every query result carries (design s16-1)."""

    request_id: str
    kb: str
    index_version: str = ""
    data_generation: int = 0
    degraded: bool = False
    rerank_timeout: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_categories(self) -> list[ErrorCategory]:
        """Set flags as category names; included in the JSON returned to callers."""
        flags = [
            (self.degraded, ErrorCategory.DEGRADED),
            (self.rerank_timeout, ErrorCategory.RERANK_TIMEOUT),
            (getattr(self, "answer_parse_failed", False), ErrorCategory.ANSWER_PARSE_FAILED),
        ]
        return [category for is_set, category in flags if is_set]


class RouteHit(BaseModel):
    """One entry of a single route's own ranked list (BM25 or vector), before fusion (s9-1)."""

    chunk_id: str
    doc_id: str
    rank: int = Field(ge=1)
    score: float | None = None


class RetrievalResult(_ResultMeta):
    """Output of the retrieval layer and of the `search_docs` tool."""

    chunks: list[Chunk] = Field(default_factory=list)
    routes: dict[str, list[RouteHit]] = Field(default_factory=dict)
    """Each route's own ranked list (`bm25`, `vector`, ...), for per-route eval metrics and
    traces (s9-1, s14-1). Serialized like every other field; the search_docs response (P0-09)
    leaves it out at its boundary (`model_dump(exclude={"routes"})`), so tool callers do not
    pay for about 40 extra doc ids per call."""


class Citation(BaseModel):
    """A `[n]` marker in the answer text and the chunk it points to."""

    marker: int = Field(ge=1)
    chunk_id: str


class AnswerSentence(BaseModel):
    """One parsed answer sentence; empty `citations` means the sentence has no support."""

    text: str
    citations: list[Citation] = Field(default_factory=list)


class Answer(_ResultMeta):
    """Output of the `ask_docs` tool (design s11-1, s16-1)."""

    status: AnswerStatus = AnswerStatus.ANSWERED
    found: bool = False
    sentences: list[AnswerSentence] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    answer_parse_failed: bool = False
    resume_token: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)
