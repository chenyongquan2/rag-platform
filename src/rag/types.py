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
    """The score of the list the chunk came from: from the hybrid retriever, the RRF fusion
    score (sum of 1 / (k + rank) over the routes, s9-3), not a BM25 score or a similarity;
    each route's own rank and score are in `metadata["route_ranks"]` / `["route_scores"]`."""
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
        categories = [category for is_set, category in flags if is_set]
        generation_error = getattr(self, "generation_error", None)
        if generation_error is not None:
            categories.append(ErrorCategory(generation_error))
        return categories


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


class CitationCheck(BaseModel):
    """Level 1 of the citation check (design s11-2), done on every answer by rules."""

    removed_markers: list[int] = Field(default_factory=list)
    """`[n]` numbers that matched no fragment of this answer's context; removed from the answer."""
    unsupported_sentences: int = Field(default=0, ge=0)
    """Sentences left without any valid citation."""
    parse_attempts: int = Field(default=0, ge=0)
    """Model calls made for this answer (1, or 2 after a retry; 0 when no fragment was given)."""


class Answer(_ResultMeta):
    """Output of the `ask_docs` tool (design s11-1, s16-1).

    `chunks` are the numbered fragments the answer was generated from, in order: the citation
    with marker n points to `chunks[n - 1]`. With `answer_parse_failed` or `generation_error`
    there are no sentences and `status` is `not_found`: the caller gets the chunks only, and the
    error category says why (s16-8).
    """

    status: AnswerStatus = AnswerStatus.ANSWERED
    found: bool = False
    sentences: list[AnswerSentence] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    answer_parse_failed: bool = False
    generation_error: ErrorCategory | None = None
    """Why the model gave no answer: `llm_timeout`, `llm_error` or `budget_exceeded` (s16-8)."""
    citation_check: CitationCheck = Field(default_factory=CitationCheck)
    resume_token: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)
