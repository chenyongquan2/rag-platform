"""Data classes passed between layers (design s16-1, s16-2, s16-8).

Framework types (LlamaIndex `NodeWithScore`, LangChain messages) stay inside their layer;
everything that crosses a layer boundary is one of these models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


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
    # Search service (spec P0-09): arguments outside their range or shape; a knowledge base
    # that does not exist or that the caller may not use (one category for both, s16-4); the
    # reranker still loading; anything unexpected (details only in the logs).
    INVALID_REQUEST = "invalid_request"
    KB_NOT_FOUND = "kb_not_found"
    NOT_READY = "not_ready"
    INTERNAL_ERROR = "internal_error"
    # ask_docs graph (spec P3-01, s16-8): the run was stopped by its total deadline or its
    # recursion limit, or by the per-request token cap; the answer uses what was found so far.
    AGENT_DEADLINE = "agent_deadline"
    TOKEN_CAP_REACHED = "token_cap_reached"
    # ask_docs clarification (spec P3-02, s13-3): a `resume_token` that is unknown, expired, already
    # used, issued to another caller, knowledge base or session, or whose run is not waiting for
    # an answer. One category for all of them, so a caller cannot tell which tokens exist.
    INVALID_RESUME_TOKEN = "invalid_resume_token"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEED_CLARIFICATION = "need_clarification"
    NOT_FOUND = "not_found"


class ContextPiece(BaseModel):
    """Text of another chunk of the same page, added around a retrieved chunk (design s10-3,
    s7-2): a neighbour (same doc_id, chunk_index nearby) or a chunk of its heading section."""

    chunk_id: str
    chunk_index: int
    text: str


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
    rerank_score: float | None = None
    """The cross-encoder's score in [0, 1] (sigmoid, s10-2), comparable across knowledge bases;
    None when the chunk was not reranked (rerank off, or `rerank_timeout` before its batch).
    `score` keeps the fusion score."""
    context: list[ContextPiece] = Field(default_factory=list)
    """Neighbour or section chunks added by context expansion, in page order (s10-3)."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThresholdDecision(BaseModel):
    """What the rerank-score threshold did to one request's list (design s9-1, s10-2, s16-8).

    `applied` is False after a rerank timeout (nothing is cut by the threshold; chunks below it
    only move behind the unscored ones) and when no chunk had a rerank score; `passed` says
    whether the best rerank score reached the threshold (None without scores); `kept` and
    `dropped` count chunks before the top_k cut.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float
    top_score: float | None = None
    applied: bool = True
    passed: bool | None = None
    kept: int = 0
    dropped: int = 0


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
        stop_reason = getattr(self, "stop_reason", None)
        if stop_reason in ("deadline", "recursion_limit"):
            categories.append(ErrorCategory.AGENT_DEADLINE)
        elif stop_reason == "token_cap":
            categories.append(ErrorCategory.TOKEN_CAP_REACHED)
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
    rerank: list[RouteHit] = Field(default_factory=list)
    """The reranker's own list: every candidate it scored, best first, `score` = rerank score
    (before the threshold and the top_k cut; for business records and eval). Left out of the
    search_docs response like `routes`."""
    threshold: ThresholdDecision | None = None
    """What the rerank threshold did; None when the chain has no threshold step."""
    not_found: bool = False
    """No chunk reached the rerank threshold (s10-2): `chunks` is empty, the best ones are in
    `below_threshold`. Not an error category."""
    below_threshold: list[Chunk] = Field(default_factory=list)
    """With `not_found`: the best chunks by rerank score (3 by default), kept apart from
    `chunks` so callers can show "nearest matches" without treating them as found."""
    config_version: str = ""
    """Stamp of the retrieval config that produced the result (KB config, embedding and
    reranker model); with `index_version` and `data_generation` it says what was searched."""


class ContextResult(BaseModel):
    """Answer of `get_context` (design s7-6): a chunk and the chunks around it, in page order.

    When the chunk id no longer exists (the page was re-chunked or deleted after the caller got
    the id), `chunks` is the page's current content (by the doc_id in the chunk id) and
    `updated` is set; `found` is False when the page is gone too. `truncated`: the token budget
    cut the list.
    """

    chunk_id: str
    doc_id: str
    mode: str
    found: bool = True
    updated: bool = False
    truncated: bool = False
    chunks: list[Chunk] = Field(default_factory=list)


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


StopReason = Literal["deadline", "recursion_limit", "token_cap", "max_rounds"]
"""Why an `ask_docs` run stopped searching before it confirmed it had searched enough (s13-3):
its total deadline, its recursion limit, the per-request token cap, or the KB's `max_rounds`."""


class AskResult(_ResultMeta):
    """Output of the `ask_docs` graph (design s4-2, s13-3, s16-1).

    `sources` are the numbered fragments the answer was written from (the citation with marker n
    points to `sources[n - 1]`); after a run cut short, the chunks found so far. `rounds` counts
    retrieval rounds (0 when the question needed none). `status` is `need_clarification` when the
    run stopped to ask the caller (spec P3-02): `answer` is then the question to ask and
    `resume_token` the opaque token that continues the run (`ask_docs(..., resume_token=...,
    answer=...)`); `clarified` marks a result of a run that was continued that way.
    `standalone_question` and `search_queries` are what the rewrite made of the question (empty:
    searched as asked); `fallbacks` lists the steps that fell back to the rules or the original
    question (`route:timeout`, `rewrite:parse_failed`, ...), which also sets `degraded`.
    `coverage_unconfirmed` is the s13-3 flag "not confirmed that everything was searched": set
    with `stop_reason`, and `notice` says so in the answer's language. `cache_allowed` is False
    for a call that carried history (the exact cache neither reads nor writes it, s12-1).
    """

    status: AnswerStatus = AnswerStatus.ANSWERED
    found: bool = False
    answer: str = ""
    sentences: list[AnswerSentence] = Field(default_factory=list)
    sources: list[Chunk] = Field(default_factory=list)
    rounds: int = Field(default=0, ge=0)
    session_id: str | None = None
    thread_id: str = ""
    answer_parse_failed: bool = False
    generation_error: ErrorCategory | None = None
    citation_check: CitationCheck = Field(default_factory=CitationCheck)
    config_version: str = ""
    coverage_unconfirmed: bool = False
    stop_reason: StopReason | None = None
    notice: str | None = None
    tokens_used: int = Field(default=0, ge=0)
    used_history: bool = False
    cache_allowed: bool = True
    resume_token: str | None = None
    clarified: bool = False
    question: str = ""
    """The question as the caller first asked it (a resumed call carries only the answer)."""
    standalone_question: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)

    @property
    def chunks(self) -> list[Chunk]:
        """Same as `sources`: what `RequestScope.set_result` counts as hits."""
        return self.sources
