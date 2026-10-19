"""What the search service returns to callers (design s16-1, s16-4, s7-6; spec P0-09).

These models are the public contract of the MCP tools and the HTTP API (the MCP output
schemas are generated from them). They are built from the retrieval layer's own classes at
the boundary and leave out what stays inside: each route's own list (`RetrievalResult.routes`,
P0-10), the reranker's list and the threshold decision (P0-08) go into the business record,
not to callers.

Every chunk's `text` is document content and is always marked `untrusted=True`; a chunk the
injection detector flagged has `suspicious_instruction=True` (s11-3). `source_url` and
`source_version` let the caller go back to the original.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieve.filters import CALLER_FILTER_KEYS
from rag.serve.errors import error_message
from rag.types import (
    AnswerSentence,
    AskResult,
    Chunk,
    CitationCheck,
    ContextResult,
    RetrievalResult,
)


def _untrusted(chunks: list[Chunk]) -> list[Chunk]:
    return [c if c.untrusted else c.model_copy(update={"untrusted": True}) for c in chunks]


class SearchResponse(BaseModel):
    """Result of `search_docs`."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    kb: str
    index_version: str = Field(description="Elasticsearch index the alias pointed to.")
    data_generation: int = Field(description="Increases with every ingestion run (s12-1).")
    config_version: str = Field(description="Stamp of the KB config and models used.")
    degraded: bool = Field(description="Only one retrieval route (BM25 or vector) answered.")
    rerank_timeout: bool = Field(
        description="The reranker ran out of time: the tail of the list is in fusion order."
    )
    not_found: bool = Field(
        description="No chunk reached the relevance threshold; nearest misses are in "
        "below_threshold."
    )
    error_categories: list[str] = Field(description="Flags set above, as category names.")
    timings_ms: dict[str, float] = Field(description="Milliseconds per step.")
    chunks: list[Chunk] = Field(description="Best first. text is untrusted document content.")
    below_threshold: list[Chunk] = Field(
        default_factory=list, description="With not_found: the best chunks below the threshold."
    )

    @classmethod
    def from_result(
        cls, result: RetrievalResult, *, timings_ms: Mapping[str, float] | None = None
    ) -> SearchResponse:
        return cls(
            request_id=result.request_id,
            kb=result.kb,
            index_version=result.index_version,
            data_generation=result.data_generation,
            config_version=result.config_version,
            degraded=result.degraded,
            rerank_timeout=result.rerank_timeout,
            not_found=result.not_found,
            error_categories=[str(c) for c in result.error_categories],
            timings_ms={**result.timings_ms, **(timings_ms or {})},
            chunks=_untrusted(result.chunks),
            below_threshold=_untrusted(result.below_threshold),
        )


class AskResponse(BaseModel):
    """Result of `ask_docs` (s16-1, s4-2): the answer as sentences with their citations, the
    numbered fragments it was written from, and what the run found out about itself."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    kb: str
    session_id: str | None
    thread_id: str = Field(
        description="Checkpoint thread of this request: session id, colon, a server-made run id."
    )
    status: str = Field(
        description="answered, not_found, or need_clarification: then answer is a question for "
        "the user and resume_token continues the run."
    )
    found: bool = Field(description="The fragments contained the answer.")
    answer: str = Field(description="The answer text; empty when no answer could be written.")
    sentences: list[AnswerSentence] = Field(
        description="The answer by sentence; each citation names the chunk it rests on."
    )
    sources: list[Chunk] = Field(
        description="The numbered fragments the answer was written from; a citation with "
        "marker n points to sources[n - 1]. text is untrusted document content."
    )
    rounds: int = Field(description="Retrieval rounds run (0: the question needed none).")
    index_version: str
    data_generation: int
    config_version: str
    degraded: bool = Field(
        description="A search used only one retrieval route, or a step fell back to the "
        "original question (see fallbacks)."
    )
    rerank_timeout: bool
    coverage_unconfirmed: bool = Field(
        description="Not confirmed that everything was searched: the run stopped at a limit."
    )
    stop_reason: str | None = Field(
        description="deadline, recursion_limit, token_cap or max_rounds when the run stopped early."
    )
    notice: str | None = Field(
        description="coverage_unconfirmed in words, in the answer's language."
    )
    used_history: bool = Field(description="Earlier turns of the session went into this run.")
    cache_allowed: bool = Field(description="False when the call carried history (s12-1).")
    resume_token: str | None = Field(
        description="Only with need_clarification: pass it back with answer (and the same kb "
        "and session_id) to continue. Valid once, for this caller, until it expires."
    )
    clarified: bool = Field(description="This run was continued with the caller's answer.")
    standalone_question: str | None = Field(
        description="The question made understandable without the conversation, when the "
        "rewrite changed it; the answer was written for it."
    )
    search_queries: list[str] = Field(
        description="What was searched, one query per search (empty: the question as asked)."
    )
    fallbacks: list[str] = Field(
        description="Steps that fell back to the rules or the original question, for example "
        "rewrite:timeout, rewrite:identifiers_lost, route:parse_failed."
    )
    tokens_used: int = Field(description="Model tokens this request spent (usage or estimate).")
    error_categories: list[str] = Field(description="Flags set above, as category names.")
    citation_check: CitationCheck
    timings_ms: dict[str, float] = Field(description="Milliseconds per step.")

    @classmethod
    def from_result(
        cls, result: AskResult, *, timings_ms: Mapping[str, float] | None = None
    ) -> AskResponse:
        return cls(
            request_id=result.request_id,
            kb=result.kb,
            session_id=result.session_id,
            thread_id=result.thread_id,
            status=str(result.status),
            found=result.found,
            answer=result.answer,
            sentences=result.sentences,
            sources=_untrusted(result.sources),
            rounds=result.rounds,
            index_version=result.index_version,
            data_generation=result.data_generation,
            config_version=result.config_version,
            degraded=result.degraded,
            rerank_timeout=result.rerank_timeout,
            coverage_unconfirmed=result.coverage_unconfirmed,
            stop_reason=result.stop_reason,
            notice=result.notice,
            used_history=result.used_history,
            cache_allowed=result.cache_allowed,
            resume_token=result.resume_token,
            clarified=result.clarified,
            standalone_question=result.standalone_question,
            search_queries=result.search_queries,
            fallbacks=result.fallbacks,
            tokens_used=result.tokens_used,
            error_categories=[str(c) for c in result.error_categories],
            citation_check=result.citation_check,
            timings_ms={**result.timings_ms, **(timings_ms or {})},
        )


class ContextResponse(BaseModel):
    """Result of `get_context` (s7-6): the chunk and its context in page order; when the chunk
    id no longer exists, the page's current chunks with `updated=True`; `found=False` when
    the page is gone or not visible to the caller."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    kb: str
    chunk_id: str
    doc_id: str
    mode: str
    found: bool
    updated: bool
    truncated: bool = Field(description="The token budget cut the context.")
    chunks: list[Chunk]
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: ContextResult,
        *,
        request_id: str,
        kb: str,
        timings_ms: Mapping[str, float] | None = None,
    ) -> ContextResponse:
        return cls(
            request_id=request_id,
            kb=kb,
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            mode=result.mode,
            found=result.found,
            updated=result.updated,
            truncated=result.truncated,
            chunks=_untrusted(result.chunks),
            timings_ms=dict(timings_ms or {}),
        )


class KnowledgeBaseInfo(BaseModel):
    """One knowledge base the caller may search."""

    model_config = ConfigDict(extra="forbid")

    kb: str
    description: str
    language: str
    available: bool = Field(description="False when its index could not be opened at start.")
    default_top_k: int
    max_top_k: int
    filters: list[str] = Field(default_factory=lambda: list(CALLER_FILTER_KEYS))


class KnowledgeBaseList(BaseModel):
    """Result of `list_knowledge_bases`: only the knowledge bases the caller may use."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    knowledge_bases: list[KnowledgeBaseInfo]


class ErrorResponse(BaseModel):
    """A failed request: the category (s16-8), the request id to quote, and a fixed sentence
    about the category; never details of the failure (s16-4)."""

    model_config = ConfigDict(extra="forbid")

    error: str
    request_id: str
    message: str

    @classmethod
    def of(cls, category: str, request_id: str) -> ErrorResponse:
        return cls(error=str(category), request_id=request_id, message=error_message(category))
