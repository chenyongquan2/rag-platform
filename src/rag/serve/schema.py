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
from rag.types import Chunk, ContextResult, RetrievalResult


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
