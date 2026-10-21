"""The query object that travels through the retrieval chain (design s9-1).

LlamaIndex passes only a `QueryBundle` to a retriever and to each node postprocessor. A
`HybridQuery` is a `QueryBundle` that also carries the request's caller access, filters and
deadline in, and a `RetrievalTrace` out: each route's own list, failed routes, timings and
counts, which `NodeWithScore` lists cannot hold. One object per request, so the chain stays
safe to share between threads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.retrieve.deadline import Deadline
from rag.types import RouteHit, ThresholdDecision


@dataclass
class RetrievalTrace:
    """What the retriever and postprocessors report about one request."""

    routes: dict[str, list[RouteHit]] = field(default_factory=dict)
    """Each successful route's own ranked list before fusion (s9-1, P0-10 `routes`)."""
    failed_routes: dict[str, str] = field(default_factory=dict)
    """Route name -> short reason (`timeout`, `embedding: EmbeddingError`, `400 <type>`)."""
    degraded: bool = False
    index_version: str = ""
    """Concrete index the alias pointed to when the hits were read (`kb_<kb>_v<N>`)."""
    candidates: int = 0
    """Chunks after fusion, before postprocessing."""
    candidates_after_dedupe: int | None = None
    duplicates: dict[str, str] = field(default_factory=dict)
    """Removed chunk id -> the kept chunk id with the same body (TextDedupe)."""
    page_key_duplicates: dict[str, str] = field(default_factory=dict)
    """Removed chunk id -> its page key: the other language version of a page (PageKeyDedupe)."""
    rerank_candidates: int | None = None
    """Candidates sent to the reranker (the first `rerank_n` after dedupe)."""
    rerank: list[RouteHit] = field(default_factory=list)
    """The scored candidates, best first, score = rerank score (BatchedTimeoutRerank)."""
    rerank_timeout: bool = False
    """The deadline stopped the reranker before every candidate was scored (s10-1)."""
    threshold: ThresholdDecision | None = None
    not_found: bool = False
    below_threshold: list[NodeWithScore] = field(default_factory=list)
    """With `not_found`: the best nodes by rerank score (RerankThreshold, s10-2)."""
    expanded: int = 0
    """Context pieces added by the ContextExpander."""
    expand_error: str | None = None
    """Why expansion was skipped (`deadline` or the ES error's class name)."""
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class HybridQuery(QueryBundle):
    """`QueryBundle` plus the request context (see the module doc).

    `acl_groups` are the caller's groups from its identity (server side); `filters` are the
    caller's whitelisted filters (`rag.retrieve.filters`); `deadline` is required by
    `EsHybridRetriever`.
    """

    acl_groups: tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    deadline: Deadline | None = None
    rerank_reserve_ms: int = 0
    """Time the query-vector wait leaves for the rerank that follows (at most half of the
    deadline's total). 0, the default, is the plain search: `search_docs` never sets it; the
    ask_docs graph sets it for the searches of round two and later (spec P3-06)."""
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
