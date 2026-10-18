"""`EsHybridRetriever`: BM25 and kNN in one `_msearch`, fused with RRF (design s9-1 to s9-3).

One request carries both searches, each with the same filter (knowledge base, the caller's ACL
groups, whitelisted caller filters; `rag.retrieve.filters`):

- BM25: `multi_match` over `content` and `title_path` (title boosted), `top_k_per_route` hits;
- vector: `knn` on `embedding` with the filter inside `knn` (filtered during the HNSW search,
  not afterwards), `k = top_k_per_route`, `num_candidates` from the KB config.

Elasticsearch answers each search of an `_msearch` separately, so one route can fail while the
other succeeds (s16-8): a route whose answer is an error or `timed_out` is left out, the other
is used and the trace says `degraded`. The query vector comes from the query-profile embedding
client (P0-05); if it fails, or is not ready while enough time is left for the search, only the
BM25 search is sent (degraded). Only when no route answers does the retriever raise
`RetrievalError` (`retrieval_failed` or `retrieval_timeout`).

Deadline: the request's `Deadline` (s16-2) bounds everything. A query vector in the L0 cache
is used whenever the cache answers within half of the remaining time. The gateway call is
waited for at most until min(`es_reserve_ms`, `es_reserve_ratio` x the request's time limit)
before the deadline, the time kept for the search; the `_msearch` gets the remaining time as its
HTTP time-out, and each search inside it an Elasticsearch `timeout` shorter by `es_margin_ms` (at
most half of the remaining time, so the searches keep the other half), so a
slow route is reported as timed out before the HTTP request itself is cut off.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from elastic_transport import ConnectionTimeout, TransportError
from elasticsearch import ApiError, Elasticsearch
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.config import RetrievalConfig
from rag.index.mapping import VECTOR_FIELD
from rag.retrieve.deadline import Deadline
from rag.retrieve.filters import build_es_filter
from rag.retrieve.fusion import rrf_fuse
from rag.retrieve.query import HybridQuery, RetrievalTrace
from rag.types import ErrorCategory, RouteHit

__all__ = [
    "BM25",
    "SOURCE_FIELDS",
    "VECTOR",
    "EsHybridRetriever",
    "HybridQuery",
    "RetrievalError",
    "RetrievalTrace",
]

logger = logging.getLogger(__name__)

BM25 = "bm25"
VECTOR = "vector"
ROUTES: tuple[str, ...] = (BM25, VECTOR)
TIMEOUT = "timeout"
EMBED_TIMEOUT = "embedding: timeout"

SOURCE_FIELDS = (
    "chunk_id",
    "doc_id",
    "kb",
    "content",
    "title_path",
    "source_rel",
    "source_url",
    "source_version",
    "doc_type",
    "language",
    "chunk_index",
    "parent_id",
    "images",
    "suspicious_instruction",
)
"""Fields read from `_source`: what a `Chunk` needs; never the vector or the ACL groups."""

DEFAULT_EMBED_THREADS = 4
"""Threads for query embeddings (the factory passes `embed_max_concurrency`)."""


class RetrievalError(RuntimeError):
    """No route answered. `category` is `retrieval_timeout` when every route ran out of time,
    else `retrieval_failed`; `route_errors` holds the reason per route (also logged)."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory,
        route_errors: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.route_errors = dict(route_errors or {})


def _is_timeout(reason: str) -> bool:
    return reason in (TIMEOUT, EMBED_TIMEOUT)


def _route_failure(response: Mapping[str, Any]) -> str | None:
    """Why one search of the `_msearch` did not answer, or None when it did."""
    if "error" in response:
        error = response.get("error") or {}
        kind = error.get("type", "unknown") if isinstance(error, Mapping) else str(error)
        return f"{response.get('status', '?')} {kind}"
    if response.get("timed_out"):
        return TIMEOUT
    return None


class EsHybridRetriever(BaseRetriever):
    """Hybrid retriever over one knowledge base's alias.

    Needs a `HybridQuery` (caller ACL groups, filters, deadline). Returns fused `NodeWithScore`s
    (score = RRF score, best first) whose metadata holds the chunk fields plus `route_ranks` and
    `route_scores`; each route's own list, failed routes, index version, candidate count and
    timings go into `query.trace`.
    """

    def __init__(
        self,
        es: Elasticsearch,
        embed_model: BaseEmbedding,
        *,
        kb: str,
        index: str,
        config: RetrievalConfig,
        es_reserve_ms: int,
        es_margin_ms: int,
        es_reserve_ratio: float = 0.2,
        embed_threads: int = DEFAULT_EMBED_THREADS,
        routes: Sequence[str] = ROUTES,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        unknown = [r for r in routes if r not in ROUTES]
        if unknown or not routes:
            raise ValueError(f"routes must be a non-empty subset of {ROUTES}, got {list(routes)}")
        self.routes = tuple(r for r in ROUTES if r in routes)
        """Routes this retriever sends (both by default; one for a single-route eval run, P0-12).
        A route left out here is not a failed route: the result is not degraded."""
        if es_reserve_ms < 0 or es_margin_ms < 0:
            raise ValueError("es_reserve_ms and es_margin_ms must be >= 0")
        if not 0 <= es_reserve_ratio < 1:
            raise ValueError(f"es_reserve_ratio must be in [0, 1), got {es_reserve_ratio}")
        self._es = es
        self._embed_model = embed_model
        self.kb = kb
        self.index = index
        self.config = config
        self.es_reserve_ms = es_reserve_ms
        self.es_reserve_ratio = es_reserve_ratio
        self.es_margin_ms = es_margin_ms
        # Query embeddings run in these threads so the wait can stop at the deadline; a call
        # still running then finishes in the background (bounded by the query profile's own
        # deadline, P0-05). If all threads are busy with such calls, new questions wait for a
        # thread, time out and run BM25 only until the gateway answers again.
        self._executor = ThreadPoolExecutor(
            max_workers=embed_threads, thread_name_prefix="query-embed"
        )
        # L0 cache lookups get their own threads: they must still answer while the gateway
        # threads are all waiting on a gateway that does not respond.
        self._cache_executor = ThreadPoolExecutor(
            max_workers=embed_threads, thread_name_prefix="query-embed-cache"
        )

    @property
    def embed_model(self) -> BaseEmbedding:
        return self._embed_model

    def es_reserve_s(self, deadline: Deadline) -> float:
        """Time kept for the `_msearch` while waiting for the gateway: min(`es_reserve_ms`,
        `es_reserve_ratio` x the request's whole time limit), so a short deadline is not used
        up by the reserve alone (an `_msearch` takes about 20 ms)."""
        cap = self.es_reserve_ms / 1000.0
        if deadline.total_s <= 0:
            return cap
        return min(cap, self.es_reserve_ratio * deadline.total_s)

    def close(self) -> None:
        """Stop the query-embedding threads (calls already running finish on their own)."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cache_executor.shutdown(wait=False, cancel_futures=True)

    # ---- request bodies ---------------------------------------------------------------------

    def bm25_body(self, query: str, es_filter: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", f"title_path^{self.config.title_boost:g}"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": es_filter,
                }
            },
            "size": self.config.top_k_per_route,
            "_source": {"includes": list(SOURCE_FIELDS)},
            "track_total_hits": False,
        }

    def knn_body(self, vector: list[float], es_filter: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "knn": {
                "field": VECTOR_FIELD,
                "query_vector": vector,
                "k": self.config.top_k_per_route,
                "num_candidates": self.config.num_candidates,
                "filter": es_filter,
            },
            "size": self.config.top_k_per_route,
            "_source": {"includes": list(SOURCE_FIELDS)},
            "track_total_hits": False,
        }

    # ---- steps ------------------------------------------------------------------------------

    def _embed(self, query: HybridQuery) -> tuple[list[float] | None, str | None]:
        """The query vector, or None and the reason the vector route cannot run."""
        if query.embedding is not None:
            return list(query.embedding), None
        deadline = query.deadline
        assert deadline is not None
        cached = self._cached_vector(query.query_str, deadline)
        if cached is not None:
            return cached, None
        wait_s = deadline.remaining_s() - self.es_reserve_s(deadline)
        if wait_s <= 0:
            return None, EMBED_TIMEOUT
        future = self._executor.submit(self._embed_model.get_query_embedding, query.query_str)
        try:
            return list(future.result(timeout=wait_s)), None
        except Exception as exc:  # any failure only costs the vector route (s16-8)
            if not future.done():
                # A call already running finishes in the background (and fills the L0 cache).
                future.cancel()
                logger.warning("query embedding not ready after %.2f s; BM25 only", wait_s)
                return None, EMBED_TIMEOUT
            logger.warning("query embedding failed; BM25 only: %s: %s", type(exc).__name__, exc)
            return None, f"embedding: {type(exc).__name__}"

    def _cached_vector(self, text: str, deadline: Deadline) -> list[float] | None:
        """The L0 cache's vector for `text` (no gateway call), waited for at most half of the
        remaining time; None on a miss, a cache error or no answer in time. Checked before the
        gateway budget applies, so a cached question keeps its vector route under any deadline.
        On a miss the gateway client looks the cache up once more (one extra Redis GET)."""
        lookup = getattr(self._embed_model, "cached_query_embedding", None)
        if not callable(lookup):
            return None
        wait_s = deadline.remaining_s() / 2.0
        if wait_s <= 0:
            return None
        future = self._cache_executor.submit(lookup, text)
        try:
            vector = future.result(timeout=wait_s)
        except Exception as exc:  # a cache problem only means "not cached"
            future.cancel()
            logger.warning("L0 cache lookup failed or slow: %s", type(exc).__name__)
            return None
        return list(vector) if vector is not None else None

    def _resolve_index(self, deadline: Deadline) -> str:
        """The index behind the alias, for a result without hits (hits name it themselves).
        Empty when the alias is unknown, the request fails or no time is left."""
        remaining_s = deadline.remaining_s()
        if remaining_s <= 0:
            return ""
        try:
            resp = self._es.options(
                request_timeout=remaining_s, ignore_status=404
            ).indices.get_alias(name=self.index)
        except (TransportError, ApiError) as exc:
            logger.warning("alias %s not resolved: %s", self.index, type(exc).__name__)
            return ""
        meta = getattr(resp, "meta", None)
        if meta is not None and getattr(meta, "status", 200) != 200:
            return ""
        body = getattr(resp, "body", resp)
        return ",".join(sorted(body))

    def _msearch(
        self, query: HybridQuery, searches: list[tuple[str, dict[str, Any]]], trace: RetrievalTrace
    ) -> list[Mapping[str, Any]]:
        assert query.deadline is not None
        remaining_s = query.deadline.remaining_s()
        if remaining_s <= 0:
            raise RetrievalError(
                "deadline passed before the search was sent",
                category=ErrorCategory.RETRIEVAL_TIMEOUT,
                route_errors={**trace.failed_routes, **{name: TIMEOUT for name, _ in searches}},
            )
        # The margin lets ES report a slow route as timed out before the HTTP request is cut;
        # it takes at most half of what is left, so the searches always get the other half
        # (after a slow gateway only the reserve, e.g. 160 ms of an 800 ms limit, is left).
        remaining_ms = remaining_s * 1000.0
        margin_ms = min(float(self.es_margin_ms), remaining_ms / 2.0)
        per_search_ms = max(1, int(remaining_ms - margin_ms))
        lines: list[Mapping[str, Any]] = []
        for _, body in searches:
            lines.append({})
            lines.append({**body, "timeout": f"{per_search_ms}ms"})
        try:
            resp = self._es.options(request_timeout=remaining_s).msearch(
                index=self.index, searches=lines
            )
        except ConnectionTimeout as exc:
            raise RetrievalError(
                f"_msearch on {self.index} timed out after {remaining_s:.2f} s",
                category=ErrorCategory.RETRIEVAL_TIMEOUT,
                route_errors={name: TIMEOUT for name, _ in searches},
            ) from exc
        except (TransportError, ApiError) as exc:
            raise RetrievalError(
                f"_msearch on {self.index} failed: {type(exc).__name__}",
                category=ErrorCategory.RETRIEVAL_FAILED,
                route_errors={name: type(exc).__name__ for name, _ in searches},
            ) from exc
        return list(resp["responses"])

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if not isinstance(query_bundle, HybridQuery):
            raise TypeError(
                "EsHybridRetriever needs a HybridQuery carrying the caller's ACL groups, filters "
                f"and deadline, got {type(query_bundle).__name__}"
            )
        query = query_bundle
        es_filter = build_es_filter(kb=self.kb, acl_groups=query.acl_groups, filters=query.filters)
        if query.deadline is None:
            raise ValueError("HybridQuery.deadline is required: every query has a time limit")
        trace = query.trace
        if query.deadline.expired:
            raise RetrievalError(
                "deadline passed before retrieval started",
                category=ErrorCategory.RETRIEVAL_TIMEOUT,
            )

        start = time.perf_counter()
        vector, vector_error = self._embed(query) if VECTOR in self.routes else (None, None)
        embedded = time.perf_counter()
        trace.timings_ms["embed"] = (embedded - start) * 1000.0
        if vector_error is not None:
            trace.failed_routes[VECTOR] = vector_error

        searches: list[tuple[str, dict[str, Any]]] = []
        if BM25 in self.routes:
            searches.append((BM25, self.bm25_body(query.query_str, es_filter)))
        if vector is not None:
            searches.append((VECTOR, self.knn_body(vector, es_filter)))
        if not searches:  # vector-only retriever whose query embedding failed
            raise RetrievalError(
                f"no retrieval route answered on {self.index}: {dict(trace.failed_routes)}",
                category=(
                    ErrorCategory.RETRIEVAL_TIMEOUT
                    if all(_is_timeout(r) for r in trace.failed_routes.values())
                    else ErrorCategory.RETRIEVAL_FAILED
                ),
                route_errors=dict(trace.failed_routes),
            )
        responses = self._msearch(query, searches, trace)
        searched = time.perf_counter()
        trace.timings_ms["search"] = (searched - embedded) * 1000.0

        sources: dict[str, dict[str, Any]] = {}
        for (name, _), response in zip(searches, responses, strict=True):
            failure = _route_failure(response)
            if failure is not None:
                trace.failed_routes[name] = failure
                logger.warning("route %s failed on %s: %s", name, self.index, failure)
                continue
            hits = response["hits"]["hits"]
            trace.routes[name] = [
                RouteHit(
                    chunk_id=str(hit["_id"]),
                    doc_id=str(hit["_source"]["doc_id"]),
                    rank=rank,
                    score=float(hit["_score"]) if hit.get("_score") is not None else None,
                )
                for rank, hit in enumerate(hits, start=1)
            ]
            for hit in hits:
                sources.setdefault(str(hit["_id"]), dict(hit["_source"]))
                if not trace.index_version:
                    trace.index_version = str(hit.get("_index", ""))

        if not trace.routes:
            reasons = dict(trace.failed_routes)
            timed_out = all(_is_timeout(r) for r in reasons.values())
            raise RetrievalError(
                f"no retrieval route answered on {self.index}: {reasons}",
                category=(
                    ErrorCategory.RETRIEVAL_TIMEOUT if timed_out else ErrorCategory.RETRIEVAL_FAILED
                ),
                route_errors=reasons,
            )
        trace.degraded = bool(trace.failed_routes)
        if not trace.index_version:
            trace.index_version = self._resolve_index(query.deadline)

        fused = rrf_fuse(trace.routes, k=self.config.rrf_k, weights=self.config.rrf_weights)
        nodes: list[NodeWithScore] = []
        for hit in fused:
            source = dict(sources[hit.chunk_id])
            text = str(source.pop("content", "") or "")
            metadata = {k: v for k, v in source.items() if v is not None}
            metadata["route_ranks"] = hit.route_ranks
            metadata["route_scores"] = hit.route_scores
            node = TextNode(id_=hit.chunk_id, text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=hit.score))
        trace.candidates = len(nodes)
        trace.timings_ms["fuse"] = (time.perf_counter() - searched) * 1000.0
        return nodes
