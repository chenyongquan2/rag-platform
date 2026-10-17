"""Hybrid search as our own `RetrievalResult` (design s9-1, s16-1; P0-10 `Retriever` protocol).

`HybridSearch` runs the LlamaIndex chain for one question - `EsHybridRetriever`, then the
postprocessors (`TextDedupe` until P0-08 adds rerank, threshold and context expansion) - and
converts the nodes into `Chunk`s at the layer boundary: callers (eval harness, search_docs,
the LangGraph retrieval node) never see `NodeWithScore`. Each route's own list goes into
`RetrievalResult.routes`, failed routes into `degraded`, step times into `timings_ms`.

`make_hybrid_search` builds it from the KB config and service settings (design s16-7).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from elasticsearch import Elasticsearch
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore
from redis import Redis
from redis.exceptions import RedisError

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.factories import make_es_client
from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation
from rag.index.manage import aliased_indices, index_embed_model
from rag.index.names import KbIndexNames
from rag.ingest.meta import title_parts
from rag.retrieve.deadline import Deadline
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.hybrid import EsHybridRetriever
from rag.retrieve.query import HybridQuery
from rag.types import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class IndexEmbedModelError(RuntimeError):
    """The index was written with another embedding model than the query would use: the
    vectors are not comparable (s7-5), so the vector route would return noise."""


def node_to_chunk(item: NodeWithScore, kb: str) -> Chunk:
    """Our `Chunk` from a fused node; leftover metadata (doc_type, language, chunk_index,
    parent_id, images, route_ranks, route_scores) stays in `Chunk.metadata`."""
    node = item.node
    meta: dict[str, Any] = dict(node.metadata)
    meta.pop("chunk_id", None)
    title = meta.pop("title_path", "") or ""
    return Chunk(
        chunk_id=node.node_id,
        doc_id=str(meta.pop("doc_id")),
        kb=str(meta.pop("kb", kb)),
        text=node.get_content(metadata_mode=MetadataMode.NONE),
        suspicious_instruction=bool(meta.pop("suspicious_instruction", False)),
        title_path=title_parts(str(title)),
        source_rel=str(meta.pop("source_rel", "") or ""),
        source_url=meta.pop("source_url", None),
        source_version=meta.pop("source_version", None),
        score=item.score,
        metadata=meta,
    )


class HybridSearch:
    """The retrieval chain for one knowledge base and one caller identity (its ACL groups).

    Implements `rag.eval.retrieval.Retriever`: `retrieve(question, kb=..., top_k=...)`. The
    final list is the fused order after the postprocessors, cut to `top_k` (dedupe runs before
    the cut, so duplicates do not take places in it).
    """

    def __init__(
        self,
        retriever: EsHybridRetriever,
        *,
        acl_groups: Sequence[str],
        deadline_ms: int,
        postprocessors: Sequence[BaseNodePostprocessor] | None = None,
        generation: Callable[[], int] | None = None,
    ) -> None:
        if deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {deadline_ms}")
        self.retriever = retriever
        self.acl_groups = tuple(acl_groups)
        self.deadline_ms = deadline_ms
        self.postprocessors = list(postprocessors) if postprocessors is not None else [TextDedupe()]
        self._generation = generation

    @property
    def kb(self) -> str:
        return self.retriever.kb

    def retrieve(
        self,
        question: str,
        kb: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
    ) -> RetrievalResult:
        """Search `kb` (must be this chain's knowledge base); raises `RetrievalError` when no
        route answered and `FilterError` for a filter outside the whitelist."""
        if kb != self.kb:
            raise ValueError(f"this search is bound to knowledge base {self.kb!r}, not {kb!r}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        start = time.perf_counter()
        query = HybridQuery(
            question,
            acl_groups=self.acl_groups,
            filters=dict(filters or {}),
            deadline=Deadline.after_ms(
                deadline_ms if deadline_ms is not None else self.deadline_ms
            ),
        )
        nodes = self.retriever.retrieve(query)
        for post in self.postprocessors:
            began = time.perf_counter()
            nodes = post.postprocess_nodes(nodes, query_bundle=query)
            name = getattr(post, "timing_name", None) or post.class_name()
            query.trace.timings_ms[name] = (time.perf_counter() - began) * 1000.0
        chunks = [node_to_chunk(n, self.kb) for n in nodes[:top_k]]
        trace = query.trace
        trace.timings_ms["total"] = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(
            request_id=request_id or uuid.uuid4().hex,
            kb=self.kb,
            index_version=trace.index_version,
            data_generation=self._read_generation(),
            degraded=trace.degraded,
            timings_ms=dict(trace.timings_ms),
            chunks=chunks,
            routes={name: list(hits) for name, hits in trace.routes.items()},
        )

    def _read_generation(self) -> int:
        if self._generation is None:
            return 0
        try:
            return int(self._generation())
        except (RedisError, OSError) as exc:  # the generation is informational here
            logger.warning("data generation not readable: %s", exc)
            return 0

    def close(self) -> None:
        self.retriever.close()


def _embed_id(embed: BaseEmbedding) -> str:
    from rag.ingest.job import embed_model_id  # the id the ingest job writes into _meta

    return embed_model_id(embed)


def make_hybrid_search(
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    *,
    acl_groups: Sequence[str],
    index_prefix: str = "",
    es: Elasticsearch | None = None,
    embed_model: BaseEmbedding | None = None,
    redis: Redis | None = None,
) -> HybridSearch:
    """The chain over alias `<index_prefix>kb_<kb>` with the KB's retrieval config.

    Defaults: an ES client without retries (the deadline decides; s16-2), the query-profile
    embedding (P0-05). Every index behind the alias must have been written with the same
    embedding model (`_meta.embed_model`), else `IndexEmbedModelError`. With `redis`, results
    carry the data generation (key prefix as the ingest job's for this index prefix).
    """
    from rag.embed import make_embedding

    names = KbIndexNames(cfg.kb, prefix=index_prefix)
    if es is None:
        es = make_es_client(
            settings.es_url, request_timeout=settings.search_deadline_ms / 1000.0, max_retries=0
        )
    if embed_model is None:
        embed_model = make_embedding("query", settings)
    indices = aliased_indices(es, names)
    if not indices:
        raise IndexEmbedModelError(f"alias {names.alias} points to no index; run ingestion first")
    expected = _embed_id(embed_model)
    for index in indices:
        recorded = index_embed_model(es, index)
        if recorded != expected:
            raise IndexEmbedModelError(
                f"index {index} was written with {recorded!r}; the query would embed with "
                f"{expected!r}"
            )
    generation: Callable[[], int] | None = None
    if redis is not None:
        key_prefix = f"{DEFAULT_KEY_PREFIX}{index_prefix}"
        client = redis

        def generation() -> int:
            return get_data_generation(client, cfg.kb, key_prefix=key_prefix)

    retriever = EsHybridRetriever(
        es,
        embed_model,
        kb=cfg.kb,
        index=names.alias,
        config=cfg.retrieval,
        es_reserve_ms=settings.search_es_reserve_ms,
        es_margin_ms=settings.search_es_margin_ms,
        embed_threads=settings.embed_max_concurrency,
    )
    return HybridSearch(
        retriever,
        acl_groups=acl_groups,
        deadline_ms=settings.search_deadline_ms,
        generation=generation,
    )
