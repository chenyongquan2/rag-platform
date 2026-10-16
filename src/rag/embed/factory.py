"""The one place that builds embedding models (spec P0-05, design s16-7, s17-1).

Ingestion and query both call `make_embedding`; model, dimensions, text prefix and batch size
come from the same settings, so passage vectors and question vectors always match. The two
profiles differ only in timeout, retries, waits and deadline (design s16-2) and in the L0 cache
backend: SQLite for ingestion, the cache Redis for queries (design s12-1).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag.config import ServiceSettings
from rag.cost import CostLedger
from rag.embed.cache import (
    CachedEmbedding,
    EmbeddingCache,
    cache_model_key,
    make_embedding_cache,
)
from rag.embed.client import EmbedProfile, GatewayEmbedding
from rag.testing.fakes import FakeEmbedding


def make_embedding(
    profile: EmbedProfile,
    settings: ServiceSettings | None = None,
    *,
    cache: EmbeddingCache | None = None,
    ledger: CostLedger | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> BaseEmbedding:
    """Embedding model for `profile`.

    `embed_backend=gateway`: `CachedEmbedding` around `GatewayEmbedding`; the cache comes from
    `embed_ingest_cache_url` or `embed_query_cache_url` (empty = `redis_url`) unless given.
    `embed_backend=fake`: `FakeEmbedding` of `embed_dim` for when the gateway is down or the
    budget is used up; it makes no request, writes no ledger line and is not cached. Its
    vectors do not match real ones: never query an index built with the other backend.
    `ledger`, `transport` and `sleep` are for tests.
    """
    settings = settings or ServiceSettings()
    if profile not in ("ingest", "query"):
        raise ValueError(f"embedding profile must be 'ingest' or 'query', got {profile!r}")
    if settings.embed_backend == "fake":
        return FakeEmbedding(dim=settings.embed_dim, embed_batch_size=settings.embed_batch_size)

    if profile == "ingest":
        timeout_s = settings.embed_ingest_timeout_s
        max_retries = settings.embed_ingest_max_retries
        backoff_s = settings.embed_ingest_backoff_s
        backoff_max_s = settings.embed_ingest_backoff_max_s
        deadline_s = settings.embed_ingest_deadline_s
    else:
        timeout_s = settings.embed_query_timeout_s
        max_retries = settings.embed_query_max_retries
        backoff_s = settings.embed_query_backoff_s
        backoff_max_s = settings.embed_query_backoff_max_s
        deadline_s = settings.embed_query_deadline_s

    client = GatewayEmbedding(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        base_url=settings.embed_base_url,
        text_prefix=settings.embed_text_prefix,
        embed_batch_size=settings.embed_batch_size,
        profile=profile,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        backoff_max_s=backoff_max_s,
        deadline_s=deadline_s,
        max_concurrency=settings.embed_max_concurrency,
        api_key=settings.embed_api_key.get_secret_value(),
        ledger=ledger or CostLedger.from_settings(settings),
        transport=transport,
        sleep=sleep,
    )
    if cache is None:
        if profile == "ingest":
            url = settings.embed_ingest_cache_url
        else:
            url = settings.embed_query_cache_url or settings.redis_url
        cache = make_embedding_cache(
            url,
            ttl_s=settings.embed_cache_ttl_s,
            redis_timeout_s=settings.embed_cache_redis_timeout_s,
        )
    return CachedEmbedding(
        client,
        cache,
        cache_model=cache_model_key(
            settings.embed_model, settings.embed_dim, settings.embed_text_prefix
        ),
    )
