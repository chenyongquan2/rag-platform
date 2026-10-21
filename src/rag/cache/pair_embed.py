"""Question vectors for the pair evaluation (spec P5-01), at the least cost.

    vectors = embed_pair_texts(texts, settings, cache_path="data/cache/cache_pairs.sqlite3")

Order: the local SQLite file of this evaluation; then the L0 query cache in Redis (read only: a
`GET` of keys the service wrote; nothing is written there); then the embedding gateway, in
batches, through the same `CachedEmbedding` client the service uses (so the cost ledger gets its
lines). New vectors go to the local file only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from rag.config import ServiceSettings
from rag.embed.cache import (
    SqliteEmbeddingCache,
    cache_model_key,
    make_embedding_cache,
    text_sha256,
)
from rag.embed.factory import make_embedding

logger = logging.getLogger(__name__)


def embed_pair_texts(
    texts: Sequence[str],
    settings: ServiceSettings,
    *,
    cache_path: str | Path,
    use_redis_l0: bool = True,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """`(vectors by text, counts)`; counts: `local`, `redis`, `gateway` texts."""
    model_key = cache_model_key(
        settings.embed_model, settings.embed_dim, settings.embed_text_prefix
    )
    cache = SqliteEmbeddingCache(cache_path)
    unique = list(dict.fromkeys(texts))
    hashes = {text: text_sha256(text) for text in unique}
    found = cache.get_many(model_key, list(hashes.values()))
    counts = {"local": len(found), "redis": 0, "gateway": 0}
    missing = [t for t in unique if hashes[t] not in found]
    if missing and use_redis_l0:
        try:
            l0 = make_embedding_cache(
                settings.embed_query_cache_url or settings.redis_url,
                ttl_s=settings.embed_cache_ttl_s,
                redis_timeout_s=settings.embed_cache_redis_timeout_s,
            )
            from_redis = l0.get_many(model_key, [hashes[t] for t in missing])
        except Exception:  # the L0 cache is an optimisation here
            logger.warning("L0 cache not read", exc_info=True)
            from_redis = {}
        if from_redis:
            cache.put_many(model_key, from_redis)
            counts["redis"] = len(from_redis)
            missing = [t for t in missing if hashes[t] not in from_redis]
    if missing:
        model = make_embedding("query", settings, cache=cache)
        model.get_text_embedding_batch(missing)
        counts["gateway"] = len(missing)
    vectors = cache.get_many(model_key, list(hashes.values()))
    cache.close()
    return {text: vectors[hashes[text]] for text in unique}, counts
