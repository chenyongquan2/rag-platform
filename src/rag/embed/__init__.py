"""Embedding client factory and L0 cache, shared by ingestion and query (spec P0-05)."""

from rag.embed.cache import (
    CachedEmbedding,
    EmbeddingCache,
    InMemoryEmbeddingCache,
    RedisEmbeddingCache,
    SqliteEmbeddingCache,
    cache_model_key,
    make_embedding_cache,
    text_sha256,
)
from rag.embed.client import EmbeddingError, EmbedProfile, GatewayEmbedding

__all__ = [
    "CachedEmbedding",
    "EmbedProfile",
    "EmbeddingCache",
    "EmbeddingError",
    "GatewayEmbedding",
    "InMemoryEmbeddingCache",
    "RedisEmbeddingCache",
    "SqliteEmbeddingCache",
    "cache_model_key",
    "make_embedding_cache",
    "text_sha256",
]
