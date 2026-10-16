"""Embedding client factory and L0 cache, shared by ingestion and query (spec P0-05)."""

from rag.embed.client import EmbeddingError, EmbedProfile, GatewayEmbedding

__all__ = [
    "EmbedProfile",
    "EmbeddingError",
    "GatewayEmbedding",
]
