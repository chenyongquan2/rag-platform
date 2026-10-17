"""L0 embedding cache (design s12-1): key = (model key, sha256 of the text), value = the vector.

Ingestion uses a local SQLite file, queries use Redis (design s16-7); `memory://` is an
in-process dict for tests. `CachedEmbedding` puts the cache in front of an embedding model and
writes after every request, so an interrupted ingest recomputes only the texts still missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import PrivateAttr

from rag.factories import split_url, url_path

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

_SQL_CHUNK = 500  # keys per `IN (...)` query, below every SQLite variable limit
_VECTOR_DTYPE = "<f4"  # little-endian float32, the precision Elasticsearch stores anyway


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_model_key(model: str, dim: int, text_prefix: str = "") -> str:
    """The model part of the cache key.

    Includes the dimensions and the text prefix, because either changes the vector of the same
    text under the same model name. Changing the model therefore never returns old vectors.
    """
    key = f"{model}@{dim}"
    if text_prefix:
        key += f"+prefix:{text_sha256(text_prefix)[:12]}"
    return key


def _encode(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=_VECTOR_DTYPE).tobytes()


def _decode(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=_VECTOR_DTYPE).astype(float).tolist()


def _as_float32(vector: Sequence[float]) -> list[float]:
    return _decode(_encode(vector))


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingCache(Protocol):
    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Vectors found for these text hashes; missing hashes are left out."""
        ...

    def put_many(self, model: str, vectors: Mapping[str, Sequence[float]]) -> None:
        """Store vectors by text hash, overwriting existing entries."""
        ...


class InMemoryEmbeddingCache:
    """Process-local dict; for unit tests and `memory://`."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        with self._lock:
            found = {h: self._data.get((model, h)) for h in hashes}
        return {h: _decode(blob) for h, blob in found.items() if blob is not None}

    def put_many(self, model: str, vectors: Mapping[str, Sequence[float]]) -> None:
        encoded = {(model, h): _encode(v) for h, v in vectors.items()}
        with self._lock:
            self._data.update(encoded)


class SqliteEmbeddingCache:
    """One SQLite file (WAL mode); one transaction per `put_many`, i.e. per request."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path), timeout=30, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            " model TEXT NOT NULL,"
            " text_sha256 TEXT NOT NULL,"
            " vector BLOB NOT NULL,"
            " PRIMARY KEY (model, text_sha256)"
            ") WITHOUT ROWID"
        )

    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            for chunk in _chunks(list(hashes), _SQL_CHUNK):
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT text_sha256, vector FROM embeddings"
                    f" WHERE model = ? AND text_sha256 IN ({marks})",
                    [model, *chunk],
                ).fetchall()
                found.update((h, _decode(blob)) for h, blob in rows)
        return found

    def put_many(self, model: str, vectors: Mapping[str, Sequence[float]]) -> None:
        rows = [(model, h, _encode(v)) for h, v in vectors.items()]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (model, text_sha256, vector)"
                    " VALUES (?, ?, ?)",
                    rows,
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RedisEmbeddingCache:
    """One Redis string per vector, `<prefix>:<model>:<sha256>`, with a TTL.

    The cache Redis may evict keys (design s7-4), and a query must not fail because the cache
    is down: Redis errors are logged and treated as misses.
    """

    def __init__(self, client: Redis, *, ttl_s: int | None, prefix: str = "rag:l0") -> None:
        self.client = client
        self.ttl_s = ttl_s
        self.prefix = prefix

    def _key(self, model: str, text_hash: str) -> str:
        return f"{self.prefix}:{model}:{text_hash}"

    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        from redis.exceptions import RedisError

        hashes = list(hashes)
        if not hashes:
            return {}
        try:
            blobs = self.client.mget([self._key(model, h) for h in hashes])
        except RedisError as exc:
            logger.warning("L0 cache: Redis read failed, treating as miss: %s", exc)
            return {}
        return {h: _decode(blob) for h, blob in zip(hashes, blobs, strict=True) if blob}

    def put_many(self, model: str, vectors: Mapping[str, Sequence[float]]) -> None:
        from redis.exceptions import RedisError

        if not vectors:
            return
        try:
            pipe = self.client.pipeline(transaction=False)
            for h, v in vectors.items():
                pipe.set(self._key(model, h), _encode(v), ex=self.ttl_s)
            pipe.execute()
        except RedisError as exc:
            logger.warning("L0 cache: Redis write failed, vectors not cached: %s", exc)


def make_embedding_cache(
    url: str, *, ttl_s: int | None = None, redis_timeout_s: float = 0.5
) -> EmbeddingCache:
    """`sqlite://<path>` -> SQLite file, `redis://` / `rediss://` -> Redis, `memory://` -> dict."""
    scheme, _ = split_url(url)
    if scheme == "sqlite":
        return SqliteEmbeddingCache(url_path(url))
    if scheme in {"redis", "rediss"}:
        from redis import Redis

        client = Redis.from_url(
            url, socket_timeout=redis_timeout_s, socket_connect_timeout=redis_timeout_s
        )
        return RedisEmbeddingCache(client, ttl_s=ttl_s)
    if scheme == "memory":
        return InMemoryEmbeddingCache()
    raise NotImplementedError(
        f"embedding cache: scheme {scheme!r} is not implemented yet "
        "(supported: sqlite, redis, rediss, memory)"
    )


class CachedEmbedding(BaseEmbedding):
    """L0 cache in front of an embedding model; itself a LlamaIndex `BaseEmbedding`.

    - Texts are looked up by (`cache_model`, sha256(text)); only misses go to `inner`, each
      distinct text once, at most `inner.embed_batch_size` per call, and every call's vectors
      are written to the cache before the next call starts.
    - Queries and passages share entries: `inner` must embed a query exactly like a passage of
      the same text (design s17-1: same model, dims and prefix on both sides).
    - Vectors are returned as float32 values whether they came from the cache or not, so the
      same text always gives the same numbers.
    """

    inner: BaseEmbedding
    cache_model: str

    _cache: EmbeddingCache = PrivateAttr()

    def __init__(
        self, inner: BaseEmbedding, cache: EmbeddingCache, *, cache_model: str, **data: Any
    ) -> None:
        data.setdefault("model_name", inner.model_name)
        data.setdefault("embed_batch_size", inner.embed_batch_size)
        super().__init__(inner=inner, cache_model=cache_model, **data)
        self._cache = cache

    @classmethod
    def class_name(cls) -> str:
        return "CachedEmbedding"

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    def to_payload(self) -> dict[str, Any]:
        return {**self.inner.to_payload(), "cache_model": self.cache_model}

    def _embed(self, texts: Sequence[str]) -> list[Embedding]:
        hashes = [text_sha256(t) for t in texts]
        found = self._cache.get_many(self.cache_model, list(dict.fromkeys(hashes)))
        missing: dict[str, str] = {}
        for text_hash, text in zip(hashes, texts, strict=True):
            if text_hash not in found:
                missing.setdefault(text_hash, text)
        pending = list(missing.items())
        for chunk in _chunks(pending, self.inner.embed_batch_size):
            vectors = self.inner._get_text_embeddings([text for _, text in chunk])
            fresh = {h: _as_float32(v) for (h, _), v in zip(chunk, vectors, strict=True)}
            self._cache.put_many(self.cache_model, fresh)
            found.update(fresh)
        return [found[h] for h in hashes]

    def cached_query_embedding(self, query: str) -> Embedding | None:
        """The cached vector of `query`, or None on a miss; never calls `inner`. A cache error
        counts as a miss (the Redis cache already treats its errors that way)."""
        text_hash = text_sha256(query)
        return self._cache.get_many(self.cache_model, [text_hash]).get(text_hash)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)
