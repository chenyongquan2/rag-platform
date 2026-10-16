"""Spec P0-05: embedding client, L0 cache, retries, ledger lines and the factory.

The gateway is replaced by `httpx.MockTransport`; nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from rag.config import ModelPrice, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger, UnknownModelPriceError
from rag.embed import (
    CachedEmbedding,
    EmbeddingCache,
    EmbeddingError,
    GatewayEmbedding,
    InMemoryEmbeddingCache,
    RedisEmbeddingCache,
    SqliteEmbeddingCache,
    cache_model_key,
    make_embedding,
    make_embedding_cache,
    text_sha256,
)
from rag.testing import FakeEmbedding

_EMBED_ENV = [
    "EMBED_BACKEND",
    "EMBED_TEXT_PREFIX",
    "EMBED_BATCH_SIZE",
    "EMBED_INGEST_TIMEOUT_S",
    "EMBED_INGEST_MAX_RETRIES",
    "EMBED_INGEST_BACKOFF_S",
    "EMBED_QUERY_TIMEOUT_S",
    "EMBED_QUERY_MAX_RETRIES",
    "EMBED_QUERY_BACKOFF_S",
    "EMBED_BACKOFF_MAX_S",
    "EMBED_INGEST_CACHE_URL",
    "EMBED_QUERY_CACHE_URL",
    "EMBED_CACHE_TTL_S",
    "EMBED_CACHE_REDIS_TIMEOUT_S",
    "MODEL_PRICES",
]


def test_embed_settings_defaults_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _EMBED_ENV:
        monkeypatch.delenv(key, raising=False)
    s = ServiceSettings(_env_file=None)
    assert s.embed_backend == "gateway"
    assert s.embed_text_prefix == ""
    assert s.embed_batch_size == 10
    assert s.embed_query_max_retries <= 1
    assert s.embed_query_timeout_s < s.embed_ingest_timeout_s
    assert s.embed_ingest_max_retries > s.embed_query_max_retries
    assert s.embed_ingest_cache_url.startswith("sqlite://")
    assert s.embed_query_cache_url == ""  # empty = the cache Redis in redis_url

    monkeypatch.setenv("EMBED_BACKEND", "fake")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "5")
    monkeypatch.setenv("EMBED_QUERY_CACHE_URL", "memory://")
    monkeypatch.setenv("MODEL_PRICES", '{"m": {"input_cny_per_m": 1.5}}')
    s = ServiceSettings(_env_file=None)
    assert s.embed_backend == "fake"
    assert s.embed_batch_size == 5
    assert s.embed_query_cache_url == "memory://"
    assert s.model_prices["m"].input_cny_per_m == 1.5

    with pytest.raises(ValidationError, match="embed_query_max_retries"):
        ServiceSettings(_env_file=None, embed_query_max_retries=2)
    with pytest.raises(ValidationError, match="embed_backend"):
        ServiceSettings(_env_file=None, embed_backend="openai")


# ---- fake gateway ---------------------------------------------------------------------------


class FakeGateway:
    """`httpx.MockTransport` handler that answers like an OpenAI-compatible `/embeddings`.

    `script` holds what the next requests get, one item per request: an exception to raise, an
    `httpx.Response`, or `None` for a normal answer. After the script, every answer is normal.
    Normal answers list the vectors in reverse order (with `index`) to check that the client
    sorts them, and report 2 prompt tokens per input in `usage`.
    """

    def __init__(self, dim: int = 4, script: list[object] | None = None) -> None:
        self.dim = dim
        self.script = list(script or [])
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self._vectors = FakeEmbedding(dim=dim)

    def vector(self, text: str) -> list[float]:
        return self._vectors.get_text_embedding(text)

    @property
    def inputs(self) -> list[list[str]]:
        return [body["input"] for body in self.bodies]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.script:
            action = self.script.pop(0)
            if isinstance(action, Exception):
                raise action
            if isinstance(action, httpx.Response):
                return action
        return self.answer(body["input"])

    def answer(self, texts: list[str], *, usage: bool = True) -> httpx.Response:
        data = [
            {"object": "embedding", "index": i, "embedding": self.vector(t)}
            for i, t in enumerate(texts)
        ]
        payload: dict = {"object": "list", "data": data[::-1], "model": "m"}
        if usage:
            payload["usage"] = {"prompt_tokens": 2 * len(texts), "total_tokens": 2 * len(texts)}
        return httpx.Response(200, json=payload)


def _ledger(tmp_path: Path, stop: float = 28.0) -> CostLedger:
    return CostLedger(
        tmp_path / "cost" / "ledger.jsonl",
        prices={"m": ModelPrice(input_cny_per_m=0.5), "other": ModelPrice(input_cny_per_m=0.5)},
        budget_stop_cny=stop,
    )


def _client(
    tmp_path: Path,
    gateway: FakeGateway,
    *,
    batch: int = 3,
    retries: int = 2,
    prefix: str = "",
    model: str = "m",
    ledger: CostLedger | None = None,
    sleeps: list[float] | None = None,
) -> GatewayEmbedding:
    return GatewayEmbedding(
        model_name=model,
        dim=gateway.dim,
        base_url="https://gw.example/v1/",
        api_key="sk-test-secret",
        text_prefix=prefix,
        embed_batch_size=batch,
        timeout_s=5.0,
        max_retries=retries,
        backoff_s=1.0,
        backoff_max_s=8.0,
        profile="ingest",
        ledger=ledger or _ledger(tmp_path),
        transport=httpx.MockTransport(gateway),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )


TEXTS = [f"text number {i}" for i in range(7)]


# ---- client ---------------------------------------------------------------------------------


def test_client_batches_requests_in_order(tmp_path: Path) -> None:
    gateway = FakeGateway()
    client = _client(tmp_path, gateway, batch=3, prefix="passage: ")

    vectors = client.get_text_embedding_batch(TEXTS)

    assert [len(x) for x in gateway.inputs] == [3, 3, 1]
    assert gateway.inputs[0] == [f"passage: text number {i}" for i in range(3)]
    for body in gateway.bodies:
        assert body["model"] == "m"
        assert body["dimensions"] == 4
        assert body["encoding_format"] == "float"
    request = gateway.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://gw.example/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk-test-secret"
    assert vectors == [gateway.vector("passage: " + t) for t in TEXTS]

    # More texts than the batch size in one internal call are still split by the client.
    assert len(client._get_text_embeddings(TEXTS)) == 7
    assert [len(x) for x in gateway.inputs[3:]] == [3, 3, 1]

    # The query side uses the same prefix: a query and a passage with the same text agree.
    assert client.get_query_embedding("text number 0") == vectors[0]
    assert "sk-test-secret" not in repr(client)
    assert "sk-test-secret" not in str(client.to_dict())


def test_client_records_each_call_in_ledger(tmp_path: Path) -> None:
    gateway = FakeGateway()
    ledger = _ledger(tmp_path)
    client = _client(tmp_path, gateway, batch=3, ledger=ledger)

    client.get_text_embedding_batch(TEXTS)

    rows = ledger.entries()
    assert [r["inputs"] for r in rows] == [3, 3, 1]
    assert [r["prompt_tokens"] for r in rows] == [6, 6, 2]  # from `usage`
    assert all(r["kind"] == "embedding" and r["model"] == "m" for r in rows)
    assert all(r["profile"] == "ingest" and r["usage_estimated"] is False for r in rows)
    assert ledger.total_spent() == pytest.approx(14 * 0.5 / 1e6)

    # A response without `usage` is still recorded, with tokens estimated from the characters.
    gateway.script = [gateway.answer(["abcd"], usage=False)]
    client.get_text_embedding("abcd")
    last = ledger.entries()[-1]
    assert last["usage_estimated"] is True
    assert last["prompt_tokens"] == 4


def test_client_retries_transient_errors(tmp_path: Path) -> None:
    probe = httpx.Request("POST", "https://gw.example/v1/embeddings")
    gateway = FakeGateway(
        script=[
            httpx.ReadTimeout("slow", request=probe),
            httpx.ConnectError("refused", request=probe),
            httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "rate"}),
            httpx.Response(503, json={"error": "busy"}),
        ]
    )
    sleeps: list[float] = []
    ledger = _ledger(tmp_path)
    client = _client(tmp_path, gateway, retries=4, sleeps=sleeps, ledger=ledger)

    vector = client.get_text_embedding("hello")

    assert vector == gateway.vector("hello")
    assert len(gateway.requests) == 5
    # Exponential backoff from 1 s, capped at 8 s; Retry-After replaces the computed wait.
    assert sleeps == [1.0, 2.0, 3.0, 8.0]
    assert len(ledger.entries()) == 1  # only the successful call is billed


def test_client_gives_up_after_max_retries(tmp_path: Path) -> None:
    gateway = FakeGateway(script=[httpx.Response(s, text="upstream") for s in (500, 502, 504)])
    ledger = _ledger(tmp_path)
    client = _client(tmp_path, gateway, retries=2, ledger=ledger)

    with pytest.raises(EmbeddingError, match="3 attempts") as info:
        client.get_text_embedding("hello")

    assert info.value.retryable is True
    assert info.value.status_code == 504
    assert len(gateway.requests) == 3
    assert ledger.entries() == []


@pytest.mark.parametrize(
    ("response", "status", "message"),
    [
        (httpx.Response(400, json={"error": "bad input"}), 400, "400"),
        (httpx.Response(401, json={"error": "bad key"}), 401, "401"),
        (httpx.Response(404, json={"error": "no model"}), 404, "404"),
        (httpx.Response(200, text="<html>oops</html>"), 200, "not JSON"),
        (httpx.Response(200, json={"data": []}), 200, "0 vectors for 1 inputs"),
        (
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}),
            200,
            "dimension",
        ),
    ],
)
def test_client_fails_fast_on_non_retryable_errors(
    tmp_path: Path, response: httpx.Response, status: int, message: str
) -> None:
    gateway = FakeGateway(script=[response])
    ledger = _ledger(tmp_path)
    client = _client(tmp_path, gateway, retries=3, ledger=ledger)

    with pytest.raises(EmbeddingError, match=message) as info:
        client.get_text_embedding("hello")

    assert info.value.retryable is False
    assert info.value.status_code == status
    assert len(gateway.requests) == 1
    assert "sk-test-secret" not in str(info.value)
    # A 2xx answer may have been billed even if it is unusable, so it is recorded (estimated).
    rows = ledger.entries()
    assert len(rows) == (1 if status == 200 else 0)
    assert all(r["usage_estimated"] is True for r in rows)


def test_client_refuses_call_over_budget(tmp_path: Path) -> None:
    gateway = FakeGateway()
    # "0123456789" is estimated at 10 tokens = 5e-6 CNY, more than the 1e-6 CNY stop.
    client = _client(tmp_path, gateway, ledger=_ledger(tmp_path, stop=1e-6))
    with pytest.raises(BudgetExceeded):
        client.get_text_embedding("0123456789")
    assert gateway.requests == []

    client = _client(tmp_path, gateway, ledger=_ledger(tmp_path / "other", stop=28.0))
    assert client.get_text_embedding("0123456789") == gateway.vector("0123456789")


# ---- L0 cache -------------------------------------------------------------------------------


def _f32(vector: list[float]) -> list[float]:
    """Vectors leave the cache wrapper rounded to float32, hit or miss."""
    return np.asarray(vector, dtype=np.float32).astype(float).tolist()


def _cached(
    tmp_path: Path,
    gateway: FakeGateway,
    cache: EmbeddingCache | None = None,
    *,
    model: str = "m",
    batch: int = 3,
) -> CachedEmbedding:
    client = _client(tmp_path, gateway, batch=batch, model=model)
    return CachedEmbedding(
        client,
        cache if cache is not None else InMemoryEmbeddingCache(),
        cache_model=cache_model_key(model, gateway.dim),
    )


def test_cache_hit_and_miss(tmp_path: Path) -> None:
    gateway = FakeGateway()
    cache = InMemoryEmbeddingCache()
    emb = _cached(tmp_path, gateway, cache, batch=3)

    first = emb.get_text_embedding_batch(["a", "b", "a", "c"])  # batches [a, b, a] and [c]
    assert gateway.inputs == [["a", "b"], ["c"]]  # the duplicate is sent once
    assert first == [_f32(gateway.vector(t)) for t in ["a", "b", "a", "c"]]

    assert emb.get_text_embedding_batch(["c", "b", "a"]) == [first[3], first[1], first[0]]
    assert emb.get_query_embedding("b") == first[1]  # queries share the cache
    assert len(gateway.requests) == 2  # all hits

    emb.get_text_embedding_batch(["a", "d"])
    assert gateway.inputs[-1] == ["d"]  # only the miss is sent

    # Key: (model key, sha256 of the text).
    key = cache_model_key("m", 4)
    assert cache.get_many(key, [text_sha256("a")]) == {text_sha256("a"): first[0]}
    assert text_sha256("a") == hashlib.sha256(b"a").hexdigest()

    # Another model, other dimensions or another prefix never share entries.
    other_gateway = FakeGateway()
    other = _cached(tmp_path, other_gateway, cache, model="other")
    other.get_text_embedding("a")
    assert other_gateway.inputs == [["a"]]
    keys = {cache_model_key("m", 4), cache_model_key("m", 8), cache_model_key("m", 4, "query: ")}
    assert len(keys) == 3


def test_interrupted_run_resumes_from_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "l0.sqlite3"
    gateway = FakeGateway(script=[None, None, httpx.Response(400, json={"error": "stop"})])
    cache = SqliteEmbeddingCache(path)
    with pytest.raises(EmbeddingError):
        _cached(tmp_path, gateway, cache, batch=3).get_text_embedding_batch(TEXTS)
    assert [len(x) for x in gateway.inputs] == [3, 3, 1]  # the third batch failed
    cache.close()

    # A new run (new cache object on the same file) computes only the missing text.
    rerun_gateway = FakeGateway()
    vectors = _cached(tmp_path, rerun_gateway, SqliteEmbeddingCache(path)).get_text_embedding_batch(
        TEXTS
    )
    assert rerun_gateway.inputs == [["text number 6"]]
    assert vectors == [_f32(gateway.vector(t)) for t in TEXTS]


def test_sqlite_cache_persists(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "l0.sqlite3"
    vector = [0.5, 0.25, -1.0, 0.0]
    cache = SqliteEmbeddingCache(path)
    cache.put_many("m@4", {"h1": vector})
    cache.close()

    cache = SqliteEmbeddingCache(path)
    assert cache.get_many("m@4", ["h1", "h2"]) == {"h1": vector}
    assert cache.get_many("other@4", ["h1"]) == {}
    cache.put_many("m@4", {"h1": [1.0, 2.0, 3.0, 4.0]})
    assert cache.get_many("m@4", ["h1"]) == {"h1": [1.0, 2.0, 3.0, 4.0]}

    many = {f"k{i}": [float(i)] * 4 for i in range(1_200)}  # more than one IN (...) query
    cache.put_many("m@4", many)
    assert cache.get_many("m@4", list(many)) == many
    cache.close()


class FakeRedis:
    """The two Redis calls the cache makes: MGET and a pipeline of SET with EX."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttl: dict[str, int | None] = {}
        self.down = False

    def mget(self, keys: list[str]) -> list[bytes | None]:
        if self.down:
            raise RedisConnectionError("redis is down")
        return [self.store.get(k) for k in keys]

    def pipeline(self, transaction: bool = True) -> FakeRedisPipeline:
        return FakeRedisPipeline(self)


class FakeRedisPipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.ops: list[tuple[str, bytes, int | None]] = []

    def set(self, key: str, value: bytes, ex: int | None = None) -> FakeRedisPipeline:
        self.ops.append((key, value, ex))
        return self

    def execute(self) -> list[bool]:
        if self.redis.down:
            raise RedisConnectionError("redis is down")
        for key, value, ex in self.ops:
            self.redis.store[key] = value
            self.redis.ttl[key] = ex
        return [True] * len(self.ops)


def test_redis_cache_keys_ttl_and_errors(tmp_path: Path) -> None:
    redis = FakeRedis()
    cache = RedisEmbeddingCache(redis, ttl_s=60)  # type: ignore[arg-type]
    vector = [0.5, 0.25, -1.0, 0.0]

    cache.put_many("m@4", {"h1": vector})
    assert list(redis.store) == ["rag:l0:m@4:h1"]
    assert redis.ttl["rag:l0:m@4:h1"] == 60
    assert cache.get_many("m@4", ["h1", "h2"]) == {"h1": vector}

    # Redis errors are cache misses: the query still gets its vector from the gateway.
    redis.down = True
    assert cache.get_many("m@4", ["h1"]) == {}
    cache.put_many("m@4", {"h3": vector})
    gateway = FakeGateway()
    assert _cached(tmp_path, gateway, cache).get_query_embedding("q") == _f32(gateway.vector("q"))


def test_make_embedding_cache_by_scheme(tmp_path: Path) -> None:
    sqlite_cache = make_embedding_cache(f"sqlite://{tmp_path.as_posix()}/x/l0.sqlite3")
    assert isinstance(sqlite_cache, SqliteEmbeddingCache)
    assert (tmp_path / "x" / "l0.sqlite3").exists()
    sqlite_cache.close()
    assert isinstance(make_embedding_cache("memory://"), InMemoryEmbeddingCache)

    # Connecting is lazy: building the Redis cache opens no connection.
    redis_cache = make_embedding_cache("redis://127.0.0.1:1/0", ttl_s=5, redis_timeout_s=0.1)
    assert isinstance(redis_cache, RedisEmbeddingCache)
    assert redis_cache.ttl_s == 5
    kwargs = redis_cache.client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 0.1
    assert kwargs["socket_connect_timeout"] == 0.1

    with pytest.raises(NotImplementedError, match="postgres"):
        make_embedding_cache("postgres://db/x")


# ---- factory --------------------------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: object) -> ServiceSettings:
    values: dict[str, object] = {
        "embed_base_url": "https://gw.example/v1",
        "embed_api_key": "sk-test-secret",
        "embed_backend": "gateway",
        "embed_text_prefix": "",
        "embed_batch_size": 10,
        "cost_ledger_path": str(tmp_path / "cost" / "ledger.jsonl"),
        "embed_ingest_cache_url": f"sqlite://{tmp_path.as_posix()}/cache/l0.sqlite3",
        "embed_query_cache_url": "",
        "redis_url": "redis://127.0.0.1:1/3",
    }
    values.update(overrides)
    return ServiceSettings(_env_file=None, **values)  # type: ignore[arg-type]


def test_factory_profiles_share_model_and_differ_in_timeouts(tmp_path: Path) -> None:
    settings = _settings(tmp_path, embed_text_prefix="p: ", embed_batch_size=7)
    ingest = make_embedding("ingest", settings)
    query = make_embedding("query", settings)

    assert isinstance(ingest, CachedEmbedding) and isinstance(ingest.inner, GatewayEmbedding)
    assert isinstance(query, CachedEmbedding) and isinstance(query.inner, GatewayEmbedding)
    for name in ("model_name", "dim", "base_url", "text_prefix", "embed_batch_size"):
        assert getattr(ingest.inner, name) == getattr(query.inner, name), name
    assert ingest.inner.model_name == "qwen/text-embedding-v4"
    assert ingest.inner.dim == 1024
    assert ingest.inner.text_prefix == "p: "
    assert ingest.embed_batch_size == ingest.inner.embed_batch_size == 7
    assert (
        ingest.cache_model
        == query.cache_model
        == cache_model_key("qwen/text-embedding-v4", 1024, "p: ")
    )

    assert (ingest.inner.profile, query.inner.profile) == ("ingest", "query")
    assert ingest.inner.timeout_s == settings.embed_ingest_timeout_s
    assert ingest.inner.max_retries == settings.embed_ingest_max_retries
    assert query.inner.timeout_s == settings.embed_query_timeout_s < ingest.inner.timeout_s
    assert query.inner.max_retries == settings.embed_query_max_retries <= 1

    assert isinstance(ingest.cache, SqliteEmbeddingCache)
    ingest.cache.close()
    assert isinstance(query.cache, RedisEmbeddingCache)  # empty URL -> the cache Redis
    kwargs = query.cache.client.connection_pool.connection_kwargs
    assert (kwargs["port"], kwargs["db"]) == (1, 3)
    assert query.cache.ttl_s == settings.embed_cache_ttl_s
    memory = make_embedding("query", _settings(tmp_path, embed_query_cache_url="memory://"))
    assert isinstance(memory, CachedEmbedding)
    assert isinstance(memory.cache, InMemoryEmbeddingCache)

    # The query profile makes at most two attempts.
    gateway = FakeGateway(dim=1024, script=[httpx.Response(503)] * 3)
    fast = make_embedding(
        "query",
        settings,
        cache=InMemoryEmbeddingCache(),
        transport=httpx.MockTransport(gateway),
        sleep=lambda _s: None,
    )
    with pytest.raises(EmbeddingError, match="2 attempts"):
        fast.get_query_embedding("question")
    assert len(gateway.requests) == 2

    # Missing gateway settings or a model without a price fail before any request.
    with pytest.raises(ValueError, match="EMBED_BASE_URL"):
        make_embedding(
            "ingest", _settings(tmp_path, embed_base_url=""), cache=InMemoryEmbeddingCache()
        )
    with pytest.raises(UnknownModelPriceError):
        make_embedding(
            "ingest", _settings(tmp_path, embed_model="new-model"), cache=InMemoryEmbeddingCache()
        )


def test_fake_backend_fallback(tmp_path: Path) -> None:
    gateway = FakeGateway()
    settings = _settings(
        tmp_path, embed_backend="fake", embed_dim=16, embed_batch_size=5, embed_base_url=""
    )
    for profile in ("ingest", "query"):
        emb = make_embedding(profile, settings, transport=httpx.MockTransport(gateway))
        assert isinstance(emb, FakeEmbedding)
        assert emb.dim == 16
        assert emb.embed_batch_size == 5
        vector = emb.get_query_embedding("hello")
        assert len(vector) == 16
        assert vector == emb.get_text_embedding("hello")
    assert gateway.requests == []
    assert not (tmp_path / "cost" / "ledger.jsonl").exists()
    assert not (tmp_path / "cache").exists()


def test_works_in_ingestion_pipeline(tmp_path: Path) -> None:
    gateway = FakeGateway(dim=1024)
    settings = _settings(tmp_path, embed_batch_size=2, embed_ingest_cache_url="memory://")
    emb = make_embedding("ingest", settings, transport=httpx.MockTransport(gateway))
    pipeline = IngestionPipeline(transformations=[emb])

    nodes = [TextNode(id_=f"n{i}", text=f"chunk {i}") for i in range(5)]
    out = pipeline.run(nodes=nodes)
    assert [n.embedding for n in out] == [_f32(gateway.vector(f"chunk {i}")) for i in range(5)]
    assert [len(x) for x in gateway.inputs] == [2, 2, 1]

    more = [TextNode(id_=f"m{i}", text=f"chunk {i + 3}") for i in range(4)]  # 2 cached, 2 new
    out = asyncio.run(pipeline.arun(nodes=more))
    assert [n.embedding for n in out] == [_f32(gateway.vector(f"chunk {i + 3}")) for i in range(4)]
    assert sorted(t for batch in gateway.inputs[3:] for t in batch) == ["chunk 5", "chunk 6"]
