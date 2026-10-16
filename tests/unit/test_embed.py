"""Spec P0-05: embedding client, L0 cache, retries, ledger lines and the factory.

The gateway is replaced by `httpx.MockTransport`; nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from rag.config import ModelPrice, ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.embed import EmbeddingError, GatewayEmbedding
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
