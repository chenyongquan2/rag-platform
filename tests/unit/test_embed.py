"""Spec P0-05: embedding client, L0 cache, retries, ledger lines and the factory.

The gateway is replaced by `httpx.MockTransport`; nothing here touches the network.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import ServiceSettings

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
