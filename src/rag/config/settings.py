"""Service settings: connection strings, model ids and budget, read from env vars and `.env`."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MASK = "***"
_SECRET_QUERY_KEY = re.compile(r"pass|pwd|secret|token|key|auth|sig|credential", re.IGNORECASE)


def mask_url(value: str) -> str:
    """Replace credentials in a connection string with `***`.

    Masks the password in `user:password@host`, the whole user part when there is no password
    (it may be a token), and query values whose key looks secret (`api_key=`, `password=`, ...).
    Values that are not URLs are returned unchanged.
    """
    if "://" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return _MASK
    changed = False
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        user, has_password, _ = userinfo.partition(":")
        netloc = f"{user}:{_MASK}@{host}" if has_password else f"{_MASK}@{host}"
        changed = True
    query = parts.query
    if query:
        pairs = []
        for pair in query.split("&"):
            key, sep, _ = pair.partition("=")
            if sep and _SECRET_QUERY_KEY.search(key):
                pair = f"{key}={_MASK}"
                changed = True
            pairs.append(pair)
        query = "&".join(pairs)
    if not changed:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


class ServiceSettings(BaseSettings):
    """Process-wide settings. Env var names are the upper-case field names (no prefix).

    Every external dependency is a connection string; `rag.factories` picks the implementation
    by its scheme (design s16-7). Secrets are `SecretStr` so they never show in repr or logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Connection strings (design s16-7).
    es_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"
    docstore_url: str = "file://data/docstore"
    checkpoint_url: str = "sqlite://data/checkpoints/langgraph.db"
    lock_url: str = "file://data/locks"
    otlp_endpoint: str = "http://localhost:4317"
    kb_dir: str = "kb"

    # Model gateway (OpenAI-compatible). Empty means "not configured"; callers must check.
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    embed_base_url: str = ""
    embed_api_key: SecretStr = SecretStr("")

    # Model ids.
    chat_model: str = "deepseek-v4-flash"
    embed_model: str = "qwen/text-embedding-v4"
    embed_dim: int = Field(default=1024, gt=0)
    reranker_model: str = "BAAI/bge-reranker-base"

    # Hugging Face data (spec P0-03). huggingface.co is blocked here; the mirror serves the same
    # `/datasets/<repo>/resolve/<revision>/<path>` URLs. `HF_ENDPOINT` is also what huggingface_hub
    # reads. Point `HF_DATA_DIR` at one shared directory so several checkouts reuse one download.
    hf_endpoint: str = "https://hf-mirror.com"
    hf_data_dir: str = "data/raw/hf"
    hf_timeout_s: float = Field(default=60.0, gt=0)

    # Money: stop real model jobs when spent + estimate would exceed budget_stop_cny.
    budget_cny: float = Field(default=30.0, ge=0)
    budget_stop_cny: float = Field(default=28.0, ge=0)
    cost_ledger_path: str = "data/cost/ledger.jsonl"

    @model_validator(mode="after")
    def _stop_within_budget(self) -> Self:
        if self.budget_stop_cny > self.budget_cny:
            raise ValueError(
                f"budget_stop_cny ({self.budget_stop_cny}) must not exceed "
                f"budget_cny ({self.budget_cny})"
            )
        return self

    def redacted(self) -> dict[str, Any]:
        """Effective settings for logging at startup.

        `SecretStr` fields become `***`; credentials inside connection strings are masked too.
        """
        out: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                out[name] = _MASK
            elif isinstance(value, str):
                out[name] = mask_url(value)
            else:
                out[name] = value
        return out

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        # repr() and str() show the redacted values, never credentials from URLs.
        yield from self.redacted().items()
