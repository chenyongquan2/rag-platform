"""Service settings: connection strings, model ids and budget, read from env vars and `.env`."""

from __future__ import annotations

from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_MASK = "***"


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

    # Money: stop real model jobs when spent + estimate would exceed budget_stop_cny.
    budget_cny: float = Field(default=30.0, ge=0)
    budget_stop_cny: float = Field(default=28.0, ge=0)
    cost_ledger_path: str = "data/cost/ledger.jsonl"

    def redacted(self) -> dict[str, Any]:
        """Effective settings for logging at startup, with every secret replaced by `***`."""
        out: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            out[name] = _MASK if isinstance(value, SecretStr) else value
        return out
