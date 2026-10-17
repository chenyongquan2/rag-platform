"""Service settings: connection strings, model ids and budget, read from env vars and `.env`."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.config.paths import shared_data_dir

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


class ModelPrice(BaseModel):
    """Price of one model in CNY per million tokens (spec P0-05)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_cny_per_m: float = Field(ge=0)
    output_cny_per_m: float = Field(default=0.0, ge=0)
    # True when the price is a guess, not checked against the gateway's bill (spec P1-01).
    estimated: bool = False


def _default_model_prices() -> dict[str, ModelPrice]:
    return {
        # Design s20-2: text-embedding-v4 costs 0.5 CNY per million tokens.
        "qwen/text-embedding-v4": ModelPrice(input_cny_per_m=0.5),
        # The gateway lists no prices and shows no bill per call. Deliberately high estimates:
        # the budget check then stops early rather than late. See docs/decisions.md.
        "deepseek-v4-flash": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0, estimated=True),
    }


class ChatProfile(BaseModel):
    """One use of the chat model (design s11-1, s16-2): which model, how long to wait, how often
    to retry and how many output tokens to allow (spec P1-01).

    - `read_timeout_s`: the longest wait for the next bytes of the answer. Without streaming the
      gateway sends nothing until the answer is complete, so this caps the whole generation;
      with streaming it caps the gap between chunks (reasoning is streamed too).
    - `deadline_s`: all attempts and waits of one call until its answer starts must fit into this
      time; each attempt's timeouts are cut to what is left. None: no limit.
    - Only failures before the gateway got the request (connect errors and timeouts) and
      408/429/5xx answers are retried, with waits `backoff_s` * 2**n up to `backoff_max_s`; a
      longer `Retry-After` ends the call. A read timeout is not retried: the gateway may still
      be generating, and billing, the first answer.
    - `max_tokens` caps reasoning and answer tokens together (the default model is a reasoning
      model), so it must leave room for the reasoning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = ""  # empty: `chat_model`
    connect_timeout_s: float = Field(gt=0)
    read_timeout_s: float = Field(gt=0)
    deadline_s: float | None = Field(default=None, gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    temperature: float | None = Field(default=0.0, ge=0, le=2)


ChatProfileName = Literal["online", "offline", "cheap"]

# online: answering a user; cheap: rewrite, route, sufficiency checks. Both must never keep the
# user waiting long (design s11-1, s16-8). offline: question generation and judging, may wait.
CHAT_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "online": {
        "connect_timeout_s": 3.0,
        "read_timeout_s": 10.0,
        "deadline_s": 12.0,
        "max_retries": 1,
        "backoff_s": 0.5,
        "backoff_max_s": 1.0,
        "max_tokens": 4096,
    },
    "cheap": {
        "connect_timeout_s": 3.0,
        "read_timeout_s": 8.0,
        "deadline_s": 10.0,
        "max_retries": 1,
        "backoff_s": 0.5,
        "backoff_max_s": 1.0,
        "max_tokens": 1024,
    },
    "offline": {
        "connect_timeout_s": 10.0,
        "read_timeout_s": 180.0,
        "deadline_s": None,
        "max_retries": 4,
        "backoff_s": 2.0,
        "backoff_max_s": 60.0,
        "max_tokens": 8192,
    },
}


def _chat_profile_default(name: str) -> ChatProfile:
    return ChatProfile(**CHAT_PROFILE_DEFAULTS[name])


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

    # Connection strings (design s16-7). ES, Redis and OTLP defaults are the host ports of
    # docker-compose.yml (standard port + 20000). 127.0.0.1, not localhost: on Windows
    # "localhost" tries ::1 first and each connection then waits about 2 s.
    es_url: str = "http://127.0.0.1:29200"
    redis_url: str = "redis://127.0.0.1:26379/0"
    docstore_url: str = "file://data/docstore"
    checkpoint_url: str = "sqlite://data/checkpoints/langgraph.db"
    lock_url: str = "file://data/locks"
    otlp_endpoint: str = "http://127.0.0.1:24317"  # OTel Collector, OTLP gRPC
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

    # Embedding client (spec P0-05, design s7-4, s16-2, s17-1). Ingest and query share model,
    # dims, prefix and batch size; only timeout, retries and backoff differ by profile.
    # `fake` switches both profiles to FakeEmbedding (gateway down or budget used up).
    embed_backend: Literal["gateway", "fake"] = "gateway"
    embed_text_prefix: str = ""
    embed_batch_size: int = Field(default=10, gt=0, le=2048)
    embed_ingest_timeout_s: float = Field(default=60.0, gt=0)
    embed_ingest_max_retries: int = Field(default=4, ge=0)
    embed_ingest_backoff_s: float = Field(default=2.0, ge=0)
    embed_ingest_backoff_max_s: float = Field(default=60.0, ge=0)
    embed_ingest_deadline_s: float | None = Field(default=None, gt=0)  # None: no total limit
    embed_query_timeout_s: float = Field(default=3.0, gt=0)
    embed_query_max_retries: int = Field(default=1, ge=0, le=1)
    embed_query_backoff_s: float = Field(default=0.2, ge=0)
    embed_query_backoff_max_s: float = Field(default=1.0, ge=0)
    embed_query_deadline_s: float | None = Field(default=7.0, gt=0)
    # A wait (backoff or the gateway's Retry-After) longer than the profile's backoff_max ends
    # the request instead; with a deadline, each attempt gets at most the time that is left.
    # Requests in flight per client: IngestionPipeline.arun starts every batch at once.
    embed_max_concurrency: int = Field(default=4, ge=1)
    # L0 embedding cache (design s12-1): `sqlite://<path>` or `memory://` for ingest;
    # `redis://...` or `memory://` for query, empty = the shared cache Redis `redis_url`.
    embed_ingest_cache_url: str = "sqlite://data/cache/embeddings.sqlite3"
    embed_query_cache_url: str = ""
    embed_cache_ttl_s: int = Field(default=30 * 24 * 3600, gt=0)
    embed_cache_redis_timeout_s: float = Field(default=0.5, gt=0)

    # Chat models (spec P1-01, design s11-1). One profile per use; each is a JSON object in the
    # env (`CHAT_ONLINE='{"read_timeout_s": 8}'`), keys left out keep that profile's defaults.
    # `fake` switches every profile to FakeChatModel answering `chat_fake_reply` (gateway down or
    # budget used up); it is never switched automatically.
    chat_backend: Literal["gateway", "fake"] = "gateway"
    chat_fake_reply: str = "[fake chat model] no answer"
    chat_structured_output_method: Literal["function_calling", "json_schema", "json_mode"] = (
        "function_calling"
    )
    chat_online: ChatProfile = Field(default_factory=lambda: _chat_profile_default("online"))
    chat_offline: ChatProfile = Field(default_factory=lambda: _chat_profile_default("offline"))
    chat_cheap: ChatProfile = Field(default_factory=lambda: _chat_profile_default("cheap"))

    # Hugging Face data (spec P0-03). huggingface.co is blocked here; the mirror serves the same
    # `/datasets/<repo>/resolve/<revision>/<path>` URLs. `HF_ENDPOINT` is also what huggingface_hub
    # reads. Point `HF_DATA_DIR` at one shared directory so several checkouts reuse one download.
    hf_endpoint: str = "https://hf-mirror.com"
    hf_data_dir: str = "data/raw/hf"
    hf_timeout_s: float = Field(default=60.0, gt=0)

    # Money: stop real model jobs when spent + estimate would exceed budget_stop_cny.
    budget_cny: float = Field(default=30.0, ge=0)
    budget_stop_cny: float = Field(default=28.0, ge=0)
    # One ledger for every checkout and worktree on this machine, so the stop sees all spend.
    cost_ledger_path: str = Field(
        default_factory=lambda: (shared_data_dir() / "cost" / "ledger.jsonl").as_posix()
    )
    model_prices: dict[str, ModelPrice] = Field(default_factory=_default_model_prices)

    @field_validator("chat_online", "chat_offline", "chat_cheap", mode="before")
    @classmethod
    def _merge_chat_profile_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        """A partial profile (from env JSON or code) keeps that profile's other defaults."""
        if isinstance(value, Mapping):
            name = (info.field_name or "").removeprefix("chat_")
            return {**CHAT_PROFILE_DEFAULTS[name], **value}
        return value

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
