"""Service settings: connection strings, model ids and budget, read from env vars and `.env`."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

import psutil
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


def physical_cores() -> int:
    """Physical CPU cores (psutil), else the logical CPUs `os.cpu_count()` reports; at least 1.
    Torch's CPU kernels gain nothing from hyper-threads, so the reranker defaults count cores."""
    count = psutil.cpu_count(logical=False)
    return max(1, count or os.cpu_count() or 1)


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
        # Answer judge (spec P1-04): another family than the generator. Same high guess.
        "qwen/qwen3.6-flash": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0, estimated=True),
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
    otlp_endpoint: str = "http://127.0.0.1:24317"  # OTel Collector, OTLP gRPC (see otlp_protocol)
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

    # Reranker (spec P0-08, design s10-1). One local cross-encoder per process, loaded from the
    # local HF cache only (`local_files_only`; also set HF_HUB_OFFLINE=1). `reranker_max_length`
    # is the model's 512 positions for question + title path + chunk. Candidates go in batches
    # of `rerank_batch_size`; a batch that would not finish before the deadline is not started.
    # Each rerank runs on `reranker_torch_threads` torch threads (set process-wide at load;
    # default all physical cores, spec P0-13) and at most `rerank_max_concurrency` requests
    # rerank at the same time (default physical cores // torch threads, so together they use
    # each physical core once: 1 by default); the others wait, and the wait counts against
    # their deadline. P0-13 load test on 8 cores, rerank_n 10: 8 x 1 had search P95 3.4 s at one
    # request and 6.5-7.1 s at two, like 8 x 2 and better than 4 x 2 (5.8-6.2 s at one).
    reranker_device: str = "cpu"
    reranker_max_length: int = Field(default=512, gt=0, le=512)
    rerank_batch_size: int = Field(default=8, gt=0)
    reranker_torch_threads: int = Field(default_factory=lambda: max(1, physical_cores()), ge=1)
    rerank_max_concurrency: int = Field(
        default_factory=lambda data: max(
            1, physical_cores() // max(1, int(data.get("reranker_torch_threads") or 1))
        ),
        ge=1,
    )

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

    # Observability (spec P0-11, design s15-1). `otlp`: traces and metrics go over OTLP to the
    # Collector at `otlp_endpoint` (gRPC: the endpoint as is; http/protobuf: `/v1/traces` and
    # `/v1/metrics` are appended). `memory`: kept in memory for tests. `off`: nothing recorded.
    otel_mode: Literal["otlp", "memory", "off"] = "otlp"
    otlp_protocol: Literal["grpc", "http/protobuf"] = "grpc"
    otlp_timeout_s: float = Field(default=5.0, gt=0)
    otel_metric_export_interval_s: float = Field(default=30.0, gt=0)
    # OpenInference spans for LlamaIndex and LangChain/LangGraph calls. With trace content on,
    # prompts, answers and retrieved texts go into the spans (local only in P0); embedding
    # vectors never do.
    otel_auto_instrument: bool = True
    otel_trace_content: bool = True
    # Commit put on every span's resource; empty = ask git (containers set GIT_COMMIT).
    git_commit: str = ""
    # Business records: one JSON line per request for eval and failure analysis. Relative
    # paths are taken from the repo root; empty disables. Rotated by size, `backups` old files.
    business_record_path: str = "data/observe/records/business.jsonl"
    business_record_max_bytes: int = Field(default=20_000_000, gt=0)
    business_record_backups: int = Field(default=5, ge=0)
    # Reachability gauge of Elasticsearch and Redis (spec P5-02): each check waits at most
    # `dependency_check_timeout_s`; a verdict is reused for `dependency_check_ttl_s` seconds, so
    # a metric export every 30 s never pings more often than that.
    dependency_check_timeout_s: float = Field(default=1.5, gt=0)
    dependency_check_ttl_s: float = Field(default=10.0, gt=0)
    # Scheduled probe (spec P5-02, `python -m rag.monitor probe`): one fixed `search_docs` request
    # every `probe_interval_s` seconds to the running service (`probe_url`, empty: the service's
    # own host and port). The answer is sane when a returned passage's source (path, URL or title)
    # contains `probe_expected`, and the result is not degraded. The ask check costs model calls:
    # off by default, at most one per `probe_ask_interval_s` and `probe_ask_daily_cap` per UTC
    # day (counted in `probe_state_path`, so a restart does not reset it).
    probe_url: str = ""
    probe_kb: str = "hf_docs"
    probe_query: str = "How do I upload a model to the Hugging Face Hub with push_to_hub?"
    probe_expected: str = "chapter4/3.mdx"
    probe_top_k: int = Field(default=5, ge=1, le=20)
    # A made-up word is added to the query of every run, so the result cache (an hour) cannot
    # answer the probe and hide a failure of Elasticsearch or the gateway; the ask check uses a
    # session id for the same reason.
    probe_unique_query: bool = True
    probe_interval_s: float = Field(default=60.0, gt=0)
    probe_timeout_s: float = Field(default=20.0, gt=0)
    probe_metric_export_interval_s: float = Field(default=10.0, gt=0)
    probe_ask_enabled: bool = False
    probe_ask_question: str = "How do I upload a model to the Hugging Face Hub with push_to_hub?"
    probe_ask_interval_s: float = Field(default=1800.0, gt=0)
    probe_ask_timeout_s: float = Field(default=45.0, gt=0)
    probe_ask_daily_cap: int = Field(default=3, ge=0)
    probe_state_path: str = "data/observe/probe_state.json"
    # Online sampling (spec P5-02, `python -m rag.eval.online_sample`): `online_eval_n` random
    # requests of the last `online_eval_window_hours` hours (never more than 50), each judged once
    # by the sufficiency judge's model profile; the job refuses to start when its estimated cost
    # (`online_eval_est_cny_per_call` per request) would pass the ledger's stop, and stops after
    # `online_eval_max_cny` of its own spend.
    online_eval_n: int = Field(default=20, ge=1)
    online_eval_window_hours: float = Field(default=168.0, gt=0)
    online_eval_max_cny: float = Field(default=0.20, gt=0)
    online_eval_est_cny_per_call: float = Field(default=0.007, gt=0)
    online_eval_report_dir: str = "data/reports/online"
    # Alerts (spec P5-02): Prometheus posts to the receiver (`python -m rag.monitor receiver`)
    # at `alert_receiver_host:alert_receiver_port` (Alertmanager's 9093 + 20000); it writes
    # `alert_log_path` and, when `ALERT_WEBHOOK_URL` is set, posts each change there too.
    alert_receiver_host: str = "127.0.0.1"
    alert_receiver_port: int = Field(default=29093, gt=0, lt=65536)
    alert_log_path: str = "data/observe/alerts.jsonl"
    alert_webhook_url: SecretStr = SecretStr("")
    # Trace files of the Collector (`python -m rag.monitor prune`): anything matching the pattern
    # older than the days, or beyond the total size, goes (the newest file always stays).
    trace_retention_dir: str = "data/otel"
    trace_retention_pattern: str = "traces*"
    trace_retention_days: float = Field(default=3.0, gt=0)
    trace_retention_max_mb: int = Field(default=100, gt=0)
    # JSON log lines in `<log_dir>/<service name>.jsonl` (relative to the repo root; empty: no
    # file); stderr gets lines at `log_stderr_level` and above (OFF: none).
    log_dir: str = "data/observe/logs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_stderr_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "OFF"] = "WARNING"
    log_max_bytes: int = Field(default=10_000_000, gt=0)
    log_backups: int = Field(default=3, ge=0)

    # Hugging Face data (spec P0-03). huggingface.co is blocked here; the mirror serves the same
    # `/datasets/<repo>/resolve/<revision>/<path>` URLs. `HF_ENDPOINT` is also what huggingface_hub
    # reads. Point `HF_DATA_DIR` at one shared directory so several checkouts reuse one download.
    hf_endpoint: str = "https://hf-mirror.com"
    hf_data_dir: str = "data/raw/hf"
    hf_timeout_s: float = Field(default=60.0, gt=0)

    # Ingestion job (spec P0-06, design s7-4, s16-8). A run stops before changing anything when
    # more than `ingest_max_read_failure_ratio` of the documents fail to read, or when it would
    # pass fewer than `ingest_min_input_ratio` times the documents in the docstore (a source that
    # failed to mount looks like a mass delete). Reports go to `<ingest_report_dir>/<kb>/`.
    ingest_max_read_failure_ratio: float = Field(default=0.05, ge=0, le=1)
    ingest_min_input_ratio: float = Field(default=0.5, ge=0, le=1)
    ingest_bulk_batch_size: int = Field(default=500, gt=0)
    ingest_lock_timeout_s: float = Field(default=5.0, ge=0)
    ingest_report_dir: str = "data/reports"

    # New index version (spec P2-01, design s7-5). The force merge of the finished new index may
    # take minutes on a full corpus; `rebuild_min_chunk_ratio` is the floor of the built-in
    # sanity gate (the new index must hold at least this share of the old index's chunks).
    rebuild_force_merge_timeout_s: float = Field(default=3600.0, gt=0)
    rebuild_min_chunk_ratio: float = Field(default=0.5, ge=0, le=1)

    # Query path (spec P0-07, design s16-1, s16-2). `search_deadline_ms`: total time of one
    # search when the caller passes no deadline_ms. While waiting for the question vector from
    # the gateway the retriever keeps min(`search_es_reserve_ms`, `search_es_reserve_ratio` x
    # the request's time limit) for the Elasticsearch request (an `_msearch` takes ~20 ms); each
    # search inside the `_msearch` gets an ES time-out `search_es_margin_ms` before the HTTP
    # request's (at most half of the time left, so the searches keep the other half), so a slow
    # route comes back as timed out while the other route still counts.
    search_deadline_ms: int = Field(default=10_000, gt=0)
    search_es_reserve_ms: int = Field(default=200, ge=0)
    search_es_reserve_ratio: float = Field(default=0.2, ge=0, lt=1)
    search_es_margin_ms: int = Field(default=200, ge=0)
    # Search service (spec P0-09, design s16-1). A caller's deadline_ms must lie in
    # 1..`search_max_deadline_ms` and its query be at most `search_max_query_chars` long.
    # `search_index_prefix` selects other indices (`fake_` = the fake-vector index, used with
    # EMBED_BACKEND=fake so no query embedding costs money).
    search_max_deadline_ms: int = Field(default=60_000, gt=0)
    search_max_query_chars: int = Field(default=2000, gt=0)
    search_index_prefix: str = ""

    # Result cache (spec P5-01, design s12). `cache_mode`: `off`; `l1` = exact cache in Redis for
    # `search_docs` and `ask_docs`; `l1_shadow` = l1 plus the semantic cache in shadow mode (it
    # records what it would have returned and never returns it). `cache_url`: empty = `redis_url`;
    # the Redis keys start with `cache_key_prefix`. The store gets `cache_timeout_ms` for connecting
    # and for each command (a slower Redis counts as down: the request is computed without the
    # cache); after `cache_breaker_failures` failures in a row nothing is sent for
    # `cache_breaker_open_s` seconds. A result lives `cache_ttl_s` (`cache_not_found_ttl_s` when it
    # is a not-found result); a value above `cache_max_value_bytes` is not stored (no compression).
    # Identical requests that arrive together compute once: the one that computes holds a Redis
    # lock for at most `cache_lock_ttl_ms`; the others in other processes poll every
    # `cache_lock_poll_ms` for at most `cache_lock_wait_ms`. Shadow mode: `cache_semantic_threshold`
    # is how far the shadow records look: the largest cosine distance (1 - cosine similarity) of a
    # stored question that counts as a candidate. 0.20 is wide on purpose, so real traffic can show
    # which smaller value is safe (`shadow-report --max-distance`); 0.06 is the value chosen on the
    # dev pairs of P5-01 (docs/reports/P5-report.md), not a default. Entries live
    # `cache_semantic_ttl_s`, the records go to `<cache_shadow_dir>/shadow.jsonl`, and the stored
    # questions live in `cache_semantic_store` (`memory` in this process, or `es`: index
    # `cache_semantic_<kb>`; it creates that index in the real Elasticsearch and deletes expired
    # entries every 500 writes). `cache_semantic_lexical_guard`: besides the entity guard of the
    # design, reject a stored question whose content words differ from the new one's by one or
    # two words (or are the same words in another order). `cache_ask_answers`: also cache the
    # final answers of `ask_docs` (design s12-1 leaves L3, the answer cache, out; see
    # docs/decisions.md): off, only `search_docs` results are cached.
    cache_mode: Literal["off", "l1", "l1_shadow"] = "l1_shadow"
    cache_url: str = ""
    cache_ask_answers: bool = True
    cache_key_prefix: str = "rag:l1:"
    cache_timeout_ms: int = Field(default=50, gt=0)
    cache_ttl_s: int = Field(default=3600, gt=0)
    cache_not_found_ttl_s: int = Field(default=600, gt=0)
    cache_max_value_bytes: int = Field(default=262_144, gt=0)
    cache_breaker_failures: int = Field(default=5, ge=1)
    cache_breaker_open_s: float = Field(default=5.0, gt=0)
    cache_lock_ttl_ms: int = Field(default=30_000, gt=0)
    cache_lock_wait_ms: int = Field(default=3_000, ge=0)
    cache_lock_poll_ms: int = Field(default=25, gt=0)
    cache_semantic_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    cache_semantic_ttl_s: int = Field(default=86_400, gt=0)
    cache_semantic_lexical_guard: bool = True
    cache_semantic_store: Literal["memory", "es"] = "memory"
    cache_shadow_dir: str = "data/observe/cache"

    # Search service process (spec P0-09, design s16-1, s16-4). Until P8 (OAuth) every request
    # runs as one caller from config: `service_caller` is matched against `access.callers` of
    # each KB config and `service_acl_groups` are the document ACL groups it may read. The
    # HTTP API and the MCP streamable-http endpoint listen on `service_host:service_port`.
    service_caller: str = Field(default="local", min_length=1)
    service_acl_groups: list[str] = Field(default_factory=lambda: ["public"], min_length=1)
    service_host: str = "127.0.0.1"
    service_port: int = Field(default=28080, gt=0, lt=65536)

    # The ask_docs graph (spec P3-01, design s13-3, s16-4, s16-8). `agent_deadline_ms`: total time
    # of one ask_docs request (a caller's deadline_ms replaces it, at most `search_max_deadline_ms`;
    # MCP clients time out tool calls). `agent_answer_reserve_ms`: time kept for writing the
    # answer: no further round starts when less is left. The answer step of the graph runs of
    # P3-05 and P3-06 (208 answers): median 2.7 s, P95 4.7 s, P99 6.0 s, 6.8 to 7.0 s at P99 in
    # the two runs made on a busy machine, longest 7.4 s; so 7 s (spec P3-06; it was 12 s, the
    # online chat profile's own deadline, which made round one end by 10 s).
    # `agent_recursion_limit`: LangGraph steps of one run (route, rewrite, retrieve, judge
    # count one each; three rounds need about 12). `agent_token_cap`: model tokens of one request
    # (usage from the gateway, else an estimate); `agent_answer_reserve_tokens`: tokens kept for
    # the answer, so the calls before it (rewrite, judge) stop earlier. `agent_history_turns`:
    # past turns of a session given to the graph (0 = no history). `checkpoint_url`: one
    # checkpoint thread per request. `session_store_url`: the turns of each session;
    # `session_lock_url`: `memory://` serialises one session's requests inside this process,
    # `file://<dir>` also across processes on this machine (design s16-7).
    #
    # Route fallback and rewrite (spec P3-02, design s8-2, s11-1): `agent_rewrite_profile` is the
    # chat profile of the small model that rewrites the question and decides what the rules
    # cannot (the `cheap` profile: short timeouts, 1,024 output tokens, reasoning included);
    # `agent_rewrite_timeout_ms` is the longest one such call may take (it never uses the time
    # kept for the answer, whatever this says); `agent_rewrite_history_turns` and
    # `agent_rewrite_history_chars` bound the earlier turns shown to it (0 turns: no history
    # for rewriting). `agent_resume_ttl_s`: how long a question asked back waits for its answer
    # (the `resume_token` is valid that long, once).
    agent_deadline_ms: int = Field(default=30_000, gt=0)
    agent_answer_reserve_ms: int = Field(default=7_000, ge=0)
    agent_recursion_limit: int = Field(default=25, gt=0)
    agent_token_cap: int = Field(default=20_000, gt=0)
    agent_answer_reserve_tokens: int = Field(default=6_000, ge=0)
    agent_history_turns: int = Field(default=6, ge=0)
    agent_rewrite_profile: ChatProfileName = "cheap"
    agent_rewrite_timeout_ms: int = Field(default=6_000, gt=0)
    agent_rewrite_history_turns: int = Field(default=3, ge=0)
    agent_rewrite_history_chars: int = Field(default=400, gt=0)
    agent_resume_ttl_s: int = Field(default=1_800, gt=0)
    # Sufficiency judge and correction rounds (spec P3-03, design s13-4): `agent_judge_profile` is
    # the chat profile of the model that says whether every sub-question has grounds in what
    # was found; `agent_judge_timeout_ms` the longest one such call may take (it never uses the
    # time kept for the answer); `agent_judge_context_tokens` caps the fragments sent to it
    # (estimated tokens, shared between the sub-questions); `agent_round_min_ms` is the time one
    # more round needs (a small-model call, a search, a judgement): no round starts when less
    # than this, plus the answer's time, is left. Measured on the `cheap` model (pilot of 10
    # judgements): 3.0 to 5.6 s per judgement, so 6 s timed out half of the calls and the retry
    # still billed them (0.02 CNY each against 0.003 for a call that finished): hence 9 s.
    agent_judge_profile: ChatProfileName = "cheap"
    agent_judge_timeout_ms: int = Field(default=9_000, gt=0)
    agent_judge_context_tokens: int = Field(default=1_500, gt=0)
    agent_round_min_ms: int = Field(default=8_000, ge=0)
    # Spec P3-06 (P3-05 findings): `agent_round_min_ms` is the floor; a further round needs at
    # least 0.9 x the time the last round took (`agent_round_ratio`), so a slow first round does
    # not start a second one that cannot end. `agent_search_min_ms`: the least time a search of
    # round two or later may get (what is left minus the answer's time): a search with less
    # embeds the question and then has no time to rerank (P3-05: 54 such searches, all without
    # a rerank score), so it is not made. Measured on a busy machine (P3-06 review): a rerank
    # batch of 8 pairs took 2.5 to 2.7 s (0.31 to 0.34 s per pair, 3.3 s for 10), a cold query
    # embedding plus the search about 1.2 s, so a whole search 4.5 s: 6 s. On a quiet machine
    # (0.12 to 0.15 s per pair) a search takes about 3 s.
    # `agent_rerank_reserve_ms`: the query-vector wait of such a search ends this long before its
    # deadline (at most half of it), so the rerank has time (3 s: the 3.3 s above, rounded
    # down). Only these searches get it: `search_docs` and the retrieval evals wait as before.
    # `agent_answer_max_tokens`: output tokens (reasoning included) of the graph's answer call;
    # the `online` profile's 4096 let one answer run to 4096 tokens and past the deadline
    # (P3-06, no_answer_hard:03). Of 208 answer calls in the P3-05 and P3-06 graph runs the
    # median was 101 tokens, P99 731, the longest normal one 1,058; 1,536 keeps all of them.
    agent_round_ratio: float = Field(default=0.9, ge=0, le=2)
    agent_search_min_ms: int = Field(default=6_000, ge=0)
    agent_rerank_reserve_ms: int = Field(default=3_000, ge=0)
    agent_answer_max_tokens: int = Field(default=1_536, gt=0)
    # Streaming endpoint (spec P3-04, design s16-9): `sse_heartbeat_s` is how often a comment
    # line is sent while the stream is silent (the retrieval and rerank seconds), so a proxy
    # that closes idle connections after 30 to 60 s keeps it open.
    sse_heartbeat_s: float = Field(default=10.0, gt=0)
    # `sse_queue_max_events`: the events of one stream wait in a queue until the client reads
    # them; past this many the run is cancelled (a client that stopped reading). A normal
    # answer is a few hundred events. `service_shutdown_timeout_s`: at shutdown the server
    # cancels the open streams at once and waits this long for the connections to close,
    # then cancels what is left. `service_allowed_hosts`: host names, besides localhost,
    # 127.0.0.1, [::1] and the listen address, that the HTTP routes answer to (the Host
    # header); anything else is refused with 421.
    sse_queue_max_events: int = Field(default=5000, gt=0)
    service_shutdown_timeout_s: int = Field(default=5, gt=0)
    service_allowed_hosts: list[str] = Field(default_factory=list)
    session_store_url: str = "sqlite://data/sessions/sessions.db"
    session_lock_url: str = "memory://"

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
    def _default_deadline_within_max(self) -> Self:
        if self.search_deadline_ms > self.search_max_deadline_ms:
            raise ValueError(
                f"search_deadline_ms ({self.search_deadline_ms}) must not exceed "
                f"search_max_deadline_ms ({self.search_max_deadline_ms})"
            )
        if not all(isinstance(g, str) and g for g in self.service_acl_groups):
            raise ValueError("service_acl_groups must be non-empty group names")
        return self

    @field_validator("service_allowed_hosts")
    @classmethod
    def _host_names(cls, value: list[str]) -> list[str]:
        for name in value:
            if not re.fullmatch(r"[A-Za-z0-9.-]+|\[[0-9A-Fa-f:]+\]", name):
                raise ValueError(f"service_allowed_hosts: {name!r} is not a host name")
        return value

    @model_validator(mode="after")
    def _agent_limits_consistent(self) -> Self:
        if self.agent_deadline_ms > self.search_max_deadline_ms:
            raise ValueError(
                f"agent_deadline_ms ({self.agent_deadline_ms}) must not exceed "
                f"search_max_deadline_ms ({self.search_max_deadline_ms})"
            )
        if self.agent_answer_reserve_ms >= self.agent_deadline_ms:
            raise ValueError(
                f"agent_answer_reserve_ms ({self.agent_answer_reserve_ms}) must be below "
                f"agent_deadline_ms ({self.agent_deadline_ms}): the search needs time too"
            )
        if self.agent_answer_reserve_tokens >= self.agent_token_cap:
            raise ValueError(
                f"agent_answer_reserve_tokens ({self.agent_answer_reserve_tokens}) must be "
                f"below agent_token_cap ({self.agent_token_cap})"
            )
        return self

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
