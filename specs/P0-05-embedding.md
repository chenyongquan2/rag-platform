# P0-05 Embeddings, L0 cache, cost ledger, budget check

Status: review
Phase: P0
Depends on: P0-01
Design sections: s7-4 ("算向量"), s12-1 (L0 row), s16-7, s17-1 (embedding client); also s16-2 (timeouts per profile), s16-8 (embedding errors)
Calls real models: yes (batch-size probe and one `llm` test; spent 0.00004 CNY)

## Goal
One embedding factory used by ingestion and query (same model, dims and text prefix; only timeout and retries
differ by profile) that returns a LlamaIndex `BaseEmbedding` talking to the OpenAI-compatible gateway, wrapped in
an L0 cache (SQLite for ingest, Redis for query), retrying transient errors, writing every real call to the cost
ledger and refusing calls over the budget; plus a `FakeEmbedding` fallback switch and `python -m rag.cost summary`.

## Interface
- `ServiceSettings` adds: `embed_backend: "gateway" | "fake"`, `embed_text_prefix`, `embed_batch_size`,
  `embed_ingest_timeout_s`, `embed_ingest_max_retries`, `embed_ingest_backoff_s`, `embed_query_timeout_s`,
  `embed_query_max_retries` (0 or 1), `embed_query_backoff_s`, `embed_backoff_max_s`, `embed_ingest_cache_url`
  (`sqlite://<path>` | `memory://`), `embed_query_cache_url` (empty = `redis_url`; `redis://` | `memory://`),
  `embed_cache_ttl_s`, `embed_cache_redis_timeout_s`, `model_prices: dict[str, ModelPrice(input_cny_per_m, output_cny_per_m)]`.
- `rag.embed`: `make_embedding(profile: "ingest" | "query", settings=None, *, cache=None, ledger=None, transport=None, sleep=None) -> BaseEmbedding`;
  `GatewayEmbedding(BaseEmbedding)`; `EmbeddingError(retryable, status_code)`; `CachedEmbedding(BaseEmbedding)`;
  `EmbeddingCache` protocol (`get_many(model, hashes)`, `put_many(model, vectors)`), `SqliteEmbeddingCache`,
  `RedisEmbeddingCache`, `InMemoryEmbeddingCache`, `make_embedding_cache(url, *, ttl_s, redis_timeout_s)`,
  `cache_model_key(model, dim, text_prefix="")`, `text_sha256(text)`.
- `rag.cost`: `CostLedger(path, *, prices, budget_stop_cny)` with `from_settings`, `price`, `estimate`, `record`,
  `entries`, `total_spent`, `check_budget`, `summary`; `total_spent(settings=None)`, `check_budget(estimated_cny, settings=None)`,
  `BudgetExceeded`, `UnknownModelPriceError`; CLI `uv run python -m rag.cost summary [--ledger PATH]`.
- `data/cost/ledger.jsonl` is tracked in git with `merge=union` (`.gitattributes`).

## Acceptance criteria
Unit tests in `tests/unit/test_cost.py` and `tests/unit/test_embed.py` use `httpx.MockTransport` (no network).
- [x] AC1 Embedding settings have defaults and env overrides; query retries above 1 are rejected — `test_embed.py::test_embed_settings_defaults_and_limits`
- [x] AC2 `record` appends one JSON line (time, kind, model, tokens, CNY from the price table); an unknown model is rejected — `test_cost.py::test_record_appends_json_line_with_cny`, `test_cost.py::test_unknown_model_price_is_rejected`
- [x] AC3 `total_spent` sums the ledger (a truncated last line is skipped); `check_budget` raises `BudgetExceeded` when spent + estimate > `budget_stop_cny` — `test_cost.py::test_total_spent_sums_ledger`, `test_cost.py::test_check_budget_raises_over_stop`
- [x] AC4 `python -m rag.cost summary` prints spent, budget and totals per kind and model; the ledger is tracked with union merge — `test_cost.py::test_cli_summary_prints_totals`, `test_cost.py::test_ledger_is_tracked_with_union_merge`
- [x] AC5 The client sends at most `embed_batch_size` texts per request with model, dimensions and prefix, and returns vectors in input order — `test_embed.py::test_client_batches_requests_in_order`
- [x] AC6 Every real request appends one ledger line with the `usage` tokens and the profile — `test_embed.py::test_client_records_each_call_in_ledger`
- [x] AC7 Timeouts, connection errors, 429 and 5xx are retried with backoff (`Retry-After` honoured); after the last retry `EmbeddingError` — `test_embed.py::test_client_retries_transient_errors`, `test_embed.py::test_client_gives_up_after_max_retries`
- [x] AC8 Other 4xx, malformed responses and wrong dimensions fail after one attempt — `test_embed.py::test_client_fails_fast_on_non_retryable_errors`
- [x] AC9 A request whose estimate would pass the budget stop is not sent — `test_embed.py::test_client_refuses_call_over_budget`
- [x] AC10 Cache key is (model, dims, prefix) + sha256(text): hits skip the gateway, duplicates are sent once, another model misses — `test_embed.py::test_cache_hit_and_miss`
- [x] AC11 An interrupted run keeps finished batches in the cache; the rerun requests only the missing texts — `test_embed.py::test_interrupted_run_resumes_from_cache`
- [x] AC12 SQLite cache persists across instances; Redis cache uses one key per vector with TTL and treats Redis errors as misses; the cache factory picks by scheme — `test_embed.py::test_sqlite_cache_persists`, `test_embed.py::test_redis_cache_keys_ttl_and_errors`, `test_embed.py::test_make_embedding_cache_by_scheme`
- [x] AC13 Both profiles share model, dims, prefix and URL and differ only in timeout and retries; ingest uses SQLite, query uses `redis_url` — `test_embed.py::test_factory_profiles_share_model_and_differ_in_timeouts`
- [x] AC14 `embed_backend=fake` returns `FakeEmbedding(embed_dim)`: no HTTP, no ledger line — `test_embed.py::test_fake_backend_fallback`
- [x] AC15 The cached client works as a LlamaIndex transformation in `IngestionPipeline` (sync and async) — `test_embed.py::test_works_in_ingestion_pipeline`
- [x] AC16 (`llm`) Two short texts through the real gateway: 1024 dims, the query profile agrees with the ingest profile, a cache hit returns the identical vector — `tests/llm/test_embed_gateway.py::test_real_gateway_embeds_two_texts`
- [x] AC17 (`integration`) Redis cache round trip on the compose Redis — `tests/integration/test_embed_cache_redis.py::test_redis_cache_round_trip`

Also (checked by running, not by a test): the gateway accepts at most 10 inputs per request (11 gives HTTP 400),
so `embed_batch_size` defaults to 10; method and timings in `docs/decisions.md`. AC16 and AC17 ran once and pass.

## Out of scope
Using the client in `IngestionPipeline` with `EsVectorStore` and a job-level budget check before a full ingest
(P0-06); query embedding inside the retriever and the `degraded` flag when it fails (P0-07); chat model calls and
their ledger lines (P1-01); circuit breaker (P8).

## Handoff (filled at the end)
- Files: `pyproject.toml`, `uv.lock` (httpx as a direct dependency), `.gitattributes`, `.gitignore`, `.env.example`, `README.md`, `src/rag/config/{settings,__init__}.py`, `src/rag/cost/{__init__,__main__,cli,ledger}.py`, `src/rag/embed/{__init__,client,cache,factory}.py`, `tests/unit/{test_cost,test_embed}.py`, `tests/llm/test_embed_gateway.py`, `tests/integration/test_embed_cache_redis.py`, `data/cost/ledger.jsonl`, `docs/decisions.md`.
- Decisions (docs/decisions.md): own httpx `BaseEmbedding` subclass instead of `OpenAILikeEmbedding` (usage tokens, one retry layer, no new packages); gateway max 10 inputs per request; ingest 60 s and 4 retries, query 3 s and 1 retry; cache key `model@dim(+prefix hash)` + sha256(text), float32 values; query cache defaults to `redis_url`, Redis errors count as misses, own Redis client because `make_redis_client` decodes responses; a ledger line for every 2xx answer and a budget check before every request; ledger committed with `merge=union`; chat price is a high placeholder (2 and 8 CNY per million); the fake backend is switched by hand, never automatically.
- Open issues: parallel checkouts calling real models must share one `COST_LEDGER_PATH`; the `deepseek-v4-flash` price is unverified (P1-01); an index must record whether it was built with fake or real vectors (P0-06); `IngestionPipeline(num_workers>1)` would pickle the httpx client and SQLite connection and is not supported; a persistent 503 (e.g. the plain `text-embedding-v4` id) is retried until the retries are used up.
- Review rounds: 0 (waiting for review). 26 unit tests in this spec; 86 unit tests pass in total.
