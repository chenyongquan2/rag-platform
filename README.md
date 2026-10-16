# rag-platform

Multi-knowledge-base RAG platform: Elasticsearch hybrid retrieval (BM25 + vector) with application-level RRF and cross-encoder reranking, LlamaIndex ingestion, LangChain model calls, LangGraph agentic RAG, MCP tools, and an evaluation harness.

Status: design done, implementation starting (phase P0).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
uv sync                          # create .venv from uv.lock (CPU-only torch)
cp .env.example .env             # fill in gateway URLs and keys; .env is git-ignored
uv run pre-commit install        # ruff, ruff format, unit tests, gitleaks on every commit
uv run pytest                    # unit tests only (default)
uv run pytest -m integration     # needs scripts/dev_up.sh (Elasticsearch, Redis, Collector, Jaeger)
uv run pytest -m llm             # calls the real model gateway and costs money; only when a spec says so
uv run pre-commit run --all-files
```

Test markers: `unit` (default; added to unmarked tests; no network, no Docker), `integration`
(local Docker services), `llm` (real model calls). The default run excludes `integration` and `llm`.
Unit tests use `rag.testing.FakeEmbedding` and `FakeChatModel`; LlamaIndex `Settings.llm` is a stub
that raises when called. The gitleaks hook is built from source and needs a Go toolchain on `PATH`.

## Local services

`docker-compose.yml` runs the infrastructure; the Python app runs on the host.

```bash
scripts/dev_up.sh                                   # docker compose up -d, then wait until healthy
uv run python -m rag.index.cli init --kb hf_docs    # create kb_hf_docs_v1 and the kb_hf_docs alias
uv run python -m rag.index.cli status --kb hf_docs  # alias, versions, document counts, data generation
docker compose down                                 # stop (Elasticsearch data stays in the es-data volume)
```

| Service | Image | Host address |
|---|---|---|
| Elasticsearch (single node, Basic, 2 GB heap) | `docker.elastic.co/elasticsearch/elasticsearch:9.5.5` | `http://127.0.0.1:29200` |
| Redis cache (allkeys-lru, 256 MB, no persistence) | `redis:8.10.2-alpine` | `redis://127.0.0.1:26379/0` |
| OTel Collector (OTLP in; traces to Jaeger and `data/otel/traces.jsonl`) | `otel/opentelemetry-collector-contrib:0.162.0` | gRPC `127.0.0.1:24317`, HTTP `127.0.0.1:24318` |
| Jaeger (all-in-one, in-memory) | `jaegertracing/jaeger:2.22.0` | UI `http://127.0.0.1:36686` |

Host ports are the standard port + 20000 and bound to 127.0.0.1 only. **Elasticsearch security is
disabled** (no TLS, no password) for local development; never expose this setup beyond the machine.
Connection defaults in `ServiceSettings` match these ports; override with `ES_URL`, `REDIS_URL`,
`OTLP_ENDPOINT`. Integration tests skip with a message when a service is not reachable, and use
`test<run>_`-prefixed indices and Redis keys that they delete afterwards.

## Embeddings and cost

`rag.embed.make_embedding("ingest" | "query")` builds every embedding model: the same gateway
model, dimensions and text prefix on both sides, only timeout and retries differ. Vectors go
through the L0 cache (`data/cache/embeddings.sqlite3` for ingestion, the cache Redis for queries).
`EMBED_BACKEND=fake` switches to `FakeEmbedding` when the gateway is down or the budget is used up.

```bash
uv run python -m rag.cost summary   # money spent so far, per kind and model
```

Every real model call appends a line to the cost ledger, by default
`<parent of the main checkout>/rag-platform-data/cost/ledger.jsonl` (here
`C:/workspace/rag-platform-data/cost/ledger.jsonl`): one git-ignored file shared by the main
checkout and every worktree, so the budget stop sees all spend. Each request reserves its
estimated cost first and stops with `BudgetExceeded` once spent + reserved + estimate would pass
`BUDGET_STOP_CNY`. At most `EMBED_MAX_CONCURRENCY` requests per client are in flight.

## Chat models

`rag.llm.make_chat_model("online" | "offline" | "cheap")` builds every chat model with
`init_chat_model` against the gateway (`LLM_BASE_URL`, `LLM_API_KEY`). Profiles differ in timeout,
retries and `max_tokens` (which covers reasoning and answer tokens of the reasoning model); override
one with JSON, e.g. `CHAT_ONLINE='{"timeout_s": 10}'`. Every HTTP attempt, including the SDK's
retries, reserves its estimated cost and writes its real usage to the same cost ledger.
`rag.llm.structured_output(model, Schema)` returns Pydantic objects via `function_calling` (the
gateway rejects `json_schema`). `CHAT_BACKEND=fake` switches every profile to a `FakeChatModel`
with a fixed reply; it is never switched automatically.
