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

## Ingestion

```bash
uv run python -m rag.ingest.job --kb hf_docs --limit 50 --fake-embed  # no cost; writes fake_kb_hf_docs
uv run python -m rag.ingest.job --kb hf_docs --limit 50               # real embeddings (ledger, budget)
uv run python -m rag.ingest.job --kb hf_docs --all                    # every document
```

Each run reads and cleans every document, writes new and changed chunks, then deletes old ones;
the docstore (`data/docstore/<index>.json`) is saved only after the whole run succeeded, and a
report goes to `data/reports/<kb>/ingest-<run>.json`. A run stops without changing anything when
more than 5% of the documents fail to read, when it would pass fewer than half of the docstore's
documents (`--allow-mass-delete` to confirm), or when the estimate does not fit the budget.

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
`init_chat_model` against the gateway (`LLM_BASE_URL`, `LLM_API_KEY`). Profiles differ in connect
and read timeouts, a deadline per call (online 12 s, cheap 10 s), retries and `max_tokens` (which
covers reasoning and answer tokens of the reasoning model); override one with JSON, e.g.
`CHAT_ONLINE='{"read_timeout_s": 8}'`. The ledger transport does the retries (never after a read
timeout) and every attempt reserves its estimated cost and writes its usage to the cost ledger.
`rag.llm.structured_output(model, Schema)` returns Pydantic objects via `function_calling` (the
gateway rejects `json_schema`). `CHAT_BACKEND=fake` switches every profile to a `FakeChatModel`
with a fixed reply; it is never switched automatically.

## Retrieval evaluation

Any retriever with `retrieve(question, kb, top_k) -> RetrievalResult` can be evaluated
(`rag.eval.retrieval.Retriever`); `RetrievalResult.routes` carries each route's own list for
per-route metrics. Question sets live in `evals/datasets/<kb>/` as JSONL with page labels
(doc_ids) and optional chunk labels; the 65 HF questions are
`evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl` (source, license and changes in
the `NOTICE.md` next to it).

```python
cfg = load_kb_config(cfg_path, kb_root=settings.kb_dir)
record = run_retrieval_eval(retriever, load_question_set(path, kb=cfg.kb), kb_config=cfg)
# evals/reports/public/<run_id>/ for kb/public/ configs, else the git-ignored private/
write_run(record, public=is_public_kb_file(cfg_path, settings.kb_dir))
```

```bash
uv run python -m rag.eval.retrieval compare <run_a> <run_b>   # McNemar, permutation, changed questions
uv run python -m rag.eval.retrieval export-hf                 # rewrite the HF question set
```

A run records Hit@k, MRR, nDCG@k (ideal DCG capped at k) and Recall@k at page and chunk level for
the final list and each route, latency, git commit, KB config, index version, data generation,
question-set hash, model ids and the `uv.lock` hash. A k larger than the final list is capped
at its length (with a warning). `rerank_latency_curve` measures a reranker
for rerank_n 10, 20, 30.
