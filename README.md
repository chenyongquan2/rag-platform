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
uv run pytest -m integration     # needs `docker compose up -d` (Elasticsearch, Redis)
uv run pytest -m llm             # calls the real model gateway and costs money; only when a spec says so
uv run pre-commit run --all-files
```

Test markers: `unit` (default; added to unmarked tests; no network, no Docker), `integration`
(local Docker services), `llm` (real model calls). The default run excludes `integration` and `llm`.
Unit tests use `rag.testing.FakeEmbedding` and `FakeChatModel`; LlamaIndex `Settings.llm` is a stub
that raises when called. The gitleaks hook is built from source and needs a Go toolchain on `PATH`.
