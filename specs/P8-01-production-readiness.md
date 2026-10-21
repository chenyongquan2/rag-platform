# P8-01 Production readiness: OAuth bearer tokens, access-leak check, rate limit, daily budget, gateway breaker, backup drill, load test

Status: review
Phase: P8
Depends on: P0-09 (service, ACL), P3-04 (stream), P5-01 (cache keys), P5-02 (alerts)
Design sections: s16-4, s16-6, s16-1, s16-5, s16-8, s12-1, s15-1
Calls real models: once: 92 query embeddings to warm the L0 cache, about 0.0009 CNY (allowed 0.02)

## Goal
The HTTP API and MCP streamable-http accept OAuth 2.1 bearer JWTs (resource-server role, SDK `TokenVerifier`/`AuthSettings`), scopes map to callers;
per-caller rate limits and daily budgets refuse requests before work; repeated chat-gateway failures open a breaker; a leak check, a logical
backup/restore drill and a load test produce real numbers (docs/reports/P8-report.md). Local development keeps the one configured caller.

## Interface
- Settings: `auth_mode` (`local` default | `oauth`), `auth_issuer`, `auth_audience`, `auth_jwks_path` | `auth_jwks_url`, `auth_algorithms`
  (asymmetric only), `auth_leeway_s`, `auth_resource_url`, `auth_callers` (list of `{scope, caller, acl_groups}`); `rate_limit_mode`
  (`off`|`redis`|`memory`), `rate_limit_per_minute`, `rate_limit_callers`, `rate_limit_key_prefix`, `rate_limit_timeout_ms`;
  `budget_daily_cny`, `budget_daily_caller_cny`, `budget_daily_callers`, `budget_ask_estimate_cny`; `chat_breaker_failures`, `chat_breaker_open_s`.
- `rag.security.tokens.JwtTokenVerifier(TokenVerifier)`, `make_token_verifier(settings)`; `rag.security.access.ScopeAclResolver`, `AccessDenied`.
  `rag.testing.auth.TestIssuer` (RSA key pair, JWKS, `token(...)`).
- Error categories `unauthorized` (401), `forbidden` (403), `rate_limited` (429 + Retry-After); `budget_exceeded` before a call is 429 + Retry-After.
- `rag.serve.limits`: `RateLimiter` (`RedisRateLimiter`, `MemoryRateLimiter`; fixed one-minute window; Redis errors let the request through),
  `DailyBudget` (ledger rows of today, UTC, per caller and in total). Ledger rows carry `caller` and `request_id` of the current request.
- `rag.llm.breaker.CircuitBreaker` (closed / open / half-open, one probe), `shared_breaker(key, ...)`; `LedgerTransport`/`AsyncLedgerTransport`
  take `breaker=`; open breaker raises `CircuitOpenError` (an `httpx.TransportError`, so the category stays `llm_error`). Counter `rag.model.breaker{rag.breaker.event}`; alert `ModelGatewayBreakerOpen`.
- CLIs: `python -m rag.eval.access` (leak report), `python -m rag.ops.backup export|restore|verify|copy-files|rebind-docstore`, `python -m rag.ops.loadtest`.
  `deploy/elasticsearch/snapshot.override.yml` (not applied).

## Acceptance criteria
- [x] AC1 Valid token reaches the tool; expired, wrong audience, wrong issuer, bad signature, missing token, `alg: none`/HS256 are 401 on HTTP and MCP — `tests/unit/test_auth_tokens.py`, `tests/unit/test_serve_auth.py::test_http_*`, `::test_mcp_*`
- [x] AC2 Scope maps to caller and groups; no or several mapped scopes is `forbidden`; a caller without the KB gets `kb_not_found`; stdio stays the local caller; local mode unchanged — `tests/unit/test_serve_auth.py::test_scope_*`, `::test_stdio_*`, `::test_local_mode_*`
- [x] AC3 Leak check: two callers, every tool incl. stream and cache, 0 leaked chunks (fakes; real ES in the integration test) — `tests/unit/test_access_leaks.py`, `tests/integration/test_access_leaks_es.py`
- [x] AC4 Rate limit per caller: 429 + Retry-After on HTTP, MCP error, other callers unaffected, window resets, Redis down lets requests through, Redis keys under the prefix — `tests/unit/test_rate_limit.py`, `tests/integration/test_rate_limit_redis.py`
- [x] AC5 Daily budget per caller and global refuses ask_docs before any model call; search_docs still works; yesterday does not count — `tests/unit/test_daily_budget.py`
- [x] AC6 Breaker opens after N failures, fails fast without sending, half-open probe closes or reopens; 4xx and budget errors do not count; ask_docs degrades to sources, search_docs works; metric and alert rule — `tests/unit/test_llm_breaker.py`, `tests/unit/test_monitor_config.py`, promtool
- [x] AC7 Logical export/restore round-trips mapping, ids and sources (vectors included) into a new index; docstore rebinding writes the new uuid — `tests/unit/test_ops_backup.py`, `tests/integration/test_ops_backup_es.py`
- [x] AC8 Load-test runner computes throughput, P50/P95 and errors per concurrency — `tests/unit/test_ops_loadtest.py`
- [x] AC9 (manual) Leak report, backup drill timings, load-test table in docs/reports/P8-report.md.

## Out of scope
Running an authorization server (tokens come from the company IdP); token introspection; admin token for ingest over HTTP (no ingest endpoint exists);
applying the ES snapshot repository (needs an ES restart); Langfuse deletion by doc_id; multi-instance deployment.

## Handoff (filled at the end)
- Files: src/rag/security/tokens.py, access.py; src/rag/serve/{http_app,mcp_server,service,cli,errors,limits}.py; src/rag/llm/{breaker,transport,factory}.py; src/rag/cost/ledger.py; src/rag/eval/access/; src/rag/ops/{backup,loadtest}.py; deploy/prometheus/rules*.yml; deploy/elasticsearch/snapshot.override.yml; README, .env.example; docs/reports/P8-report.md.
- Decisions: docs/decisions.md 'P8-01'. Scope -> exactly one caller (else forbidden); rate limit fails open; budget from ledger rows with caller; breaker per gateway URL, llm_error category kept.
- Open issues: snapshot override not applied (needs ES restart); real search P95 3.3 s at concurrency 1 (target 3 s), about 0.35 req/s on CPU rerank; no admin token (no ingest endpoint); no embedding-gateway breaker; mt5/pdf_demo docstores not in the main checkout, not copied.
- Review rounds: none yet.
