# P0-09 Search service: MCP tools and HTTP API

Status: review
Phase: P0
Depends on: P0-01 (settings, KB config `access`), P0-07 (filter whitelist, `FilterError`, `RetrievalError`), P0-08 (`SearchPipeline`, `get_context`, one reranker per process), P0-10 (`routes` dropped at the response boundary), P0-11 (`init_observability`, `request_span`, `layer_span`, metrics, business records)
Design sections: s16-1, s16-4, s16-8, s15-1, s7-6 (also s16-2 package layout `serve/`, `security/`)
Calls real models: no (integration tests use the fake-vector index `fake_kb_hf_docs` with `EMBED_BACKEND=fake`; the reranker is local)

## Goal
One process serves `search_docs`, `get_context` and `list_knowledge_bases` as MCP tools (official `mcp` 2.3
`MCPServer`, stdio and streamable-http) and as a JSON HTTP API (Starlette, same app as the MCP endpoint), for
one caller identity from config behind an ACL resolver interface that P8 (OAuth) replaces. The reranker loads
once at start; readiness follows the load. Every call gets a request span, metrics and one business record line;
errors return only a category and the request id.

## Interface
- `rag.security.access`: `Caller(caller_id, acl_groups)`, `Credentials(transport, client_id=None, subject=None, scopes=())`,
  `AclResolver` protocol `resolve(credentials) -> Caller`, `ConfigAclResolver(caller_id, acl_groups)` / `.from_settings(settings)`,
  `kb_visible(caller, cfg)`, `visible_kbs(caller, configs)`.
- `rag.serve.errors`: `ServiceError(category, request_id)`; `error_category(exc) -> str`; `http_status(category)`;
  `error_message(category)` (fixed text per category, never the exception text). New `ErrorCategory`
  values: `invalid_request`, `kb_not_found`, `not_ready`, `internal_error`.
- `rag.serve.schema`: `SearchResponse`, `ContextResponse`, `KnowledgeBaseInfo`, `KnowledgeBaseList`, `ErrorResponse`.
- `rag.serve.service.SearchService(settings, *, configs=None, resolver=None, builder=None)` (`PipelineBuilder`: `prepare`, `build(cfg)`, `close`; see docs/decisions.md):
  `start()` (load in a background thread), `load()`, `ready`, `status()`, `wait_ready(timeout_s)`, `close()`, `reject(...)` (a body the transport refused);
  `search_docs(credentials, *, kb, query, filters=None, top_k=None, deadline_ms=None, request_id=None) -> SearchResponse`,
  `get_context(credentials, *, chunk_id, mode=None, request_id=None) -> ContextResponse`,
  `list_knowledge_bases(credentials, *, request_id=None) -> KnowledgeBaseList`; raise `ServiceError`.
- `rag.serve.mcp_server.build_mcp_server(service) -> MCPServer` (tools `search_docs(kb, query, filters, top_k, deadline_ms)`,
  `get_context(chunk_id, mode)`, `list_knowledge_bases()`); `rag.serve.http_app.build_http_app(service, *, mcp=None) -> Starlette`
  (`POST /v1/search_docs`, `POST /v1/get_context`, `GET /v1/knowledge_bases`, `GET /healthz`, `GET /readyz`, MCP at `/mcp`).
- CLI `uv run python -m rag.serve stdio` and `uv run python -m rag.serve http [--host H] [--port P]`.
- `HybridSearch.retrieve(..., acl_groups=None)` and `SearchPipeline.get_context(..., acl_groups=None)` (per-request caller groups);
  `KnowledgeBaseConfig.description` (left out of `search_config_version`).
- Settings: `service_caller` (`local`), `service_acl_groups` (`["public"]`), `service_host` (`127.0.0.1`), `service_port` (28080),
  `search_index_prefix` (``), `search_max_deadline_ms` (60000), `search_max_query_chars` (2000).
- `Metrics.record_retrieval(kb, tool, not_found, top_rerank_score)` (`rag.requests.not_found`, `rag.rerank.top_score`).

## Acceptance criteria
Unit tests in `tests/unit/test_serve_*.py`, `tests/unit/test_security_access.py` (fake pipelines, in-memory observability); integration in `tests/integration/test_serve_*.py`. CLI: `test_serve_cli.py`.
- [x] AC1 The search response has request id, kb, index version, data generation, config version, degraded, rerank_timeout, not_found, error categories, timings, chunks and below-threshold chunks with `text`, `untrusted=true`, `suspicious_instruction`, source_url/source_version; `routes`, `rerank` and `threshold` are not in it — `test_serve_schema.py::test_search_response_fields_and_routes_excluded`
- [x] AC2 Context and KB-list responses; error response is category, request id and the fixed message — `test_serve_schema.py::test_context_kb_list_and_error_responses`
- [x] AC3 The config resolver returns the configured caller whatever the credentials; a KB is visible when `access.callers` has `*` or the caller id; `list_knowledge_bases` lists only visible KBs — `test_security_access.py::test_config_resolver_returns_configured_caller`, `::test_visible_kbs_follow_access_callers`, `test_serve_service.py::test_list_shows_only_permitted_kbs`
- [x] AC4 Unknown and hidden KBs give the same `kb_not_found` (search and get_context) — `test_serve_service.py::test_hidden_and_unknown_kb_give_same_error`
- [x] AC5 search_docs and get_context pass the caller's ACL groups per request to the retrieval layer, which uses them instead of the pipeline's — `test_serve_service.py::test_every_tool_uses_caller_acl_groups`, `test_retrieve_pipeline.py::test_per_request_acl_groups_override_the_pipeline`
- [x] AC6 Unknown filter keys are rejected with `invalid_filter` before retrieval runs — `test_serve_service.py::test_unknown_filter_key_rejected`
- [x] AC7 Empty or too long query, top_k outside 1..rerank_n, deadline outside 1..max, malformed chunk id, unknown mode give `invalid_request`; defaults are final_k and `search_deadline_ms` — `test_serve_service.py::test_invalid_arguments_rejected`, `::test_defaults_from_config`
- [x] AC8 Exceptions map to categories (filter, both retrieval categories, ES timeout and errors, unexpected = internal_error) and HTTP statuses; the exception text is logged with the request id, never returned — `test_serve_errors.py::test_error_category_mapping`, `::test_http_status_per_category`, `test_serve_service.py::test_error_details_only_in_logs`
- [x] AC9 Readiness: the load runs once in the background and logs its time; until it ends search/get_context wait for it within their deadline, then `not_ready`; a KB whose pipeline could not be built answers `retrieval_failed` while the others work — `test_serve_service.py::test_readiness_and_wait_for_load`, `::test_failed_kb_is_unavailable`
- [x] AC10 Observability: one request span per call (tool, request id, caller, kb, config version) with a `retrieve` layer span under it; step durations from the result timings; not_found and top rerank score metrics; one business record line per call (search ok and error, get_context, list) with rerank list, threshold, query and trace id — `test_serve_service.py::test_search_span_metrics_and_business_record`, `::test_error_and_other_tools_write_records`, `test_observe_metrics.py::test_retrieval_quality_metrics`
- [x] AC11 HTTP: the three operations return the service's JSON; unknown body fields `invalid_request`; `X-Request-Id` taken and echoed; `/healthz` 200 always, `/readyz` 503 until ready (counts only, no KB names); errors with the category's status; rejected bodies get a span and a record — `test_serve_http.py::test_http_operations`, `::test_http_errors_and_request_id`, `::test_health_and_readiness`, `::test_rejected_bodies_write_records`
- [x] AC12 MCP (in-process SDK client): three tools whose descriptions name each visible KB with its description and mark text as untrusted; output schemas; structured results equal the HTTP JSON; errors are `is_error` with category and request id only — `test_serve_mcp.py::test_tools_listed_with_kb_descriptions`, `::test_mcp_tools_return_structured_results`, `::test_mcp_errors_are_category_and_request_id`
- [x] AC13 Settings defaults; KB description does not change the config version — `test_serve_service.py::test_serve_settings_defaults`, `test_retrieve_pipeline.py::test_config_version_ignores_description`
- [x] AC14 (integration) The HTTP app in-process with the real reranker on `fake_kb_hf_docs`: readyz 503 then 200, question 025 finds its page first with every schema field and no routes, a filter narrows, an unknown filter is 400, get_context and the KB list work, a business record line carries the request and trace ids — `tests/integration/test_serve_http_es.py::test_http_end_to_end`
- [x] AC15 (integration) The SDK stdio client starts `python -m rag.serve stdio` (the README command), lists the tools, calls list_knowledge_bases and search_docs end to end; the business record line is written and the request's span reaches Jaeger through the Collector — `tests/integration/test_serve_mcp_stdio.py::test_mcp_stdio_end_to_end`

## Out of scope
`ask_docs` and `feedback` (P1/P3); OAuth token checks for streamable-http and per-token KB scopes (P8: a new
`AclResolver`); rate limits and budgets (P8); caches (P2); streaming answers (P3); the scheduled probe request (P5);
ingestion over HTTP with an admin token (later).

## Handoff (filled at the end)
- Files: `src/rag/serve/{__init__,__main__,cli,errors,schema,service,mcp_server,http_app}.py`, `src/rag/security/{__init__,access}.py`, `src/rag/testing/serve.py`, `src/rag/retrieve/{search,pipeline}.py`, `src/rag/observe/{attributes,metrics}.py`, `src/rag/config/{kb,settings}.py`, `src/rag/types.py`, `kb/public/hf_docs.yaml` (description), `tests/unit/test_serve_{schema,errors,service,http,mcp,cli}.py`, `tests/unit/test_security_access.py`, `tests/integration/test_serve_{http_es,mcp_stdio}.py`, `README.md`, `docs/decisions.md`. 419 unit, 35 integration tests.
- Measured (fake-vector index, real reranker, 30 candidates, busy CPU): search_docs end to end 16.7-17.0 s over HTTP (rerank 16.6-16.9 s, ES 108-138 ms), warm stdio search 16.6-17.0 s; reranker load 109-115 s in the stdio process; the first stdio search waited for it. No model cost.
- Decisions (docs/decisions.md): Starlette (SDK's own dependency, one app with `/mcp`); background load, searches wait within their deadline then `not_ready`; failed KB -> `retrieval_failed`, others work; unknown and hidden KB -> same `kb_not_found` (404); `PipelineBuilder` instead of two factories; fixed error messages, details only in logs; tool descriptions built at start; integration file renamed (pytest module names).
- Open issues: latency is dominated by `rerank_n: 30` (P0-12 sets `rerank_n`); a resolver exception (P8 OAuth) is answered `internal_error` without a span or record (resolve inside the call scope in P8); tool descriptions are fixed at start (per-session lists in P8); no auth on streamable-http until P8, keep `SERVICE_HOST` on 127.0.0.1.
- Review rounds: 2. Round 1 CHANGES (fixed: `/readyz` named every KB; `close()` during the load iterated a dict being filled and skipped the observability shutdown; HTTP bodies rejected before the service had no span or record; resolver failure outside the span left for P8). Round 2 APPROVE (note: a close during `prepare()` leaves the load thread's clients to process exit).
