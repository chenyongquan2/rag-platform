# P0-11 Observability building blocks

Status: review
Phase: P0
Depends on: P0-01 (settings, types), P0-02 (Collector, Jaeger), P0-05/P1-01 (cost ledger). Wired into
search_docs by P0-09/P0-12.
Design sections: s15-1, s17-8, s16-8 (error categories in result, business record and metrics)
Calls real models: no

## Goal
`rag.observe` gives every service one call that sets up OTel traces and metrics (OTLP to the Collector),
OpenInference auto-instrumentation of LlamaIndex and LangChain/LangGraph, structured JSON logs, and a
business-record JSONL writer; plus span helpers, request-id propagation and metric instruments that P0-09
uses at layer boundaries. Attribute and metric names live in one module.

## Interface
- `ServiceSettings` adds: `otel_mode: "otlp" | "memory" | "off"`, `otlp_protocol: "grpc" | "http/protobuf"`,
  `otlp_timeout_s`, `otel_metric_export_interval_s`, `otel_auto_instrument`, `otel_trace_content`, `git_commit`,
  `business_record_path` (relative to the repo root, empty disables), `business_record_max_bytes`,
  `business_record_backups`, `log_dir` (empty disables the file), `log_level`, `log_stderr_level`,
  `log_max_bytes`, `log_backups`.
- `rag.observe`: `init_observability(settings=None, service_name="rag", *, configure_logging=True) ->
  Observability` (idempotent), `get_observability()`, `shutdown_observability()`; `Observability(mode,
  service_name, resource, tracer, meter, metrics, records, span_exporter, span_processor, metric_reader, flush(),
  shutdown(), write_record())`; `write_business_record(record) -> bool`.
- `rag.observe.attributes`: every span/metric attribute name and metric name we set.
- `rag.observe.context`: `RequestContext`, `new_request_id()`, `bind_request(...)`, `current_request()`,
  `current_request_id()`, `current_trace_id()`.
- `rag.observe.spans`: `request_span(name, *, kb, request_id=None, caller=None, config_version=None,
  data_generation=None, attributes=None, classify=None) -> RequestScope` (`set_result`, `add_error_category`);
  `layer_span(step, *, attributes=None, classify=None)` (context manager and sync/async decorator);
  `set_error_category(category, span=None)`.
- `rag.observe.metrics.Metrics`: `record_request`, `record_step`, `record_ledger_row`.
- `rag.observe.records`: `RankedId`, `ThresholdDecision`, `BusinessRecord`, `business_record_from_result(...)`,
  `RotatingJsonlWriter(path, *, max_bytes, backups)`, `BusinessRecordWriter`.
- `rag.observe.logs`: `JsonLogFormatter`, `TraceContextFilter`, `setup_logging(settings, service_name)`.
- `rag.cost.add_record_listener(fn) -> remove` (called with each ledger row after it is written).
- `rag.config.paths.git_env()` (moved from the eval runner), `git_output(repo_root, *args)`.
- `rag.testing`: `memory_observability(**overrides)`, `metric_points(obs, name)`, `observe_settings(**overrides)`.
- `deploy/otel-collector.yaml`: smaller trace-file rotation, metrics pipeline to a Prometheus exporter;
  `docker-compose.yml`: its port `127.0.0.1:28889`, container log rotation for the Collector and Jaeger.

## Acceptance criteria
Unit tests in `tests/unit/test_observe_*.py` (in-memory exporters, no network).
- [x] AC1 Observability settings have explicit defaults and env overrides — `test_observe_setup.py::test_observe_settings_defaults_and_overrides`
- [x] AC2 `otlp` mode builds OTLP span and metric exporters for the configured protocol (gRPC: endpoint as is; HTTP: `/v1/traces`, `/v1/metrics`) with explicit timeouts and intervals; spans are batched and `flush()` exports spans and metrics — `test_observe_setup.py::test_otlp_exporters_follow_protocol`, `test_observe_setup.py::test_otlp_mode_batches_spans_and_exports_metrics_on_flush`
- [x] AC3 The resource carries service.name, service.version and the git commit — `test_observe_setup.py::test_resource_attributes`
- [x] AC4 Init is idempotent; shutdown flushes, uninstruments, removes the ledger hook and allows a new init — `test_observe_setup.py::test_init_is_idempotent_and_shutdown_resets`
- [x] AC5 `memory` mode records spans and metrics in memory; `off` mode records nothing and helpers still work — `test_observe_setup.py::test_memory_and_off_modes`
- [x] AC6 Layer spans nest under the request span (one trace); kb, request id, config version, data generation, step set with the central names — `test_observe_spans.py::test_layer_spans_nest_under_request_span`
- [x] AC7 The request id is generated or taken from the caller and propagates through nested calls, asyncio tasks and `copy_context` threads; none outside a request — `test_observe_spans.py::test_request_id_propagation`
- [x] AC8 An exception marks the span as error with `error.type` and the category (classifier or explicit) and is re-raised — `test_observe_spans.py::test_span_error_records_category`
- [x] AC9 `layer_span` decorates sync and async functions — `test_observe_spans.py::test_layer_span_decorator_sync_and_async`
- [x] AC10 With auto-instrumentation on, LlamaIndex embedding and LangChain/LangGraph chat spans (fakes) nest under our layer span; embedding vectors are redacted; off switch gives no framework spans; `otel_trace_content=false` hides inputs and outputs — `test_observe_spans.py::test_auto_instrumentation_nests_framework_spans`, `test_observe_spans.py::test_auto_instrumentation_switch_and_content`
- [x] AC11 Every `rag.*` attribute name in `src/rag/observe` is defined in `attributes.py` — `test_observe_spans.py::test_attribute_names_live_in_one_module`
- [x] AC12 Request count by kb/tool/outcome, error count by category, degraded and rerank_timeout counts, per-step latency histogram with explicit buckets — `test_observe_metrics.py::test_request_metrics_from_result`, `test_observe_metrics.py::test_step_latency_histogram`
- [x] AC13 Each ledger line increments the token (input/output) and CNY counters; a failing listener never breaks the ledger — `test_observe_metrics.py::test_ledger_rows_feed_token_and_cost_counters`, `tests/unit/test_cost.py::test_record_listeners_get_rows_and_never_break_the_ledger`
- [x] AC14 A business record has request_id, trace_id, kb, config version, data generation, per-route top-10 ids and scores, rerank scores, threshold decision, flags and timings, one JSON line per request; an empty path disables records — `test_observe_records.py::test_business_record_fields_from_result`, `test_observe_records.py::test_business_records_disabled_without_path`
- [x] AC15 Size-based rotation keeps N backups, also across writer objects; a write failure is logged, not raised — `test_observe_records.py::test_jsonl_rotation`, `test_observe_records.py::test_write_failure_does_not_raise`
- [x] AC16 Every log line is JSON with time, level, logger, message, service, trace_id, span_id, request_id (null outside a span/request) and the exception; setup is idempotent; files rotate by size — `test_observe_logs.py::test_json_log_lines_carry_trace_and_request_ids`, `test_observe_logs.py::test_log_rotation_and_idempotent_setup`
- [x] AC17 The Collector trace file is size-capped (total <= 100 MB), metrics go to a Prometheus exporter on a 127.0.0.1 port, container logs are rotated — `test_observe_collector_config.py::test_collector_limits_and_metrics_pipeline`
- [x] AC18 (`integration`) A request span sent by `init_observability` over gRPC and HTTP is found in Jaeger's query API with our attributes — `tests/integration/test_observe_jaeger.py::test_request_span_reaches_jaeger`

## Out of scope
Calling these helpers from search_docs/ask_docs and writing real business records (P0-09, P0-12); the
scheduled probe request, Prometheus, Grafana, Langfuse and alerts (P5); sampling (later, in the Collector);
shipping log files through the Collector (later); request ids in ledger lines.

Also (checked by running, not by a test): the new Collector config passes `otelcol-contrib validate` (image
0.162.0); a throwaway Collector on spare ports received metrics over both protocols and served
`rag_requests_total`, `rag_errors_total`, `rag_requests_degraded_total` and `rag_step_duration_seconds` with the
explicit buckets on its Prometheus endpoint, then was removed. The shared Collector was not restarted.

## Handoff (filled at the end)
- Files: `src/rag/observe/{__init__,attributes,context,spans,metrics,records,logs,instrument,setup}.py`, `src/rag/testing/{observe,__init__}.py`, `src/rag/config/{settings,paths}.py`, `src/rag/cost/{ledger,__init__}.py`, `src/rag/eval/retrieval/runner.py` (imports `git_env`), `deploy/otel-collector.yaml`, `docker-compose.yml`, `.gitignore`, `.env.example`, `README.md`, `tests/unit/test_observe_{setup,spans,metrics,records,logs,collector_config}.py`, `tests/unit/test_cost.py`, `tests/integration/test_observe_jaeger.py`, `docs/decisions.md`.
- Decisions (docs/decisions.md): OTLP gRPC by default, `http/protobuf` by setting (both tested); helpers use the tracer and meter of the active `Observability`, globals set once in `otlp` mode; every OpenInference content switch explicit, vectors always hidden, 512 attributes and 16 KB per value per span; names in `attributes.py` (`rag.*`, plus OTel `service.*`, `error.type`, `vcs.ref.head.revision`, `gen_ai.*`); metrics: outcome ok/error/cancelled, categories from result flags, classifier, exception attribute or `unclassified`, histogram in seconds with 19 explicit bounds, tokens and CNY via a ledger listener; Collector metrics pipeline to a Prometheus exporter (127.0.0.1:28889), trace file 4 x 20 MB for 3 days, container logs 3 x 10 MB for Collector and Jaeger; business records with a file-locked, per-line-open rotating writer (20 MB x 6); JSON logs per service (10 MB x 4), stderr WARNING and up, nothing on stdout.
- Open issues: the running Collector still has the P0-02 config until `docker compose up -d` is run from the main checkout after the merge (recreates Collector and Jaeger; until then OTLP metric exports get "unimplemented" errors, traces are fine). `tests/integration/test_otel_collector.py` (P0-02) fails when run from a worktree: it looks for `data/otel/traces.jsonl` under the worktree while the shared Collector mounts the main checkout's `data/otel` (the trace was there). Request ids are not in ledger lines. P0-09 must call `init_observability` at startup, wrap each tool in `request_span`, set the classifier for its error categories and write business records; P0-07/P0-08 provide rerank scores and the threshold decision.
- Review rounds: none yet (author handoff; reviewer to follow).
- Tests: 24 new unit tests; after the rebase on main (P0-06) 222 unit and 22 of 23 integration tests pass (the failing one is the P0-02 worktree path issue above); the 2 new integration tests (one per protocol) pass.
