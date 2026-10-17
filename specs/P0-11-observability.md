# P0-11 Observability building blocks

Status: in progress
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
  service_name, resource, tracer, meter, metrics, records, span_exporter, metric_reader, flush(), shutdown())`.
- `rag.observe.attributes`: every span/metric attribute name and metric name we set.
- `rag.observe.context`: `RequestContext`, `new_request_id()`, `bind_request(...)`, `current_request()`,
  `current_request_id()`, `current_trace_id()`.
- `rag.observe.spans`: `request_span(name, *, kb, request_id=None, caller=None, config_version=None,
  data_generation=None, attributes=None, classify=None) -> RequestScope` (`set_result`, `set_error_category`);
  `layer_span(step, *, attributes=None, classify=None)` (context manager and sync/async decorator);
  `set_error_category(category, span=None)`.
- `rag.observe.metrics.Metrics`: `record_request`, `record_step`, `record_ledger_row`.
- `rag.observe.records`: `RankedId`, `ThresholdDecision`, `BusinessRecord`, `business_record_from_result(...)`,
  `RotatingJsonlWriter(path, *, max_bytes, backups)`, `BusinessRecordWriter`.
- `rag.observe.logs`: `JsonLogFormatter`, `TraceContextFilter`, `setup_logging(settings, service_name)`.
- `rag.cost.add_record_listener(fn) -> remove` (called with each ledger row after it is written).
- `deploy/otel-collector.yaml`: smaller trace-file rotation, metrics pipeline to a Prometheus exporter;
  `docker-compose.yml`: its port `127.0.0.1:28889`, container log rotation for the Collector and Jaeger.

## Acceptance criteria
Unit tests in `tests/unit/test_observe_*.py` (in-memory exporters, no network).
- [ ] AC1 Observability settings have explicit defaults and env overrides — `test_observe_setup.py::test_observe_settings_defaults_and_overrides`
- [ ] AC2 `otlp` mode builds OTLP span and metric exporters for the configured protocol (gRPC: endpoint as is; HTTP: `/v1/traces`, `/v1/metrics`) with explicit timeouts and intervals — `test_observe_setup.py::test_otlp_exporters_follow_protocol`
- [ ] AC3 The resource carries service.name, service.version and the git commit — `test_observe_setup.py::test_resource_attributes`
- [ ] AC4 Init is idempotent; shutdown flushes, uninstruments, removes the ledger hook and allows a new init — `test_observe_setup.py::test_init_is_idempotent_and_shutdown_resets`
- [ ] AC5 `memory` mode records spans and metrics in memory; `off` mode records nothing and helpers still work — `test_observe_setup.py::test_memory_and_off_modes`
- [ ] AC6 Layer spans nest under the request span (one trace); kb, request id, config version, data generation, step set with the central names — `test_observe_spans.py::test_layer_spans_nest_under_request_span`
- [ ] AC7 The request id is generated or taken from the caller and propagates through nested calls, asyncio tasks and `copy_context` threads; none outside a request — `test_observe_spans.py::test_request_id_propagation`
- [ ] AC8 An exception marks the span as error with `error.type` and the category (classifier or explicit) and is re-raised — `test_observe_spans.py::test_span_error_records_category`
- [ ] AC9 `layer_span` decorates sync and async functions — `test_observe_spans.py::test_layer_span_decorator_sync_and_async`
- [ ] AC10 With auto-instrumentation on, LlamaIndex embedding and LangChain/LangGraph chat spans (fakes) nest under our layer span; embedding vectors are redacted; off switch gives no framework spans; `otel_trace_content=false` hides inputs and outputs — `test_observe_spans.py::test_auto_instrumentation_nests_framework_spans`, `test_observe_spans.py::test_auto_instrumentation_switch_and_content`
- [ ] AC11 Every `rag.*` attribute name in `src/rag/observe` is defined in `attributes.py` — `test_observe_spans.py::test_attribute_names_live_in_one_module`
- [ ] AC12 Request count by kb/tool/outcome, error count by category, degraded and rerank_timeout counts, per-step latency histogram with explicit buckets — `test_observe_metrics.py::test_request_metrics_from_result`, `test_observe_metrics.py::test_step_latency_histogram`
- [ ] AC13 Each ledger line increments the token (input/output) and CNY counters; a failing listener never breaks the ledger — `test_observe_metrics.py::test_ledger_rows_feed_token_and_cost_counters`, `test_observe_metrics.py::test_failing_listener_does_not_break_ledger`
- [ ] AC14 A business record has request_id, trace_id, kb, config version, data generation, per-route top-10 ids and scores, rerank scores, threshold decision, flags and timings, one JSON line per request — `test_observe_records.py::test_business_record_fields_from_result`
- [ ] AC15 Size-based rotation keeps N backups, also across writer objects; a write failure is logged, not raised — `test_observe_records.py::test_jsonl_rotation`, `test_observe_records.py::test_write_failure_does_not_raise`
- [ ] AC16 Every log line is JSON with time, level, logger, message, service, trace_id, span_id, request_id (null outside a span/request) and the exception; setup is idempotent; files rotate by size — `test_observe_logs.py::test_json_log_lines_carry_trace_and_request_ids`, `test_observe_logs.py::test_log_rotation_and_idempotent_setup`
- [ ] AC17 The Collector trace file is size-capped (total <= 100 MB), metrics go to a Prometheus exporter on a 127.0.0.1 port, container logs are rotated — `test_observe_collector_config.py::test_collector_limits_and_metrics_pipeline`
- [ ] AC18 (`integration`) A request span sent by `init_observability` over gRPC and HTTP is found in Jaeger's query API with our attributes — `tests/integration/test_observe_jaeger.py::test_request_span_reaches_jaeger`

## Out of scope
Calling these helpers from search_docs/ask_docs and writing real business records (P0-09, P0-12); the
scheduled probe request, Prometheus, Grafana, Langfuse and alerts (P5); sampling (later, in the Collector);
shipping log files through the Collector (later); request ids in ledger lines.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
