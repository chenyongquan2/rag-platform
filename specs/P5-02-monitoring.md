# P5-02 Monitoring: metrics backend, alerts, probe, feedback tool, online sampling, retention

Status: in progress
Phase: P5
Depends on: P0-11 (observe, Collector), P3-04 (`/v1/feedback`), P5-01 (cache metrics)
Design sections: s15-1, s15-2, s15-3, s16-1, s16-5, s14-5
Calls real models: yes, at most 0.1 CNY (one real online-sampling run of 5 records; the probe's ask check is off by default)

## Goal
The metrics the app already exports are scraped by Prometheus, drawn by a provisioned Grafana dashboard and checked by alert rules. A
local receiver writes every alert to `data/observe/alerts.jsonl` (and to `ALERT_WEBHOOK_URL` when set). A probe sends one fixed
`search_docs` request per interval and exports its result as metrics. The `feedback` MCP tool exists. A job samples recent
requests and judges them. Old trace files are pruned. Everything new in Docker is behind the compose profile `monitoring`
(and `langfuse`), so the default `docker compose up -d` is unchanged.

## Interface
- `docker-compose.yml`: profile `monitoring` = `prometheus` (127.0.0.1:29090, 512 MB) and `grafana` (127.0.0.1:23000, 384 MB); profile
  `langfuse` = Langfuse v3 stack (web 23001) with memory limits. `deploy/prometheus/{prometheus.yml,rules.yml}`,
  `deploy/grafana/{provisioning/...,dashboards/rag-overview.json}`, `deploy/otel/{extra-none,extra-langfuse}.yaml`.
- `rag.observe.health`: `DependencyChecker(settings)` and `register_dependency_gauge(meter, checker)`: gauge `rag.dependency.up{dependency}`.
- `rag.observe.metrics.Metrics.record_feedback(kb, useful)`: counter `rag.feedback{verdict=useful|not_useful}`.
- MCP tool `feedback(request_id, useful, comment?)` -> `FeedbackResponse`; same validation as `POST /v1/feedback` (shared `SearchService.feedback`).
- `rag.monitor.probe`: `ProbeConfig`, `Prober(config, client, clock)`, `run_forever`; CLI `python -m rag.monitor probe`.
  Metrics: `rag.probe.runs{kind,outcome}`, `rag.probe.duration{kind}`, gauges `rag.probe.success`, `rag.probe.sane`,
  `rag.probe.last_success_timestamp` (per kind `search`/`ask`). Ask probe off by default; daily cap `probe_ask_daily_cap` kept in a state file.
- `rag.monitor.alert_receiver`: Alertmanager-v2-compatible `POST /api/v2/alerts` receiver; CLI `python -m rag.monitor receiver`.
- `rag.monitor.prune`: `prune_files(directory, pattern, *, max_age_s, max_total_bytes, now)`; CLI `python -m rag.monitor prune`.
- `rag.eval.online_sample`: `sample_records`, `run_online_sample`, CLI `python -m rag.eval.online_sample`; gauge `rag.online_eval.score_mean`.
- Settings: `probe_*`, `alert_*`, `online_eval_*` keys in `ServiceSettings`.

## Acceptance criteria
- [ ] AC1 The dependency gauge is 1 when the checks pass and 0 when one fails, per dependency; a check never raises and is cached for its TTL — `tests/unit/test_observe_health.py::test_gauge_reports_each_dependency`, `::test_check_failure_is_zero_and_cached`
- [ ] AC2 `feedback` is an MCP tool with the HTTP endpoint's validation, writes one business record keyed by the rated request and counts `useful` / `not_useful` — `tests/unit/test_serve_mcp.py::test_feedback_tool_*`, `tests/unit/test_serve_service.py::test_feedback_counts_verdicts`
- [ ] AC3 The probe reports success, latency, sanity (expected top-1 source present, not degraded) and last-success time; a failing, slow, degraded or wrong top-1 answer is not a success — `tests/unit/test_monitor_probe.py`
- [ ] AC4 The ask probe is off by default, runs at its own interval and never more than its daily cap, also after a restart — `tests/unit/test_monitor_probe.py::test_ask_probe_*`
- [ ] AC5 The receiver writes a line when an alert starts firing and when it resolves, not at every repeat; posts to the webhook from the environment and never logs its URL — `tests/unit/test_monitor_alerts.py`
- [ ] AC6 Pruning removes files older than the age and the oldest files above the size cap, and nothing else — `tests/unit/test_monitor_prune.py`
- [ ] AC7 Every metric in the dashboard and in the alert rules is one the code defines (OTel name translated to Prometheus form); the dashboard JSON parses and has 8 to 12 panels; the rule file has the expected alerts with `for`, thresholds and annotations — `tests/unit/test_monitor_config.py`
- [ ] AC8 The online sample takes at most N random records, judges the passages of each with the judge, writes a summary with a Wilson interval and sets the gauge; the judge is called at most N times; a failed judge call is counted, not scored — `tests/unit/test_online_sample.py`
- [ ] AC9 Compose: the monitoring services have memory limits, 127.0.0.1 ports, healthchecks and the profile; the default services are unchanged — `tests/unit/test_monitor_config.py::test_compose_*`
- [ ] AC10 (integration, manual in this run) A fault produces an alert within 3 minutes; recorded in `docs/reports/P5-report.md` section 2.

## Out of scope
Alertmanager grouping/silencing and chat-bot formats (the webhook takes one generic JSON body); Langfuse prompt management; per-caller dashboards;
production sizing. Daily scheduling of the online sample (the job exists, nothing starts it).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
