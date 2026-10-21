# P5-02 Monitoring: metrics backend, alerts, probe, feedback tool, online sampling, retention

Status: review
Phase: P5
Depends on: P0-11 (observe, Collector), P3-04 (`/v1/feedback`), P5-01 (cache metrics)
Design sections: s15-1, s15-2, s15-3, s16-1, s16-5, s14-5
Calls real models: yes, at most 0.1 CNY was allowed (the real run cost about 0.15 CNY: see the report 2.11; review round 1 made no real call)

## Goal
The metrics the app already exports are scraped by Prometheus, drawn by a provisioned Grafana dashboard and checked by alert rules. A
local receiver writes every alert to `data/observe/alerts.jsonl` (and to `ALERT_WEBHOOK_URL` when set). A probe sends one fixed
`search_docs` request per interval and exports its result as metrics. The `feedback` MCP tool exists. A job samples recent
requests and judges them. Old trace files are pruned. Everything new in Docker is behind the compose profile `monitoring`
(and `langfuse`), so the default `docker compose up -d` is unchanged.

## Interface
- `docker-compose.yml`: profile `monitoring` = `prometheus` (127.0.0.1:29090, 512 MB) and `grafana` (127.0.0.1:23000, 384 MB); profile
  `langfuse` = Langfuse v3 stack (web 23001, six services, limits 4,928 MiB). The Collector reads `deploy/otel-collector.yaml` plus one overlay of
  `deploy/otel/` (`COLLECTOR_EXTRA`, default `none`; `langfuse`); Jaeger reads `deploy/jaeger.yaml` (`max_traces` 3000, from measured trace sizes). Memory limits and
  `restart: unless-stopped` on Collector and Jaeger. `deploy/prometheus/{prometheus.yml,rules.yml,rules_test.yml}`, `deploy/grafana/{provisioning/...,dashboards/rag-overview.json}`.
- `rag.observe.health`: `DependencyChecker(checks, ttl_s)`, `register_dependency_gauge(meter, checker)`: gauge `rag.dependency.up{rag.dependency}`.
- `rag.observe.metrics.Metrics.record_feedback(useful)`: counter `rag.feedback{rag.feedback.verdict=useful|not_useful}` (one counter with a label,
  not two counters). `Metrics.record_cache(..., reason)`: bypass and error points of `rag.cache.requests` carry `rag.cache.reason`.
- MCP tool `feedback(request_id, useful, comment?)` -> `FeedbackResponse`; same validation as `POST /v1/feedback` (the shared `SearchService.feedback`).
- `rag.monitor.probe`: `ProbeConfig`, `ProbeInstruments`, `Prober(config, client, clock)`, `run_forever`. Metrics: `rag.probe.runs{kind,outcome}`,
  `rag.probe.duration{kind}`, gauges `rag.probe.success`, `rag.probe.last_duration`, `rag.probe.last_success_timestamp`, `rag.probe.consecutive_failures`
  (kind `search`/`ask`; `SearchProbeFailing` fires at two failures in a row, design s15-1).
  Outcome `ok` = HTTP 200, not degraded, not rerank-timeout, found, and the expected page is among the returned passages (the order of tied
  passages is not stable: see report 2.5; the design's "first passage" check was given up). Every search run asks a new question (a made-up
  word is appended) and every request has `X-Request-Id: probe-<hex>` (`rag.observe.context.is_probe_request`). The service
  does not read or write L1, writes no L2 shadow row and no L0 query embedding for such a request, and every metric point
  recorded inside it carries `rag.request.probe=1`; request panels and quality alerts select `rag_request_probe!="1"`. The ask check is off by default and capped per UTC day in a state file.
- `rag.monitor.alert_receiver`: Alertmanager-v2-compatible `POST /api/v2/alerts` receiver (firing state rebuilt from the alert
  file at start; webhook posted from a background thread, 3 tries with 1 s and 2 s backoff, `flush()`); `rag.monitor.prune.prune_files`; `rag.monitor.langfuse_env.ensure_env`;
  CLI `python -m rag.monitor probe|receiver|prune|langfuse-env`.
- `rag.eval.online_sample`: `load_candidates`, `sample_candidates`, `relevance_messages`, `run_online_sample`, CLI
  `python -m rag.eval.online_sample` (records a seed when `--seed` is unset); gauges `rag.online_eval.judged_hit_rate`,
  `rag.online_eval.judged_mrr`. One cheap-judge call per request (`PassageRelevanceJudgement`: one relevant/not verdict per
  numbered passage, first `online_eval_k`), as design s15-1 and the plan row ask; no reference answer needed. Judged hit@k
  with a Wilson interval, judged MRR with a seeded bootstrap interval. `ask_docs` citation precision waits on P1-06.
- Settings: `dependency_check_*`, `probe_*`, `alert_*`, `trace_retention_*`, `online_eval_*` in `ServiceSettings`.

## Acceptance criteria
- [x] AC1 The dependency gauge is 1 when the checks pass and 0 when one fails, per dependency; a check never raises and is cached for its TTL; the checks use the ES and Redis factories (credentials in the URL work) and never log a URL — `tests/unit/test_observe_health.py`
- [x] AC2 `feedback` is an MCP tool with the HTTP endpoint's validation, writes one business record keyed by the rated request and counts `useful` / `not_useful` — `tests/unit/test_feedback_tool.py`
- [x] AC3 The probe reports success, latency, last-success time and failures in a row; a failing, slow, degraded, wrong-page or not-found answer is not a success; each search run asks a new question; probe requests are marked — `tests/unit/test_monitor_probe.py`
- [x] AC3b The service skips the caches for probe requests and labels their metrics — `tests/unit/test_probe_traffic.py`
- [x] AC4 The ask probe is off by default, runs at its own interval and never more than its daily cap, also after a restart — `tests/unit/test_monitor_probe.py::test_ask_probe_*`, `::test_the_cap_survives_a_restart_and_resets_the_next_day`
- [x] AC5 The receiver writes a line when an alert starts firing and when it resolves, not at every repeat, also across a restart; posts to the webhook with bounded retries and never logs its URL — `tests/unit/test_monitor_alerts.py`
- [x] AC6 Pruning removes files older than the age and the oldest files above the size cap, and nothing else — `tests/unit/test_monitor_prune.py`
- [x] AC7 Every metric in the dashboard and in the alert rules is one the code defines (OTel name translated to the Prometheus name, checked against the real Collector); the dashboard has 8 to 16 panels incl. not-found and best rerank score; request panels and quality alerts leave the probe out; the rule file has the expected alerts with `for`, thresholds and annotations; promtool unit tests pass — `tests/unit/test_monitor_config.py`, `tests/integration/test_monitor_stack.py`
- [x] AC8 The online sample takes at most N (hard cap 50) random records without probe requests, judges each of the first k passages of each in one call, writes judged hit@k (Wilson) and judged MRR (bootstrap) and sets the gauges; a failed or incomplete judge answer is counted, not scored; the job stops at its own spending cap — `tests/unit/test_online_sample.py`
- [x] AC9 Compose: the new services have memory limits, 127.0.0.1 ports and health checks and sit behind a profile; the default services are the four of before; README never suggests `down` or an unnamed profile stop; `.env.example` names every setting — `tests/unit/test_monitor_config.py::test_new_services_*`, `::test_default_services_*`, `::test_readme_*`, `::test_env_example_*`
- [x] AC10 (manual, once) Two faults produced alerts in 147 s and 162 s — report 2.4. Not a repeatable test.

## Out of scope
Alertmanager grouping, pausing and chat-bot formats (the webhook takes one generic JSON body); sending feedback to Langfuse as a score (design s15-1 says P5; not done);
per-caller dashboards; production sizing; scheduling the online sample.

## Handoff (filled at the end)
- Files: new `src/rag/monitor/{probe,alert_receiver,prune,langfuse_env,cli,__main__}.py`, `src/rag/observe/health.py`, `src/rag/eval/online_sample.py`, `src/rag/testing/monitor.py`; changed `src/rag/serve/{mcp_server,service,cli}.py` (feedback tool, counter, dependency gauge), `src/rag/observe/{metrics,attributes,context}.py` (probe label), `src/rag/cache/layer.py` (reason label, probe bypass), `src/rag/embed/cache.py` (no L0 write for probes), `src/rag/config/settings.py`; `docker-compose.yml`, `deploy/**`, `.env.example`, `README.md`, `docs/decisions.md`, `docs/reports/P5-report.md` (section 2); tests `tests/unit/test_{feedback_tool,observe_health,monitor_*,online_sample,probe_traffic}.py`, `tests/integration/test_monitor_stack.py`.
- Decisions (docs/decisions.md, P5-02 items 1 to 13): local receiver instead of Alertmanager; thresholds with their source; probe asks a new question each run, checks the page is in the list, alerts after two failures in a row; probe requests skip the caches and carry a label; Langfuse v3 in a profile; one feedback counter with a label; online sampling judges each passage (judged hit@k, MRR); Jaeger 3000 traces; Grafana anonymous viewer.
- **Post-merge commands (required):** Collector, Jaeger, Prometheus and Grafana were started from this worktree: they mount its `deploy/` and the Collector writes traces into its `data/otel/`. From the main checkout `C:/workspace/rag-platform`: `docker compose up -d otel-collector jaeger`, then `docker compose --profile monitoring up -d prometheus grafana` (or leave them stopped). This also applies `max_traces` 3000 and the restart policy. Never `docker compose down`, never an unnamed `--profile X stop`; Elasticsearch and Redis are not touched.
- Open issues: `es-data` as an external volume (and Langfuse in its own compose project) not applied: Docker Desktop was not running during review round 1, so `docker compose --dry-run up -d` could not confirm that Elasticsearch would not be recreated; the change is written in decisions item 13. promtool rule tests and the monitor-stack integration test were not run in round 1 for the same reason. The new relevance judge has no real-model run and no agreement check with human labels; alert time measured once with the old probe rule; webhook format is generic; Langfuse only tried with a fake model; real model spend 0.15 CNY, over the 0.1 CNY allowed (round 1 spent 0).
- Incident: `docker compose --profile langfuse down -v` removed every container of the project and the Elasticsearch volume (report 2.12). Both real indices were rebuilt from the local caches at 0 CNY (26,372 and 21,913 chunks). Lost too: the other indices of the shared Elasticsearch (P2-01 rollback versions, P2-03 ablation indices, `cache_semantic_*` if used), Jaeger's in-memory traces, the Redis caches. Not lost: the P4-02 experiment volume, trace files, the ledger, the embedding caches.
- Review rounds: 1 (CHANGES, 12 issues and nits; all fixed except issue 9, documented). TDD: every round-1 fix was written test first and run red.
