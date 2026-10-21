"""Spec P5-02 AC7, AC9: the dashboard, the alert rules and the compose file only name metrics and
ports the code and the repository really have."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag.config import ServiceSettings
from rag.testing.monitor import exported_series, prometheus_names, promql_metrics

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
DASHBOARD = DEPLOY / "grafana" / "dashboards" / "rag-overview.json"
RULES = DEPLOY / "prometheus" / "rules.yml"
RULES_TEST = DEPLOY / "prometheus" / "rules_test.yml"
PROMETHEUS = DEPLOY / "prometheus" / "prometheus.yml"
COMPOSE = REPO / "docker-compose.yml"


def test_name_translation_matches_series_seen_in_the_collector() -> None:
    """These names were read from the Collector's /metrics while this spec was built."""
    assert prometheus_names("counter", "rag.requests", "{request}") == {"rag_requests_total"}
    assert prometheus_names("histogram", "rag.step.duration", "s") == {
        "rag_step_duration_seconds_bucket",
        "rag_step_duration_seconds_sum",
        "rag_step_duration_seconds_count",
    }
    assert "rag_rerank_top_score_bucket" in prometheus_names(
        "histogram", "rag.rerank.top_score", "1"
    )
    assert prometheus_names("gauge", "rag.dependency.up", "") == {"rag_dependency_up"}


def test_the_code_exports_what_the_alerts_and_the_dashboard_rely_on() -> None:
    series = exported_series()
    for name in (
        "rag_requests_total",
        "rag_errors_total",
        "rag_requests_degraded_total",
        "rag_requests_rerank_timeout_total",
        "rag_step_duration_seconds_bucket",
        "rag_cache_requests_total",
        "rag_model_cost_total",
        "rag_model_tokens_total",
        "rag_model_calls_total",
        "rag_feedback_total",
        "rag_dependency_up",
        "rag_probe_success",
        "rag_probe_last_success_timestamp_seconds",
        "rag_probe_last_duration_seconds",
        "rag_probe_consecutive_failures",
        "rag_probe_runs_total",
        "rag_probe_duration_seconds_bucket",
        "rag_online_eval_judged_hit_rate",
        "rag_online_eval_judged_mrr",
    ):
        assert name in series, name


def test_promql_metric_extraction_skips_labels() -> None:
    expr = (
        'sum by (rag_tool) (rate(rag_requests_total{rag_tool=~"a|b", rag_outcome="error"}[5m]))'
        " / rag_probe_success"
    )
    assert promql_metrics(expr) == {"rag_requests_total", "rag_probe_success"}


def _rules() -> list[dict[str, Any]]:
    data = yaml.safe_load(RULES.read_text("utf-8"))
    return [rule for group in data["groups"] for rule in group["rules"]]


def _dashboard() -> dict[str, Any]:
    return json.loads(DASHBOARD.read_text("utf-8"))


def test_every_metric_in_the_alert_rules_is_exported_by_the_code() -> None:
    series = exported_series()
    for rule in _rules():
        used = promql_metrics(rule["expr"])
        assert used, rule["alert"]
        assert used <= series, (rule["alert"], used - series)


def test_every_metric_in_the_dashboard_is_exported_by_the_code() -> None:
    series = exported_series()
    expressions = [
        target["expr"] for panel in _dashboard()["panels"] for target in panel["targets"]
    ]
    assert expressions
    for expr in expressions:
        assert promql_metrics(expr) <= series, (expr, promql_metrics(expr) - series)


def test_alert_rules_have_the_alerts_the_design_asks_for() -> None:
    rules = {rule["alert"]: rule for rule in _rules()}
    assert {
        "SearchProbeFailing",
        "SearchProbeNoRecentSuccess",
        "ProbeSlow",
        "DependencyDown",
        "HighErrorRate",
        "SearchLatencyP95High",
        "AskLatencyP95High",
        "RerankTimeoutRateHigh",
        "DegradedRateHigh",
        "CacheBreakerOpen",
        "DailyCostOverBudget",
        "TokenBurnHigh",
        "NotFoundRateHigh",
    } <= set(rules)
    for name, rule in rules.items():
        assert rule["labels"]["severity"] in {"critical", "warning"}, name
        assert rule["annotations"]["summary"] and rule["annotations"]["description"], name
        assert re.fullmatch(r"\d+(s|m|h)", rule["for"]), name  # a `for:` on every alert


def test_alert_thresholds_follow_the_measured_numbers() -> None:
    """The comments in rules.yml say where each number comes from; these are the numbers."""
    rules = {rule["alert"]: rule["expr"] for rule in _rules()}
    assert "> 8" in rules["SearchLatencyP95High"]  # measured P95 4 to 6 s
    assert "> 30" in rules["AskLatencyP95High"]  # measured about 21 s, default deadline 30 s
    assert "> 0.10" in rules["HighErrorRate"]
    assert "> 2" in rules["DailyCostOverBudget"]
    assert "> 180" in rules["SearchProbeNoRecentSuccess"]
    assert "breaker_open" in rules["CacheBreakerOpen"]
    # two probe failures in a row, not one (design s15-1; review P5-02 round 1, issue 2)
    assert "rag_probe_consecutive_failures" in rules["SearchProbeFailing"]
    assert ">= 2" in rules["SearchProbeFailing"]


def test_every_tested_alert_exists_and_the_fault_alerts_are_tested() -> None:
    tests = yaml.safe_load(RULES_TEST.read_text("utf-8"))["tests"]
    tested = {case["alertname"] for test in tests for case in test.get("alert_rule_test", [])}
    names = {rule["alert"] for rule in _rules()}
    assert tested <= names
    assert {"SearchProbeFailing", "DependencyDown", "HighErrorRate", "ProbeSlow"} <= tested


def test_dashboard_is_valid_provisioned_json_with_about_a_dozen_panels() -> None:
    # 8 to 12 before review round 1; issue 11 added the not-found and rerank-score panels and
    # the mixed-unit panel was split in three, so the limit is now 16.
    dashboard = _dashboard()
    panels = dashboard["panels"]
    assert 8 <= len(panels) <= 16
    assert dashboard["uid"] == "rag-overview" and dashboard["refresh"]
    assert len({p["id"] for p in panels}) == len(panels)
    datasource = yaml.safe_load(
        (DEPLOY / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text("utf-8")
    )["datasources"][0]
    for panel in panels:
        assert panel["title"] and panel["description"]
        assert panel["datasource"]["uid"] == datasource["uid"]
        grid = panel["gridPos"]
        assert grid["x"] >= 0 and grid["x"] + grid["w"] <= 24 and grid["h"] >= 1
        assert panel["targets"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == datasource["uid"] and target["expr"].strip()
    provider = yaml.safe_load(
        (DEPLOY / "grafana" / "provisioning" / "dashboards" / "dashboards.yml").read_text("utf-8")
    )["providers"][0]
    assert provider["options"]["path"] == "/etc/grafana/dashboards"


def test_dashboard_shows_the_retrieval_quality_signals() -> None:
    """Review P5-02 round 1, issue 11: the not-found counter and the best rerank score (design
    s15-1, "indirect signals of retrieval quality") have panels."""
    used: set[str] = set()
    for panel in _dashboard()["panels"]:
        for target in panel["targets"]:
            used |= promql_metrics(target["expr"])
    assert "rag_requests_not_found_total" in used
    assert "rag_rerank_top_score_bucket" in used
    assert {"rag_online_eval_judged_hit_rate", "rag_online_eval_judged_mrr"} <= used


USER_SERIES = (
    "rag_requests_total",
    "rag_errors_total",
    "rag_step_duration_seconds_bucket",
    "rag_cache_requests_total",
    "rag_requests_rerank_timeout_total",
    "rag_requests_degraded_total",
    "rag_requests_not_found_total",
    "rag_rerank_top_score_bucket",
)


def _selectors(expr: str, name: str) -> list[str]:
    return re.findall(rf"(?<![a-z_]){name}({{[^}}]*}})?", expr)


def test_user_facing_panels_and_quality_alerts_leave_the_probe_out() -> None:
    """Review P5-02 round 1, issue 4: every selector of a request series in the dashboard and in
    the quality alerts excludes `rag_request_probe="1"`."""
    expressions = [t["expr"] for p in _dashboard()["panels"] for t in p["targets"]]
    data = yaml.safe_load(RULES.read_text("utf-8"))
    quality = next(g for g in data["groups"] if g["name"] == "rag-quality")
    expressions += [rule["expr"] for rule in quality["rules"]]
    checked = 0
    for expr in expressions:
        for name in USER_SERIES:
            for labels in _selectors(expr, name):
                assert 'rag_request_probe!="1"' in labels, (name, expr)
                checked += 1
    assert checked >= 20


def test_every_panel_has_one_unit_and_the_cost_panel_says_cny() -> None:
    panels = {p["title"]: p for p in _dashboard()["panels"]}
    assert "CNY" in panels["Model cost per hour (CNY)"]["fieldConfig"]["defaults"]["unit"]
    for panel in panels.values():
        # A panel draws one quantity: seconds, verdict counts and a 0-to-1 share are not mixed.
        metrics = set()
        for target in panel["targets"]:
            metrics |= promql_metrics(target["expr"])
        mixed = any(m.endswith("_seconds") for m in metrics) and any(
            m.endswith("_total") for m in metrics
        )
        assert not mixed, panel["title"]


# ---- Prometheus and compose ----------------------------------------------------------------


def test_prometheus_scrapes_the_collector_and_sends_alerts_to_the_receiver() -> None:
    config = yaml.safe_load(PROMETHEUS.read_text("utf-8"))
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["otel-collector"]["static_configs"][0]["targets"] == ["otel-collector:8889"]
    assert jobs["otel-collector"]["honor_labels"] is True
    assert config["rule_files"] == ["/etc/prometheus/rules.yml"]
    port = ServiceSettings(_env_file=None).alert_receiver_port  # type: ignore[call-arg]
    targets = config["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
    assert targets == [f"host.docker.internal:{port}"]
    assert config["global"]["scrape_interval"] == "15s"


def _compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text("utf-8"))["services"]


def _memory_mb(value: str) -> int:
    match = re.fullmatch(r"(\d+)([mg])", str(value))
    assert match, value
    return int(match.group(1)) * (1024 if match.group(2) == "g" else 1)


def test_default_services_are_the_four_from_before() -> None:
    services = _compose()
    default = {name for name, svc in services.items() if not svc.get("profiles")}
    assert default == {"elasticsearch", "redis-cache", "otel-collector", "jaeger"}


def test_new_services_sit_behind_a_profile_with_limits_local_ports_and_health_checks() -> None:
    services = _compose()
    monitoring = {n for n, s in services.items() if s.get("profiles") == ["monitoring"]}
    langfuse = {n for n, s in services.items() if s.get("profiles") == ["langfuse"]}
    assert monitoring == {"prometheus", "grafana"}
    assert {
        "langfuse-web",
        "langfuse-worker",
        "langfuse-postgres",
        "langfuse-redis",
        "langfuse-clickhouse",
        "langfuse-minio",
    } == langfuse
    for name in monitoring | langfuse:
        service = services[name]
        assert _memory_mb(service["mem_limit"]) <= 2048, name
        assert "healthcheck" in service or name == "langfuse-worker", name
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)  # local only
    assert services["prometheus"]["ports"] == ["127.0.0.1:29090:9090"]
    assert services["grafana"]["ports"] == ["127.0.0.1:23000:3000"]
    assert services["langfuse-web"]["ports"] == ["127.0.0.1:23001:3000"]
    # the two services of the monitoring profile together stay under a gigabyte
    assert sum(_memory_mb(services[n]["mem_limit"]) for n in monitoring) <= 1024


def test_the_collector_and_jaeger_read_their_config_files_and_have_limits() -> None:
    services = _compose()
    collector = services["otel-collector"]
    assert any("COLLECTOR_EXTRA:-none" in part for part in collector["command"])
    assert (DEPLOY / "otel" / "none.yaml").exists() and (DEPLOY / "otel" / "langfuse.yaml").exists()
    assert "mem_limit" in collector and "mem_limit" in services["jaeger"]
    assert "./deploy/jaeger.yaml:/etc/jaeger/config.yaml:ro" in services["jaeger"]["volumes"]
    jaeger = yaml.safe_load((DEPLOY / "jaeger.yaml").read_text("utf-8"))
    store = jaeger["extensions"]["jaeger_storage"]["backends"]["some_store"]["memory"]
    # Measured: a search_docs trace takes about 142 KB in Jaeger's memory, an ask_docs trace 347 to
    # 589 KB; 3,000 traces of a mix fit the 768 MB limit (review P5-02 round 1, issue 10; it was
    # 20,000, which could never fit).
    assert store["max_traces"] == 3000
    assert _memory_mb(services["jaeger"]["mem_limit"]) == 768
    for name in ("jaeger", "otel-collector"):
        assert services[name]["restart"] == "unless-stopped", name


def test_the_collector_keeps_trace_files_for_a_stated_time_and_size() -> None:
    config = yaml.safe_load((DEPLOY / "otel-collector.yaml").read_text("utf-8"))
    rotation = config["exporters"]["file/traces"]["rotation"]
    assert rotation == {"max_megabytes": 20, "max_days": 3, "max_backups": 3}
    # the safety net in rag.monitor.prune uses the same days and a size above 4 x 20 MB
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.trace_retention_days == rotation["max_days"]
    assert settings.trace_retention_max_mb >= rotation["max_megabytes"] * (
        rotation["max_backups"] + 1
    )


@pytest.mark.parametrize("overlay", ["none", "langfuse"])
def test_collector_overlays_are_yaml(overlay: str) -> None:
    loaded = yaml.safe_load((DEPLOY / "otel" / f"{overlay}.yaml").read_text("utf-8"))
    assert isinstance(loaded, dict)
    if overlay == "langfuse":
        # a pipeline of its own, so a stopped Langfuse cannot hold back the other back ends
        assert loaded["service"]["pipelines"]["traces/langfuse"]["exporters"] == [
            "otlphttp/langfuse"
        ]
        assert loaded["exporters"]["otlphttp/langfuse"]["traces_endpoint"].endswith(
            "/api/public/otel/v1/traces"
        )


def test_env_example_names_every_monitoring_setting() -> None:
    """Review P5-02 round 1, issue 12: every setting this spec added, and the names compose
    reads, are listed (commented, without values) in .env.example."""
    text = (REPO / ".env.example").read_text("utf-8")
    prefixes = ("probe_", "online_eval_", "trace_retention_", "alert_", "dependency_check_")
    names = [name.upper() for name in ServiceSettings.model_fields if name.startswith(prefixes)]
    assert len(names) >= 30
    compose_names = ["GRAFANA_ADMIN_PASSWORD", "COLLECTOR_EXTRA", "LANGFUSE_BASIC_AUTH"]
    from rag.monitor.langfuse_env import KEYS

    for name in [*names, *compose_names, *KEYS]:
        assert re.search(rf"^# {name}=", text, re.MULTILINE), name
    # no secret values: the secret names are listed empty
    for name in ("ALERT_WEBHOOK_URL", "GRAFANA_ADMIN_PASSWORD", *KEYS):
        line = re.search(rf"^# {name}=(\S*)", text, re.MULTILINE)
        assert line is not None and line.group(1) == "", name


def test_readme_never_suggests_down_or_an_unnamed_profile_stop() -> None:
    """Review P5-02 round 1, issue 1: `docker compose down` removes Elasticsearch and Redis of a
    shared setup (and `-v` their data); a `--profile X stop` without service names stops them
    too."""
    for path in (REPO / "README.md", COMPOSE, DEPLOY / "otel" / "langfuse.yaml"):
        for line in path.read_text("utf-8").splitlines():
            if "docker compose" not in line:
                continue
            if "never" in line.lower() or "not `down`" in line or "(never `down`" in line:
                continue  # a warning that names the command
            assert not re.search(r"docker compose(\s+--profile\s+\S+)?\s+down\b", line), line
            stop = re.search(r"docker compose\s+--profile\s+\S+\s+(stop|rm -sf)\s*(\S*)", line)
            if stop:
                assert stop.group(2) and not stop.group(2).startswith(("`", "#")), line


def test_the_gateway_breaker_alert_reads_the_breaker_counter_and_is_tested() -> None:
    """Spec P8-01 AC6: the breaker's counter is exported and its alert is unit-tested."""
    assert "rag_model_breaker_total" in exported_series()
    rules = {rule["alert"]: rule for rule in _rules()}
    expr = rules["ModelGatewayBreakerOpen"]["expr"]
    assert "rag_model_breaker_total" in expr and "opened" in expr and "rejected" in expr
    tests = yaml.safe_load(RULES_TEST.read_text("utf-8"))["tests"]
    tested = {case["alertname"] for test in tests for case in test.get("alert_rule_test", [])}
    assert "ModelGatewayBreakerOpen" in tested
