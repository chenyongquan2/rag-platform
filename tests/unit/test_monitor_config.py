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
        "rag_online_eval_score_mean",
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


def test_dashboard_is_valid_provisioned_json_with_about_ten_panels() -> None:
    dashboard = _dashboard()
    panels = dashboard["panels"]
    assert 8 <= len(panels) <= 12
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
    assert store["max_traces"] == 20000  # written out, not the built-in default


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
