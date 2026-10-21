"""Spec P5-02: the monitoring stack for real. Needs `docker compose --profile monitoring up -d`
(Prometheus, Grafana) next to the default services; each test skips with a hint when what it
needs is not reachable.

- every instrument the code creates arrives at the Collector's Prometheus endpoint under the name
  `rag.testing.monitor.prometheus_names` predicts (this is what the dashboard and the alert rules
  rely on), and Prometheus then holds it;
- Prometheus loaded every alert rule;
- `promtool` accepts the rules and passes the alert unit tests (in the Prometheus image);
- Grafana has the provisioned dashboard.

The series carry the service name `p5-02-it` and expire from the Collector after ten minutes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from rag.config import ServiceSettings
from rag.eval.online_sample import create_online_eval_gauges
from rag.monitor.probe import ProbeInstruments
from rag.observe import init_observability, shutdown_observability
from rag.observe.health import DependencyChecker, register_dependency_gauge
from rag.testing.monitor import exported_series

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
PROMETHEUS = "http://127.0.0.1:29090"
GRAFANA = "http://127.0.0.1:23000"
COLLECTOR_METRICS = "http://127.0.0.1:28889/metrics"
SERVICE = "p5-02-it"


def _get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return bytes(response.read())


def _reachable(url: str) -> bool:
    try:
        _get(url, timeout=3)
    except (urllib.error.URLError, OSError):
        return False
    return True


def _need(url: str, what: str) -> None:
    if not _reachable(url):
        pytest.skip(f"{what} is not reachable at {url}: docker compose --profile monitoring up -d")


def _wait_for(check, timeout_s: float, step_s: float = 3.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = check()
        if last:
            return last
        time.sleep(step_s)
    return last


def _record_every_instrument() -> None:
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        otel_mode="otlp",
        otel_auto_instrument=False,
        business_record_path="",
        log_dir="",
        otel_metric_export_interval_s=5,
    )
    shutdown_observability()
    obs = init_observability(settings, SERVICE, configure_logging=False)
    try:
        m = obs.metrics
        m.record_request(
            kb="it",
            tool="search_docs",
            outcome="error",
            error_categories=["degraded", "rerank_timeout"],
        )
        m.record_step("search_docs", 0.5, outcome="ok", kb="it")
        m.record_retrieval(kb="it", tool="search_docs", not_found=True, top_rerank_score=0.4)
        m.record_cache(
            kb="it", tool="search_docs", layer="l1", outcome="bypass", reason="breaker_open"
        )
        m.record_cache(
            kb="it", tool="search_docs", layer="l1", outcome="miss", not_stored_reason="degraded"
        )
        m.record_cache_shadow(kb="it", tool="search_docs", verdict="would_hit")
        m.record_feedback(useful=True)
        m.record_ledger_row(
            {
                "kind": "chat",
                "model": "it-model",
                "profile": "cheap",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cny": 0.001,
            }
        )
        register_dependency_gauge(obs.meter, DependencyChecker({"elasticsearch": lambda: None}))
        probe = ProbeInstruments(obs.meter)
        attrs = {"rag.probe.kind": "search"}
        probe.runs.add(1, {**attrs, "rag.probe.outcome": "ok"})
        probe.duration.record(0.3, attrs)
        probe.success.set(1, attrs)
        probe.last_duration.set(0.3, attrs)
        probe.last_success.set(time.time(), attrs)
        hit, mrr = create_online_eval_gauges(obs.meter)
        hit.set(0.75)
        mrr.set(0.5)
        obs.flush(10.0)
    finally:
        shutdown_observability()


def test_every_instrument_arrives_under_the_predicted_name_and_prometheus_holds_it() -> None:
    _need(COLLECTOR_METRICS, "the Collector's metrics endpoint")
    _need(PROMETHEUS + "/-/healthy", "Prometheus")
    _record_every_instrument()

    def seen() -> set[str]:
        text = _get(COLLECTOR_METRICS).decode("utf-8")
        names = {
            line.split("{", 1)[0].split(" ", 1)[0]
            for line in text.splitlines()
            if line and not line.startswith("#") and f'job="{SERVICE}"' in line
        }
        return names

    names = _wait_for(lambda: seen() if "rag_online_eval_judged_mrr" in seen() else None, 40)
    assert names, "the Collector never showed the test service's metrics"
    missing = exported_series() - names
    assert not missing, f"series the code should export but the Collector did not show: {missing}"

    def prometheus_has() -> bool:
        query = urllib.request.quote(f'rag_probe_success{{job="{SERVICE}"}}')
        data = json.loads(_get(f"{PROMETHEUS}/api/v1/query?query={query}"))
        return bool(data["data"]["result"])

    assert _wait_for(prometheus_has, 60), "Prometheus did not scrape the series within 60 s"


def test_prometheus_loaded_every_alert_rule() -> None:
    _need(PROMETHEUS + "/-/healthy", "Prometheus")
    data = json.loads(_get(PROMETHEUS + "/api/v1/rules"))
    loaded = {
        rule["name"]: rule["health"] for group in data["data"]["groups"] for rule in group["rules"]
    }
    wanted = {
        rule["alert"]
        for group in yaml.safe_load((REPO / "deploy/prometheus/rules.yml").read_text("utf-8"))[
            "groups"
        ]
        for rule in group["rules"]
    }
    assert set(loaded) == wanted
    assert set(loaded.values()) == {"ok"}


def test_promtool_checks_the_rules_and_runs_their_unit_tests() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    mount = f"{(REPO / 'deploy' / 'prometheus').as_posix()}:/etc/prometheus:ro"
    base = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "promtool",
        "-v",
        mount,
        "-w",
        "/etc/prometheus",
    ]
    image = "prom/prometheus:v2.54.1"
    env = {"MSYS_NO_PATHCONV": "1"}
    import os

    for args in (["check", "rules", "rules.yml"], ["test", "rules", "rules_test.yml"]):
        done = subprocess.run(
            [*base, image, *args],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, **env},
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr


def test_grafana_has_the_provisioned_dashboard() -> None:
    _need(GRAFANA + "/api/health", "Grafana")
    dashboard = json.loads(_get(GRAFANA + "/api/dashboards/uid/rag-overview"))
    titles = {p["title"] for p in dashboard["dashboard"]["panels"]}
    assert "Request rate" in titles and len(titles) >= 8
    datasources = json.loads(_get(GRAFANA + "/api/datasources"))
    assert [d["uid"] for d in datasources] == ["prometheus"]
