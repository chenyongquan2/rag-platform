"""Spec P0-11: the Collector keeps its trace file small and turns metrics into Prometheus format."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_TRACE_FILES_MB = 100


def _load(path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))


def test_collector_limits_and_metrics_pipeline() -> None:
    config = _load("deploy/otel-collector.yaml")
    compose = _load("docker-compose.yml")

    # The trace file: current file plus backups stay under the cap.
    rotation = config["exporters"]["file/traces"]["rotation"]
    assert rotation["max_megabytes"] * (rotation["max_backups"] + 1) <= MAX_TRACE_FILES_MB
    assert rotation["max_days"] >= 1

    pipelines = config["service"]["pipelines"]
    assert set(pipelines["traces"]["exporters"]) == {"otlp/jaeger", "file/traces"}
    metrics = pipelines["metrics"]
    assert metrics["receivers"] == ["otlp"]
    assert metrics["processors"][0] == "memory_limiter"
    assert metrics["exporters"] == ["prometheus"]
    prometheus = config["exporters"]["prometheus"]
    host, _, port = prometheus["endpoint"].rpartition(":")
    assert host == "0.0.0.0"

    collector = compose["services"]["otel-collector"]
    # Host port = container port + 20000, bound to 127.0.0.1 only (docker-compose.yml header).
    assert f"127.0.0.1:{int(port) + 20000}:{port}" in collector["ports"]

    # Container stdout logs live on the Docker disk too: rotated, a few MB at most.
    for name in ("otel-collector", "jaeger"):
        logging = compose["services"][name]["logging"]
        assert logging["driver"] == "json-file"
        size_mb = int(str(logging["options"]["max-size"]).rstrip("m"))
        assert size_mb * int(logging["options"]["max-file"]) <= 50
