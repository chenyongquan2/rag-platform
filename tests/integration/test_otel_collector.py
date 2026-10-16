from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pytest
import yaml
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from rag.config import ServiceSettings
from rag.config.settings import mask_url

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _host_ports(service: dict) -> dict[int, tuple[str, int]]:
    """`{container port: (host ip, host port)}` of one compose service."""
    out: dict[int, tuple[str, int]] = {}
    for spec in service.get("ports", []):
        host_ip, host_port, container_port = str(spec).split(":")
        out[int(container_port)] = (host_ip, int(host_port))
    return out


class ComposeOtel:
    """Collector and Jaeger addresses and the trace file, read from the compose and Collector
    config files, so this test follows them instead of repeating their ports and paths."""

    def __init__(self) -> None:
        compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text("utf-8"))
        collector = compose["services"]["otel-collector"]
        ports = _host_ports(collector)
        jaeger_ip, jaeger_port = _host_ports(compose["services"]["jaeger"])[16686]
        health_ip, health_port = ports[13133]
        self.otlp_grpc = ports[4317]
        self.health_url = f"http://{health_ip}:{health_port}/"
        self.jaeger_url = f"http://{jaeger_ip}:{jaeger_port}"
        config = yaml.safe_load((REPO_ROOT / "deploy" / "otel-collector.yaml").read_text("utf-8"))
        path = PurePosixPath(config["exporters"]["file/traces"]["path"])
        for volume in collector["volumes"]:
            host_dir, container_dir = str(volume).split(":")[:2]
            if path.is_relative_to(container_dir):
                self.trace_file = REPO_ROOT / host_dir / path.relative_to(container_dir)
                break
        else:
            raise AssertionError(f"{path} is not on a host volume of the collector")


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def _wait_for(check: Callable[[], bool], timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.5)
    return False


def test_collector_exports_to_jaeger_and_file(service_settings: ServiceSettings) -> None:
    otel = ComposeOtel()
    endpoint = urlsplit(service_settings.otlp_endpoint)
    if (endpoint.hostname, endpoint.port) != otel.otlp_grpc:
        pytest.skip(
            f"OTLP_ENDPOINT {mask_url(service_settings.otlp_endpoint)} is not the docker-compose "
            f"collector ({otel.otlp_grpc[0]}:{otel.otlp_grpc[1]}); this test checks that setup"
        )
    try:
        _get(otel.health_url)
    except OSError as exc:
        pytest.skip(
            f"OTel Collector not reachable at {otel.health_url} ({exc}); "
            "start the services with scripts/dev_up.sh (docker compose up -d)"
        )
    service = f"rag-integration-test-{uuid.uuid4().hex[:8]}"
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    exporter = OTLPSpanExporter(endpoint=service_settings.otlp_endpoint, insecure=True, timeout=10)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("rag.tests").start_as_current_span("collector-check") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
    provider.shutdown()

    def in_jaeger() -> bool:
        status, body = _get(f"{otel.jaeger_url}/api/v3/traces/{trace_id}")
        return status == 200 and service in body

    def in_file() -> bool:
        if not otel.trace_file.exists():
            return False
        with otel.trace_file.open(encoding="utf-8") as f:
            for line in f:
                if trace_id in line and service in line:
                    json.loads(line)  # each line is one OTLP/JSON export request
                    return True
        return False

    assert _wait_for(in_jaeger), f"trace {trace_id} did not reach Jaeger at {otel.jaeger_url}"
    assert _wait_for(in_file), f"trace {trace_id} not found in {otel.trace_file}"
