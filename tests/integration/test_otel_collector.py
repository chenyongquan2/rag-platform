from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from rag.config import ServiceSettings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
# Host ports and paths from docker-compose.yml.
COLLECTOR_HEALTH_URL = "http://127.0.0.1:33133/"
JAEGER_URL = "http://127.0.0.1:36686"
TRACE_FILE = REPO_ROOT / "data" / "otel" / "traces.jsonl"


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def _wait_for(check, timeout: float = 30.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.5)
    return False


def test_collector_exports_to_jaeger_and_file(service_settings: ServiceSettings) -> None:
    try:
        _get(COLLECTOR_HEALTH_URL)
    except OSError as exc:
        pytest.skip(
            f"OTel Collector not reachable at {COLLECTOR_HEALTH_URL} ({exc}); "
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
        status, body = _get(f"{JAEGER_URL}/api/v3/traces/{trace_id}")
        return status == 200 and service in body

    def in_file() -> bool:
        if not TRACE_FILE.exists():
            return False
        with TRACE_FILE.open(encoding="utf-8") as f:
            for line in f:
                if trace_id in line and service in line:
                    json.loads(line)  # each line is one OTLP/JSON export request
                    return True
        return False

    assert _wait_for(in_jaeger), f"trace {trace_id} did not reach Jaeger at {JAEGER_URL}"
    assert _wait_for(in_file), f"trace {trace_id} not found in {TRACE_FILE}"
