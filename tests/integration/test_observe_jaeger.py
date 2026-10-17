"""Spec P0-11 AC18: spans set up by `init_observability` reach Jaeger through the Collector."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag.config import ServiceSettings
from rag.observe import (
    attributes as attr,
)
from rag.observe import (
    init_observability,
    layer_span,
    request_span,
    shutdown_observability,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
START_HINT = "start the services with scripts/dev_up.sh (docker compose up -d)"


def _host_port(service: dict, container_port: int) -> str:
    """`http://<host ip>:<host port>` of one container port in docker-compose.yml."""
    for spec in service.get("ports", []):
        host_ip, host_port, port = str(spec).split(":")
        if int(port) == container_port:
            return f"http://{host_ip}:{host_port}"
    raise AssertionError(f"port {container_port} is not published")


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def _spans(body: bytes) -> list[dict[str, Any]]:
    """Spans of a Jaeger `/api/v3/traces/<id>` answer (OTLP JSON) with their service name."""
    out = []
    for resource_spans in json.loads(body)["result"]["resourceSpans"]:
        resource = {
            a["key"]: a["value"].get("stringValue")
            for a in resource_spans["resource"]["attributes"]
        }
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                attrs = {
                    a["key"]: next(iter(a["value"].values()), None)
                    for a in span.get("attributes", [])
                }
                out.append({"name": span["name"], "service": resource.get("service.name"), **attrs})
    return out


@pytest.fixture
def compose_endpoints() -> dict[str, str]:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text("utf-8"))
    collector = compose["services"]["otel-collector"]
    endpoints = {
        "grpc": _host_port(collector, 4317),
        "http/protobuf": _host_port(collector, 4318),
        "health": _host_port(collector, 13133),
        "jaeger": _host_port(compose["services"]["jaeger"], 16686),
    }
    try:
        _get(endpoints["health"] + "/")
        _get(endpoints["jaeger"] + "/")
    except OSError as exc:
        pytest.skip(f"OTel Collector or Jaeger not reachable ({exc}); {START_HINT}")
    return endpoints


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.mark.parametrize("protocol", ["grpc", "http/protobuf"])
def test_request_span_reaches_jaeger(protocol: str, compose_endpoints: dict[str, str]) -> None:
    service = f"rag-it-{uuid.uuid4().hex[:8]}"
    settings = ServiceSettings(
        _env_file=None,
        otel_mode="otlp",
        otlp_protocol=protocol,
        otlp_endpoint=compose_endpoints[protocol],
        otlp_timeout_s=5.0,
        otel_auto_instrument=False,
        business_record_path="",
        log_dir="",
        log_stderr_level="OFF",
    )
    obs = init_observability(settings, service, configure_logging=False)
    with (
        request_span("search_docs", kb="it_kb", config_version="it-cfg") as scope,
        layer_span("retrieve", attributes={attr.ROUTE: "bm25"}),
    ):
        pass
    assert obs.flush(timeout_s=10.0)
    shutdown_observability()

    url = f"{compose_endpoints['jaeger']}/api/v3/traces/{scope.trace_id}"
    deadline = time.monotonic() + 30.0
    spans: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        status, body = _get(url)
        if status == 200:
            spans = _spans(body)
            if {s["name"] for s in spans} >= {"search_docs", "retrieve"}:
                break
        time.sleep(0.5)
    by_name = {s["name"]: s for s in spans}
    assert set(by_name) == {"search_docs", "retrieve"}, f"trace {scope.trace_id} not in Jaeger"
    root, child = by_name["search_docs"], by_name["retrieve"]
    assert root["service"] == child["service"] == service
    assert root[attr.KB] == "it_kb"
    assert root[attr.CONFIG_VERSION] == "it-cfg"
    assert root[attr.REQUEST_ID] == child[attr.REQUEST_ID] == scope.request_id
    assert child[attr.ROUTE] == "bm25"
