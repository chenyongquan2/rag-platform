"""Spec P0-11: settings, init modes, exporters, resource and lifecycle of `rag.observe`."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from importlib.metadata import version
from pathlib import Path

import pytest
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GrpcMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HttpMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from pydantic import ValidationError

from rag.config import ModelPrice, ServiceSettings
from rag.cost import CostLedger
from rag.cost import ledger as ledger_module
from rag.observe import (
    attributes as attr,
)
from rag.observe import (
    current_request_id,
    current_trace_id,
    get_observability,
    init_observability,
    layer_span,
    request_span,
    shutdown_observability,
)
from rag.observe import setup as observe_setup
from rag.testing import memory_observability, metric_points, observe_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

_OBSERVE_ENV = [
    "OTEL_MODE",
    "OTLP_ENDPOINT",
    "OTLP_PROTOCOL",
    "OTLP_TIMEOUT_S",
    "OTEL_METRIC_EXPORT_INTERVAL_S",
    "OTEL_AUTO_INSTRUMENT",
    "OTEL_TRACE_CONTENT",
    "GIT_COMMIT",
    "BUSINESS_RECORD_PATH",
    "BUSINESS_RECORD_MAX_BYTES",
    "BUSINESS_RECORD_BACKUPS",
    "LOG_DIR",
    "LOG_LEVEL",
    "LOG_STDERR_LEVEL",
    "LOG_MAX_BYTES",
    "LOG_BACKUPS",
]


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def test_observe_settings_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _OBSERVE_ENV:
        monkeypatch.delenv(key, raising=False)
    s = ServiceSettings(_env_file=None)
    assert s.otel_mode == "otlp"
    assert s.otlp_protocol == "grpc"
    assert s.otlp_endpoint == "http://127.0.0.1:24317"
    assert s.otlp_timeout_s == 5.0
    assert s.otel_metric_export_interval_s == 30.0
    assert s.otel_auto_instrument is True
    assert s.otel_trace_content is True
    assert s.git_commit == ""
    assert s.business_record_path == "data/observe/records/business.jsonl"
    assert s.business_record_max_bytes == 20_000_000
    assert s.business_record_backups == 5
    assert s.log_dir == "data/observe/logs"
    assert s.log_level == "INFO"
    assert s.log_stderr_level == "WARNING"
    assert s.log_max_bytes == 10_000_000
    assert s.log_backups == 3

    monkeypatch.setenv("OTEL_MODE", "memory")
    monkeypatch.setenv("OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_AUTO_INSTRUMENT", "false")
    monkeypatch.setenv("LOG_BACKUPS", "1")
    s = ServiceSettings(_env_file=None)
    assert (s.otel_mode, s.otlp_protocol, s.otel_auto_instrument, s.log_backups) == (
        "memory",
        "http/protobuf",
        False,
        1,
    )
    with pytest.raises(ValidationError):
        ServiceSettings(_env_file=None, otlp_protocol="thrift")
    with pytest.raises(ValidationError):
        ServiceSettings(_env_file=None, log_max_bytes=0)
    with pytest.raises(ValidationError):
        ServiceSettings(_env_file=None, otel_metric_export_interval_s=0)


def test_otlp_exporters_follow_protocol() -> None:
    grpc = observe_settings(
        otel_mode="otlp",
        otlp_protocol="grpc",
        otlp_endpoint="http://127.0.0.1:24317",
        otlp_timeout_s=4.0,
    )
    spans, metrics = observe_setup.make_otlp_exporters(grpc)
    try:
        assert isinstance(spans, GrpcSpanExporter)
        assert isinstance(metrics, GrpcMetricExporter)
        assert spans._endpoint == metrics._endpoint == "127.0.0.1:24317"
        assert spans._timeout == metrics._timeout == 4.0
    finally:
        spans.shutdown()
        metrics.shutdown()

    http = observe_settings(
        otel_mode="otlp",
        otlp_protocol="http/protobuf",
        otlp_endpoint="http://127.0.0.1:24318/",
        otlp_timeout_s=3.0,
    )
    spans, metrics = observe_setup.make_otlp_exporters(http)
    try:
        assert isinstance(spans, HttpSpanExporter)
        assert isinstance(metrics, HttpMetricExporter)
        assert spans._endpoint == "http://127.0.0.1:24318/v1/traces"
        assert metrics._endpoint == "http://127.0.0.1:24318/v1/metrics"
        assert spans._client._timeout == metrics._client._timeout == 3.0
    finally:
        spans.shutdown()
        metrics.shutdown()


class _ListSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


class _ListMetricExporter(MetricExporter):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[MetricsData] = []

    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object
    ) -> MetricExportResult:
        self.batches.append(metrics_data)
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        return None


def test_otlp_mode_batches_spans_and_exports_metrics_on_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_exporter, metric_exporter = _ListSpanExporter(), _ListMetricExporter()
    monkeypatch.setattr(
        observe_setup, "make_otlp_exporters", lambda settings: (span_exporter, metric_exporter)
    )
    obs = init_observability(
        observe_settings(otel_mode="otlp", otel_metric_export_interval_s=60.0), "svc"
    )
    assert obs.mode == "otlp"
    assert isinstance(obs.span_processor, BatchSpanProcessor)
    assert isinstance(obs.metric_reader, PeriodicExportingMetricReader)
    assert obs.metric_reader._export_interval_millis == 60_000
    with request_span("search_docs", kb="k"):
        pass
    assert span_exporter.spans == []  # batched, not yet sent
    assert obs.flush()
    assert [s.name for s in span_exporter.spans] == ["search_docs"]
    assert metric_exporter.batches, "flush exports the metrics too"


def test_resource_attributes() -> None:
    with memory_observability(service_name="svc", git_commit="abc123") as obs:
        resource = obs.resource.attributes
        assert resource[attr.SERVICE_NAME] == "svc"
        assert resource[attr.SERVICE_VERSION] == version("rag-platform")
        assert resource[attr.GIT_COMMIT] == "abc123"
        with request_span("search_docs", kb="k"):
            pass
        (span,) = obs.span_exporter.get_finished_spans()
        assert span.resource.attributes[attr.SERVICE_NAME] == "svc"

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with memory_observability(git_commit="") as obs:
        assert obs.resource.attributes[attr.GIT_COMMIT] == head


def test_init_is_idempotent_and_shutdown_resets(tmp_path: Path) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    listeners_before = len(ledger_module._RECORD_LISTENERS)
    settings = observe_settings(otel_auto_instrument=True)
    first = init_observability(settings, "svc")
    assert init_observability(settings, "svc") is first
    assert get_observability() is first
    assert len(ledger_module._RECORD_LISTENERS) == listeners_before + 1
    assert LlamaIndexInstrumentor().is_instrumented_by_opentelemetry
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry

    shutdown_observability()
    assert get_observability().mode == "off"
    assert len(ledger_module._RECORD_LISTENERS) == listeners_before
    assert not LlamaIndexInstrumentor().is_instrumented_by_opentelemetry
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
    shutdown_observability()  # a second shutdown is harmless

    # Ledger lines after shutdown reach no counter of the old instance.
    ledger = CostLedger(
        tmp_path / "ledger.jsonl",
        prices={"m": ModelPrice(input_cny_per_m=1.0)},
        budget_stop_cny=10.0,
    )
    ledger.record(kind="chat", model="m", prompt_tokens=10)
    assert metric_points(first, attr.METRIC_MODEL_TOKENS) == []

    second = init_observability(settings, "svc")
    assert second is not first
    assert second.mode == "memory"


def test_memory_and_off_modes() -> None:
    with memory_observability() as obs:
        with request_span("search_docs", kb="k") as scope:
            assert current_trace_id() == scope.trace_id
        assert [s.name for s in obs.span_exporter.get_finished_spans()] == ["search_docs"]
        assert metric_points(obs, attr.METRIC_REQUESTS)

    obs = init_observability(observe_settings(otel_mode="off"), "svc")
    assert obs.mode == "off"
    assert obs.span_exporter is None
    assert obs.metric_reader is None
    with request_span("search_docs", kb="k") as scope:
        assert current_request_id() == scope.request_id
        assert scope.trace_id is None
        with layer_span("retrieve") as span:
            assert not span.is_recording()
    assert obs.flush()
