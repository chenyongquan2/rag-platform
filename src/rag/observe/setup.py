"""One call per process sets up traces, metrics, logs and business records (spec P0-11).

    obs = init_observability(settings, "search-service")   # at startup, once
    ...
    shutdown_observability()                               # flushes; also runs at exit

Modes (`otel_mode`): `otlp` exports traces and metrics over OTLP to the Collector, which decides
where they go (design s15-1, s17-8); `memory` keeps them in memory for tests; `off` records
nothing (helpers still work and still bind request ids).

The span and metric helpers always use the tracer and meter of the active `Observability`, not
OTel's global providers: OTel lets a process set the globals only once, while tests set up and
tear down many times. Libraries with their own OTel spans (the Elasticsearch client) take a
tracer from the global provider once and keep it, so the global TracerProvider is registered
once as `_ActiveTracerProvider`: its tracers send each span to the setup active at that moment,
also after a shutdown and a new init. The global MeterProvider is set once, to the first `otlp`
setup's provider (our own metrics never use it).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from opentelemetry import metrics as metrics_api
from opentelemetry import trace as trace_api
from opentelemetry.metrics import Meter, MeterProvider, NoOpMeterProvider
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
from opentelemetry.trace import NoOpTracerProvider, Tracer, TracerProvider

from rag.config import REPO_ROOT, ServiceSettings
from rag.config.paths import git_output
from rag.cost import add_record_listener
from rag.observe import attributes as attr
from rag.observe.instrument import instrument_frameworks
from rag.observe.logs import setup_logging, teardown_logging
from rag.observe.metrics import Metrics
from rag.observe.records import BusinessRecord, BusinessRecordWriter

logger = logging.getLogger(__name__)

PACKAGE = "rag-platform"
UNKNOWN = "unknown"

# Span limits, written out: OpenInference puts one attribute per retrieved document field and
# per message, so 30 reranked chunks need more than the SDK's 128; long texts are cut so one
# trace stays small (the full text is in ES and in the business record ids).
SPAN_LIMITS = SpanLimits(
    max_span_attributes=512,
    max_span_attribute_length=16_384,
    max_events=128,
    max_links=128,
    max_event_attributes=128,
    max_link_attributes=128,
)
# Batch export (otlp mode): at most 2048 spans wait in memory; a batch leaves every second.
SPAN_QUEUE_SIZE = 2048
SPAN_BATCH_SIZE = 512
SPAN_SCHEDULE_DELAY_MS = 1000


def service_version() -> str:
    try:
        return package_version(PACKAGE)
    except PackageNotFoundError:
        return UNKNOWN


def git_commit(settings: ServiceSettings) -> str:
    """`git_commit` from the settings (containers: GIT_COMMIT), else `git rev-parse HEAD`."""
    if settings.git_commit:
        return settings.git_commit
    head = git_output(REPO_ROOT, "rev-parse", "HEAD", timeout_s=5.0)
    return head.strip() if head else UNKNOWN


def build_resource(settings: ServiceSettings, service_name: str) -> Resource:
    return Resource.create(
        {
            attr.SERVICE_NAME: service_name,
            attr.SERVICE_VERSION: service_version(),
            attr.GIT_COMMIT: git_commit(settings),
        }
    )


def make_otlp_exporters(settings: ServiceSettings) -> tuple[SpanExporter, MetricExporter]:
    """OTLP span and metric exporters for `otlp_protocol`. gRPC takes the endpoint as is
    (`http://` means no TLS); HTTP gets the signal path appended."""
    timeout = settings.otlp_timeout_s
    if settings.otlp_protocol == "grpc":
        from grpc import Compression
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        insecure = settings.otlp_endpoint.startswith("http://")
        return (
            OTLPSpanExporter(
                endpoint=settings.otlp_endpoint,
                insecure=insecure,
                timeout=timeout,
                compression=Compression.NoCompression,
            ),
            OTLPMetricExporter(
                endpoint=settings.otlp_endpoint,
                insecure=insecure,
                timeout=timeout,
                compression=Compression.NoCompression,
            ),
        )
    from opentelemetry.exporter.otlp.proto.http import Compression as HttpCompression
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HttpMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )

    base = settings.otlp_endpoint.rstrip("/")
    return (
        HttpSpanExporter(
            endpoint=f"{base}/v1/traces",
            timeout=timeout,
            compression=HttpCompression.NoCompression,
        ),
        HttpMetricExporter(
            endpoint=f"{base}/v1/metrics",
            timeout=timeout,
            compression=HttpCompression.NoCompression,
        ),
    )


@dataclass
class Observability:
    """Providers, tracer, meter, instruments and the business record writer of one setup."""

    mode: str
    service_name: str
    resource: Resource
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    tracer: Tracer
    meter: Meter
    metrics: Metrics
    records: BusinessRecordWriter | None = None
    span_exporter: SpanExporter | None = None
    span_processor: SpanProcessor | None = None
    metric_reader: MetricReader | None = None
    log_path: Path | None = None
    _cleanup: list[Callable[[], None]] = field(default_factory=list, repr=False)
    _closed: bool = field(default=False, repr=False)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Export what is buffered now; True when every provider finished within the time."""
        millis = int(timeout_s * 1000)
        ok = True
        if isinstance(self.tracer_provider, SdkTracerProvider):
            ok = self.tracer_provider.force_flush(millis) and ok
        if isinstance(self.meter_provider, SdkMeterProvider):
            ok = self.meter_provider.force_flush(millis) and ok
        return ok

    def shutdown(self) -> None:
        """Flush, remove the hooks and instrumentation, close the providers. Idempotent."""
        if self._closed:
            return
        self._closed = True
        while self._cleanup:
            undo = self._cleanup.pop()
            try:
                undo()
            except Exception:
                logger.warning("observability cleanup step failed", exc_info=True)
        if isinstance(self.tracer_provider, SdkTracerProvider):
            self.tracer_provider.shutdown()
        if isinstance(self.meter_provider, SdkMeterProvider):
            self.meter_provider.shutdown()

    def write_record(self, record: BusinessRecord) -> bool:
        return self.records.write(record) if self.records is not None else False


_LOCK = threading.RLock()
_ACTIVE: Observability | None = None
_GLOBAL_TRACER_DONE = False
_GLOBAL_METER_DONE = False


class _ActiveTracer(Tracer):
    """A tracer handed out by OTel's global provider: each span goes to the tracer provider of
    the setup active when the span starts (none active: a no-op span)."""

    def __init__(self, scope: tuple[str, str | None, str | None, Any]) -> None:
        self._scope = scope
        self._cached: tuple[TracerProvider, Tracer] | None = None

    def _current(self) -> Tracer:
        provider = get_observability().tracer_provider
        cached = self._cached
        if cached is None or cached[0] is not provider:
            name, version, schema_url, attributes = self._scope
            cached = (provider, provider.get_tracer(name, version, schema_url, attributes))
            self._cached = cached
        return cached[1]

    def start_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._current().start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._current().start_as_current_span(*args, **kwargs)


class _ActiveTracerProvider(TracerProvider):
    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Any = None,
    ) -> Tracer:
        return _ActiveTracer(
            (instrumenting_module_name, instrumenting_library_version, schema_url, attributes)
        )


_GLOBAL_TRACER_PROVIDER = _ActiveTracerProvider()


def _register_global_providers(obs: Observability) -> None:
    """Once per process: OTel's global TracerProvider follows the active setup; the global
    MeterProvider is the first `otlp` setup's provider."""
    global _GLOBAL_TRACER_DONE, _GLOBAL_METER_DONE
    if not _GLOBAL_TRACER_DONE:
        current = trace_api.get_tracer_provider()
        if isinstance(current, trace_api.ProxyTracerProvider):
            trace_api.set_tracer_provider(_GLOBAL_TRACER_PROVIDER)
        elif current is not _GLOBAL_TRACER_PROVIDER:
            logger.warning(
                "OTel's global TracerProvider was already set (%s); spans of other libraries "
                "go there, not to the Collector",
                type(current).__name__,
            )
        _GLOBAL_TRACER_DONE = True
    if obs.mode == "otlp" and not _GLOBAL_METER_DONE:
        metrics_api.set_meter_provider(obs.meter_provider)
        _GLOBAL_METER_DONE = True


def _off(service_name: str, resource: Resource) -> Observability:
    tracer_provider, meter_provider = NoOpTracerProvider(), NoOpMeterProvider()
    meter = meter_provider.get_meter(attr.INSTRUMENTATION_SCOPE)
    return Observability(
        mode="off",
        service_name=service_name,
        resource=resource,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        tracer=tracer_provider.get_tracer(attr.INSTRUMENTATION_SCOPE),
        meter=meter,
        metrics=Metrics(meter),
    )


# Used while no setup is active: helpers work and record nothing.
_DISABLED = _off("rag", Resource.get_empty())


def get_observability() -> Observability:
    """The active setup, or a disabled one (mode `off`) before `init_observability`."""
    active = _ACTIVE
    return active if active is not None else _DISABLED


def init_observability(
    settings: ServiceSettings | None = None,
    service_name: str = "rag",
    *,
    configure_logging: bool = True,
) -> Observability:
    """Set up observability for this process; later calls return the same object until
    `shutdown_observability`."""
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            if _ACTIVE.service_name != service_name:
                logger.warning(
                    "observability already set up for %r; ignoring the setup for %r",
                    _ACTIVE.service_name,
                    service_name,
                )
            return _ACTIVE
        settings = settings if settings is not None else ServiceSettings()
        obs = _build(settings, service_name)
        try:
            if configure_logging:
                obs.log_path = setup_logging(settings, service_name)
                obs._cleanup.append(teardown_logging)
            obs.records = BusinessRecordWriter.from_settings(settings)
            obs._cleanup.append(add_record_listener(obs.metrics.record_ledger_row))
            if settings.otel_auto_instrument and obs.mode != "off":
                obs._cleanup.append(
                    instrument_frameworks(
                        obs.tracer_provider, trace_content=settings.otel_trace_content
                    )
                )
        except BaseException:
            obs.shutdown()
            raise
        _ACTIVE = obs
        _register_global_providers(obs)
        logger.info(
            "observability set up: service %s, mode %s, protocol %s",
            service_name,
            obs.mode,
            settings.otlp_protocol if obs.mode == "otlp" else "-",
        )
        return obs


def shutdown_observability() -> None:
    """Flush and close the active setup (if any); a new `init_observability` may follow."""
    global _ACTIVE
    with _LOCK:
        obs, _ACTIVE = _ACTIVE, None
    if obs is not None:
        obs.shutdown()


def write_business_record(record: BusinessRecord) -> bool:
    """Append the record with the active setup's writer; False when records are off or the
    write failed (logged)."""
    return get_observability().write_record(record)


def _build(settings: ServiceSettings, service_name: str) -> Observability:
    resource = build_resource(settings, service_name)
    if settings.otel_mode == "off":
        return _off(service_name, resource)

    tracer_provider = SdkTracerProvider(
        resource=resource,
        sampler=ParentBased(ALWAYS_ON),  # keep everything locally; sampling later in the Collector
        span_limits=SPAN_LIMITS,
        shutdown_on_exit=settings.otel_mode == "otlp",
    )
    span_exporter: SpanExporter
    span_processor: SpanProcessor
    metric_reader: MetricReader
    if settings.otel_mode == "memory":
        span_exporter = InMemorySpanExporter()
        span_processor = SimpleSpanProcessor(span_exporter)
        metric_reader = InMemoryMetricReader()
    else:
        span_exporter, metric_exporter = make_otlp_exporters(settings)
        timeout_ms = int(settings.otlp_timeout_s * 1000)
        span_processor = BatchSpanProcessor(
            span_exporter,
            max_queue_size=SPAN_QUEUE_SIZE,
            schedule_delay_millis=SPAN_SCHEDULE_DELAY_MS,
            max_export_batch_size=SPAN_BATCH_SIZE,
            export_timeout_millis=timeout_ms,
        )
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=settings.otel_metric_export_interval_s * 1000,
            export_timeout_millis=timeout_ms,
        )
    tracer_provider.add_span_processor(span_processor)
    meter_provider = SdkMeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
        shutdown_on_exit=settings.otel_mode == "otlp",
    )
    version = service_version()
    meter = meter_provider.get_meter(attr.INSTRUMENTATION_SCOPE, version)
    return Observability(
        mode=settings.otel_mode,
        service_name=service_name,
        resource=resource,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        tracer=tracer_provider.get_tracer(attr.INSTRUMENTATION_SCOPE, version),
        meter=meter,
        metrics=Metrics(meter),
        span_exporter=span_exporter,
        span_processor=span_processor,
        metric_reader=metric_reader,
    )
