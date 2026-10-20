"""In-memory observability for unit tests (spec P0-11)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rag.config import ServiceSettings
from rag.observe.setup import Observability, init_observability, shutdown_observability

# Unit-test defaults: in memory, no framework instrumentation, no files, nothing on stderr.
_TEST_DEFAULTS: dict[str, Any] = {
    "otel_mode": "memory",
    "otel_auto_instrument": False,
    "otel_trace_content": True,
    "git_commit": "test-commit",
    "business_record_path": "",
    "log_dir": "",
    "log_stderr_level": "OFF",
    # Starlette's test client sends `Host: testserver`.
    "service_allowed_hosts": ["testserver"],
}


def observe_settings(**overrides: Any) -> ServiceSettings:
    """Service settings without `.env`, with the unit-test observability defaults."""
    return ServiceSettings(_env_file=None, **{**_TEST_DEFAULTS, **overrides})  # type: ignore[call-arg]


@contextmanager
def memory_observability(
    *, service_name: str = "test", configure_logging: bool = False, **overrides: Any
) -> Iterator[Observability]:
    """Set up observability in `memory` mode for the block (any active setup is shut down
    first) and shut it down afterwards. Spans: `obs.span_exporter.get_finished_spans()`;
    metrics: `metric_points(obs, name)`."""
    shutdown_observability()
    obs = init_observability(
        observe_settings(**overrides), service_name, configure_logging=configure_logging
    )
    try:
        yield obs
    finally:
        shutdown_observability()


def metric_points(obs: Observability, name: str) -> list[tuple[dict[str, Any], Any]]:
    """`(attributes, data point)` of every point of metric `name` collected now (counters have
    `.value`; histograms `.count`, `.sum`, `.bucket_counts`, `.explicit_bounds`)."""
    reader = obs.metric_reader
    if reader is None or not hasattr(reader, "get_metrics_data"):
        return []
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        (dict(point.attributes or {}), point)
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]
