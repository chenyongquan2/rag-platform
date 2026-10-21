"""Which Prometheus series the code can produce, and which metrics a PromQL expression reads
(spec P5-02): shared by the unit test of the dashboard and rules and the integration test against
the real Collector."""

from __future__ import annotations

import re
from typing import Any

from rag.eval.online_sample import create_online_eval_gauge
from rag.monitor.probe import ProbeInstruments
from rag.observe.health import DependencyChecker, register_dependency_gauge
from rag.observe.metrics import Metrics


class RecordingMeter:
    """Remembers the (kind, name, unit) of every instrument that is created on it."""

    def __init__(self) -> None:
        self.created: list[tuple[str, str, str]] = []

    def _make(self, kind: str, name: str, unit: str = "") -> object:
        self.created.append((kind, name, unit))
        return object()

    def create_counter(self, name: str, unit: str = "", description: str = "") -> object:
        return self._make("counter", name, unit)

    def create_histogram(
        self, name: str, unit: str = "", description: str = "", **_: Any
    ) -> object:
        return self._make("histogram", name, unit)

    def create_gauge(self, name: str, unit: str = "", description: str = "") -> object:
        return self._make("gauge", name, unit)

    def create_observable_gauge(
        self, name: str, callbacks: Any = None, unit: str = "", description: str = ""
    ) -> object:
        return self._make("gauge", name, unit)


_UNIT_WORDS = {"s": "seconds", "ms": "milliseconds", "By": "bytes"}


def prometheus_names(kind: str, name: str, unit: str) -> set[str]:
    """The series names the Collector's Prometheus exporter makes of an OTel instrument: dots
    become underscores; a plain unit is appended (`s` becomes `_seconds`; units in braces such as
    `{request}` and the unit `1` of a histogram or counter add nothing; the unit `1` of a gauge
    becomes `_ratio`); counters end in `_total`; a histogram is its `_bucket`, `_sum` and `_count`.
    tests/integration/test_monitor_stack.py checks this against the Collector's real output."""
    base = name.replace(".", "_")
    if unit in _UNIT_WORDS:
        base += "_" + _UNIT_WORDS[unit]
    elif unit == "1" and kind == "gauge":
        base += "_ratio"
    if kind == "counter":
        return {base + "_total"}
    if kind == "histogram":
        return {base + suffix for suffix in ("_bucket", "_sum", "_count")}
    return {base}


def created_instruments() -> list[tuple[str, str, str]]:
    """Every (kind, OTel name, unit) the code creates, for metrics, health, probe and sampling."""
    meter = RecordingMeter()
    Metrics(meter)  # type: ignore[arg-type]
    register_dependency_gauge(meter, DependencyChecker({}))  # type: ignore[arg-type]
    ProbeInstruments(meter)  # type: ignore[arg-type]
    create_online_eval_gauge(meter)  # type: ignore[arg-type]
    return meter.created


def exported_series() -> set[str]:
    series: set[str] = set()
    for kind, name, unit in created_instruments():
        series |= prometheus_names(kind, name, unit)
    return series


_LABEL_BLOCK = re.compile(r"\{[^}]*\}")
_GROUPING = re.compile(r"\b(?:by|without)\s*\([^)]*\)")
_STRING = re.compile(r'"[^"]*"')
_METRIC = re.compile(r"\brag_[a-z0-9_]+\b")


def promql_metrics(expr: str) -> set[str]:
    """The `rag_*` metric names an expression reads (not label names such as `rag_tool`)."""
    cleaned = _STRING.sub('""', expr)
    cleaned = _LABEL_BLOCK.sub("", cleaned)
    cleaned = _GROUPING.sub("", cleaned)
    return set(_METRIC.findall(cleaned))
