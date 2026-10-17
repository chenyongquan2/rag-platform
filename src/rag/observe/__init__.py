"""Observability building blocks (spec P0-11, design s15-1, s17-8): OTel traces and metrics over
OTLP to the Collector, OpenInference auto-instrumentation, span helpers with request ids, JSON
logs and business records. Attribute and metric names: `rag.observe.attributes`."""

from rag.observe import attributes
from rag.observe.context import (
    RequestContext,
    bind_request,
    current_request,
    current_request_id,
    current_trace_id,
    new_request_id,
)
from rag.observe.metrics import Metrics
from rag.observe.records import (
    BusinessRecord,
    BusinessRecordWriter,
    RankedId,
    RotatingJsonlWriter,
    ThresholdDecision,
    business_record_from_result,
)
from rag.observe.setup import (
    Observability,
    get_observability,
    init_observability,
    shutdown_observability,
    write_business_record,
)
from rag.observe.spans import (
    RequestScope,
    activate_span,
    layer_span,
    request_span,
    set_error_category,
)

__all__ = [
    "BusinessRecord",
    "BusinessRecordWriter",
    "Metrics",
    "Observability",
    "RankedId",
    "RequestContext",
    "RequestScope",
    "RotatingJsonlWriter",
    "ThresholdDecision",
    "activate_span",
    "attributes",
    "bind_request",
    "business_record_from_result",
    "current_request",
    "current_request_id",
    "current_trace_id",
    "get_observability",
    "init_observability",
    "layer_span",
    "new_request_id",
    "request_span",
    "set_error_category",
    "shutdown_observability",
    "write_business_record",
]
