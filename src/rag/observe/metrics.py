"""Metric instruments (spec P0-11, design s15-1).

Requests, errors by category, degraded and rerank-timeout counts per knowledge base, a latency
histogram per step, and model tokens and cost fed by the cost ledger. Attribute values are kept
to a few distinct values each (kb, tool, step, category, model): no request ids in metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from opentelemetry.metrics import Meter

from rag.observe import attributes as attr
from rag.observe.context import current_request
from rag.types import ErrorCategory

logger = logging.getLogger(__name__)

# Seconds. Explicit, not the SDK default (which is in milliseconds-sized steps up to 10 000):
# query steps take milliseconds to a few seconds; offline model calls up to minutes.
STEP_DURATION_BUCKETS_S: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
    60.0,
    120.0,
)


def _drop_none(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if v is not None}


class Metrics:
    """The instruments of one meter. Recording never raises."""

    def __init__(self, meter: Meter) -> None:
        self.requests = meter.create_counter(
            attr.METRIC_REQUESTS, unit="{request}", description="Requests by kb, tool, outcome"
        )
        self.errors = meter.create_counter(
            attr.METRIC_ERRORS,
            unit="{error}",
            description="Error categories by kb and tool (design s16-8)",
        )
        self.degraded = meter.create_counter(
            attr.METRIC_DEGRADED,
            unit="{request}",
            description="Requests answered with only one retrieval route",
        )
        self.rerank_timeouts = meter.create_counter(
            attr.METRIC_RERANK_TIMEOUTS,
            unit="{request}",
            description="Requests whose rerank step timed out",
        )
        self.step_duration = meter.create_histogram(
            attr.METRIC_STEP_DURATION,
            unit="s",
            description="Duration of each request and layer step",
            explicit_bucket_boundaries_advisory=list(STEP_DURATION_BUCKETS_S),
        )
        self.model_calls = meter.create_counter(
            attr.METRIC_MODEL_CALLS,
            unit="{call}",
            description="Real model calls (cost ledger lines) by kind, model, profile, caller",
        )
        self.model_tokens = meter.create_counter(
            attr.METRIC_MODEL_TOKENS,
            unit="{token}",
            description="Model tokens from the cost ledger",
        )
        self.model_cost = meter.create_counter(
            attr.METRIC_MODEL_COST,
            unit="{CNY}",
            description="Model cost in CNY from the cost ledger",
        )

    def record_request(
        self,
        *,
        kb: str | None,
        tool: str,
        outcome: str,
        error_categories: Iterable[str] = (),
    ) -> None:
        """One finished request; each category counts once (degraded and rerank_timeout also in
        their own counters)."""
        base = _drop_none({attr.KB: kb, attr.TOOL: tool})
        try:
            self.requests.add(1, {**base, attr.OUTCOME: outcome})
            for category in dict.fromkeys(error_categories):
                self.errors.add(1, {**base, attr.ERROR_CATEGORY: str(category)})
                if category == ErrorCategory.DEGRADED:
                    self.degraded.add(1, base)
                elif category == ErrorCategory.RERANK_TIMEOUT:
                    self.rerank_timeouts.add(1, base)
        except Exception:  # metrics must never fail a request
            logger.warning("recording request metrics failed", exc_info=True)

    def record_step(
        self, step: str, duration_s: float, *, outcome: str, kb: str | None = None
    ) -> None:
        try:
            self.step_duration.record(
                duration_s, _drop_none({attr.STEP: step, attr.KB: kb, attr.OUTCOME: outcome})
            )
        except Exception:
            logger.warning("recording step duration failed", exc_info=True)

    def record_ledger_row(self, row: Mapping[str, Any]) -> None:
        """One call, its tokens and CNY from one cost ledger line (`rag.cost.add_record_listener`).

        The caller comes from the current request (design s15-1: calls, tokens and cost per
        caller per day). Calls made outside a request, or in a thread that did not copy the
        request's context, have no caller attribute."""
        request = current_request()
        base = _drop_none(
            {
                attr.MODEL_KIND: row.get("kind"),
                attr.GEN_AI_REQUEST_MODEL: row.get("model"),
                attr.MODEL_PROFILE: row.get("profile"),
                attr.CALLER: request.caller if request is not None else None,
            }
        )
        self.model_calls.add(1, base)
        estimated = bool(row.get("usage_estimated", False))
        for token_type, key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
            count = int(row.get(key) or 0)
            if count > 0:
                self.model_tokens.add(
                    count,
                    {**base, attr.GEN_AI_TOKEN_TYPE: token_type, attr.USAGE_ESTIMATED: estimated},
                )
        cny = float(row.get("cny") or 0.0)
        if cny > 0:
            self.model_cost.add(cny, base)
