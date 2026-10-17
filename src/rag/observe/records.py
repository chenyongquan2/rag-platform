"""Business records: one JSON line per request for evaluation and failure analysis (spec P0-11).

Design s15-1: the per-route candidate lists, rerank scores and the threshold decision are data
that eval scripts compare question by question, in a format we own; traces are for people
looking at timings and may move to another backend. Both carry the trace id, so a record and
its trace can be matched.

The writer appends one line per call under a file lock and rotates by size (`path.1` is the
newest old file), so several processes can share one file (`rag.observe.rotating`). A failed
write, also a lock not free within `lock_timeout_s`, is logged and reported as False, never
raised: losing a record must not fail the request.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.observe.context import current_trace_id
from rag.observe.rotating import LOCK_TIMEOUT_S, RotatingJsonlWriter
from rag.types import RetrievalResult, ThresholdDecision

logger = logging.getLogger(__name__)

TOP_IDS_PER_ROUTE = 10
SCHEMA_VERSION = 1


class RankedId(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float | None = None


class BusinessRecord(BaseModel):
    """One request. `extra` takes fields of later phases (rounds, citation checks, feedback)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    request_id: str
    trace_id: str | None = None
    tool: str
    caller: str | None = None
    kb: str
    config_version: str
    index_version: str = ""
    data_generation: int = 0
    routes: dict[str, list[RankedId]] = Field(default_factory=dict)
    rerank: list[RankedId] = Field(default_factory=list)
    threshold: ThresholdDecision | None = None
    final: list[RankedId] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


def business_record_from_result(
    result: RetrievalResult,
    *,
    tool: str,
    config_version: str,
    caller: str | None = None,
    rerank: Sequence[RankedId] = (),
    threshold: ThresholdDecision | None = None,
    extra: Mapping[str, Any] | None = None,
) -> BusinessRecord:
    """Record of one retrieval: each route's top 10 by rank, the final list, the result's flags
    and timings; the trace id of the current span."""
    routes = {
        name: [
            RankedId(chunk_id=hit.chunk_id, score=hit.score)
            for hit in sorted(hits, key=lambda h: h.rank)[:TOP_IDS_PER_ROUTE]
        ]
        for name, hits in result.routes.items()
    }
    return BusinessRecord(
        request_id=result.request_id,
        trace_id=current_trace_id(),
        tool=tool,
        caller=caller,
        kb=result.kb,
        config_version=config_version,
        index_version=result.index_version,
        data_generation=result.data_generation,
        routes=routes,
        rerank=list(rerank),
        threshold=threshold,
        final=[RankedId(chunk_id=c.chunk_id, score=c.score) for c in result.chunks],
        flags=[str(category) for category in result.error_categories],
        timings_ms=dict(result.timings_ms),
        extra=dict(extra or {}),
    )


class BusinessRecordWriter:
    """Writes `BusinessRecord`s through a `RotatingJsonlWriter`."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int,
        backups: int,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        self._writer = RotatingJsonlWriter(
            path, max_bytes=max_bytes, backups=backups, lock_timeout_s=lock_timeout_s
        )

    @property
    def path(self) -> Path:
        return self._writer.path

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> BusinessRecordWriter | None:
        """None when `business_record_path` is empty; relative paths are under the repo root."""
        if not settings.business_record_path:
            return None
        return cls(
            repo_path(settings.business_record_path),
            max_bytes=settings.business_record_max_bytes,
            backups=settings.business_record_backups,
        )

    def write(self, record: BusinessRecord) -> bool:
        try:
            self._writer.append_line(record.model_dump_json())
        except OSError as exc:  # includes the lock timeout
            logger.warning("could not write business record to %s: %s", self.path, exc)
            return False
        return True


__all__ = [
    "BusinessRecord",
    "BusinessRecordWriter",
    "RankedId",
    "RotatingJsonlWriter",
    "ThresholdDecision",
    "business_record_from_result",
]
