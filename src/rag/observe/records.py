"""Business records: one JSON line per request for evaluation and failure analysis (spec P0-11).

Design s15-1: the per-route candidate lists, rerank scores and the threshold decision are data
that eval scripts compare question by question, in a format we own; traces are for people
looking at timings and may move to another backend. Both carry the trace id, so a record and
its trace can be matched.

The writer appends one line per call under a file lock and rotates by size (`path.1` is the
newest old file), so several processes can share one file. A failed write is logged and
reported as False, never raised: losing a record must not fail the request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.observe.context import current_trace_id
from rag.types import RetrievalResult

logger = logging.getLogger(__name__)

TOP_IDS_PER_ROUTE = 10
SCHEMA_VERSION = 1


class RankedId(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float | None = None


class ThresholdDecision(BaseModel):
    """What the rerank-score threshold did to this request's list (design s10-1, s16-8)."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    top_score: float | None = None
    applied: bool = True  # False after a rerank timeout: nothing is cut by the threshold
    passed: bool | None = None  # the best chunk reached the threshold
    kept: int = 0
    dropped: int = 0


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


class RotatingJsonlWriter:
    """Appends JSON lines to `path`, rotating by size: before a line would push the file past
    `max_bytes`, `path` becomes `path.1`, `path.1` becomes `path.2`, ..., and files beyond
    `backups` are deleted (`backups=0`: the file is simply replaced). A line longer than
    `max_bytes` is written alone into a fresh file.

    The file is opened per line under a file lock, so writers in several processes share the
    rotation, and on Windows no open handle blocks the renames.
    """

    def __init__(self, path: str | Path, *, max_bytes: int, backups: int) -> None:
        if max_bytes <= 0 or backups < 0:
            raise ValueError("max_bytes must be > 0 and backups >= 0")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()

    def write(self, obj: Mapping[str, Any]) -> bool:
        data = (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, FileLock(str(self.path) + ".lock"):
                try:
                    size = self.path.stat().st_size
                except FileNotFoundError:
                    size = 0
                if size > 0 and size + len(data) > self.max_bytes:
                    self._rotate()
                with self.path.open("ab") as fh:
                    fh.write(data)
        except OSError as exc:
            logger.warning("could not write business record to %s: %s", self.path, exc)
            return False
        return True

    def _backup(self, n: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{n}")

    def _rotate(self) -> None:
        if self.backups == 0:
            self.path.unlink()
            return
        self._backup(self.backups).unlink(missing_ok=True)
        for n in range(self.backups - 1, 0, -1):
            if self._backup(n).exists():
                os.replace(self._backup(n), self._backup(n + 1))
        os.replace(self.path, self._backup(1))


class BusinessRecordWriter:
    """Writes `BusinessRecord`s through a `RotatingJsonlWriter`."""

    def __init__(self, path: str | Path, *, max_bytes: int, backups: int) -> None:
        self._writer = RotatingJsonlWriter(path, max_bytes=max_bytes, backups=backups)

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
        return self._writer.write(record.model_dump(mode="json"))
