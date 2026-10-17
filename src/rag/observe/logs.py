"""Structured JSON logs with trace and request ids (spec P0-11, design s15-1).

Each line in `<log_dir>/<service>.jsonl` is one JSON object: time (UTC), level, logger,
message, service, trace_id, span_id, request_id, any `extra={...}` fields, and `exc` with the
traceback. Lines go to a local file first; a log platform can later read these files through the
Collector. The file rotates by size (`log_max_bytes`, `log_backups` old files), one file per
service so two processes never rotate the same file. stderr gets short text lines from
`log_stderr_level` up (stdout stays free for the MCP stdio transport).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.observe.context import current_request_id, current_span_id, current_trace_id

# Handlers added by `setup_logging` carry this attribute, so a second setup replaces them and
# teardown removes only them (never pytest's or another library's handlers).
OWN_HANDLER_ATTR = "_rag_observe_handler"

# Attributes every LogRecord has; anything else came from `extra={...}`.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
_CONTEXT_FIELDS = ("service", "trace_id", "span_id", "request_id")

_previous_root_level: int | None = None


class TraceContextFilter(logging.Filter):
    """Adds service, trace_id, span_id and request_id (None outside a span or request) to each
    record. Runs in the thread that logs, so it sees that thread's context variables."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service
        record.trace_id = current_trace_id()
        record.span_id = current_span_id()
        record.request_id = current_request_id()
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            row[field] = getattr(record, field, None)
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and key not in row:
                row[key] = value
        if record.exc_info:
            row["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            row["exc"] = record.exc_text
        if record.stack_info:
            row["stack"] = self.formatStack(record.stack_info)
        return json.dumps(row, ensure_ascii=False, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


def setup_logging(settings: ServiceSettings, service_name: str) -> Path | None:
    """Install the JSON file handler and the stderr handler on the root logger, replacing the
    ones a previous call installed; returns the log file path (None without `log_dir`)."""
    global _previous_root_level
    teardown_logging()
    root = logging.getLogger()
    _previous_root_level = root.level
    root.setLevel(settings.log_level)
    context = TraceContextFilter(service_name)

    path: Path | None = None
    if settings.log_dir:
        path = repo_path(settings.log_dir) / f"{service_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backups,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(JsonLogFormatter())
        _install(root, file_handler, context)
    if settings.log_stderr_level != "OFF":
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(settings.log_stderr_level)
        stderr_handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
        _install(root, stderr_handler, context)
    return path


def _install(root: logging.Logger, handler: logging.Handler, context: logging.Filter) -> None:
    handler.addFilter(context)
    setattr(handler, OWN_HANDLER_ATTR, True)
    root.addHandler(handler)


def teardown_logging() -> None:
    """Remove and close the handlers `setup_logging` installed; restore the root level."""
    global _previous_root_level
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, OWN_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()
    if _previous_root_level is not None:
        root.setLevel(_previous_root_level)
        _previous_root_level = None
