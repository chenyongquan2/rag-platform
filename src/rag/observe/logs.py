"""Structured JSON logs with trace and request ids (spec P0-11, design s15-1).

Each line in `<log_dir>/<service>.jsonl` is one JSON object: time (UTC), level, logger,
message, service, trace_id, span_id, request_id, any `extra={...}` fields, and `exc` with the
traceback. Lines go to a local file first; a log platform can later read these files through the
Collector. Several processes often share a service's file (each stdio MCP client starts its own
server, scripts use the default name), so lines are written by `RotatingJsonlWriter`: one line
at a time under a file lock, rotated by size (`log_max_bytes`, `log_backups` old files). That
costs a few milliseconds per line on Windows, so the logging thread only formats the line (trace
and request ids are in its context variables) and puts it on a queue; one writer thread per
process writes the lines. stderr gets short text lines from `log_stderr_level` up, written
directly (stdout stays free for the MCP stdio transport).
"""

from __future__ import annotations

import atexit
import copy
import json
import logging
import logging.handlers
import queue
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.observe.context import current_request_id, current_span_id, current_trace_id
from rag.observe.rotating import LOCK_TIMEOUT_S, RotatingJsonlWriter

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


# True while one of our file handlers writes in this thread.
_EMITTING = threading.local()
_FILELOCK_LOGGER = "filelock"


class RotatingJsonlHandler(logging.Handler):
    """Writes each formatted record as one line through a `RotatingJsonlWriter`, so handlers in
    several processes can share one file and its rotation. A line that cannot be written (also
    after `lock_timeout_s`) goes to `handleError`.

    The writer takes a file lock, and filelock logs at DEBUG while it does. Those records, and
    any other record logged in this thread while one of these handlers is writing, are dropped:
    writing them would wait for a lock this thread already holds."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int,
        backups: int,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self.writer = RotatingJsonlWriter(
            path, max_bytes=max_bytes, backups=backups, lock_timeout_s=lock_timeout_s
        )

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(_EMITTING, "busy", False) or record.name.startswith(_FILELOCK_LOGGER):
            return
        _EMITTING.busy = True
        try:
            self.writer.append_line(self.format(record))
        except Exception:
            self.handleError(record)
        finally:
            _EMITTING.busy = False


class _LineFormatter(logging.Formatter):
    """The record's message is already the finished JSON line (see `_JsonQueueHandler`)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class _JsonQueueHandler(logging.handlers.QueueHandler):
    """Formats the JSON line in the logging thread and queues it for the writer thread. When
    the queue is full (the disk stalls) the line is dropped and counted, not waited for."""

    def __init__(self, line_queue: queue.Queue[logging.LogRecord]) -> None:
        super().__init__(line_queue)
        self.dropped = 0

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        line = self.format(record)
        prepared = copy.copy(record)
        prepared.msg = prepared.message = line
        prepared.args = None
        prepared.exc_info = None
        prepared.exc_text = None
        prepared.stack_info = None
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


class _SkipWriterRecords(logging.Filter):
    """Drops filelock's records and anything the writer thread logs itself: writing them would
    produce more of them, one per line written."""

    def __init__(self, listener: logging.handlers.QueueListener) -> None:
        super().__init__()
        self.listener = listener

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(_FILELOCK_LOGGER):
            return False
        thread = getattr(self.listener, "_thread", None)
        return thread is None or record.thread != thread.ident


LOG_QUEUE_SIZE = 10_000
_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"

_listeners: list[tuple[logging.handlers.QueueListener, logging.Handler]] = []
_atexit_registered = False


def setup_logging(settings: ServiceSettings, service_name: str) -> Path | None:
    """Install the JSON file handler and the stderr handler on the root logger, replacing the
    ones a previous call installed; returns the log file path (None without `log_dir`)."""
    global _previous_root_level, _atexit_registered
    teardown_logging()
    root = logging.getLogger()
    _previous_root_level = root.level
    root.setLevel(settings.log_level)
    context = TraceContextFilter(service_name)

    path: Path | None = None
    if settings.log_dir:
        path = repo_path(settings.log_dir) / f"{service_name}.jsonl"
        file_handler = RotatingJsonlHandler(
            path, max_bytes=settings.log_max_bytes, backups=settings.log_backups
        )
        file_handler.setFormatter(_LineFormatter())
        line_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=LOG_QUEUE_SIZE)
        listener = logging.handlers.QueueListener(line_queue, file_handler)
        listener.start()
        _listeners.append((listener, file_handler))
        queue_handler = _JsonQueueHandler(line_queue)
        queue_handler.setFormatter(JsonLogFormatter())
        queue_handler.addFilter(_SkipWriterRecords(listener))
        _install(root, queue_handler, context)
        if not _atexit_registered:  # write what is queued when the process ends
            atexit.register(teardown_logging)
            _atexit_registered = True
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


def flush_logging() -> None:
    """Wait until the writer threads have written every queued line."""
    for listener, _ in list(_listeners):
        listener.queue.join()


def teardown_logging() -> None:
    """Remove the handlers `setup_logging` installed, write the queued lines, stop the writer
    threads and restore the root level."""
    global _previous_root_level
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, OWN_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()
    while _listeners:
        listener, file_handler = _listeners.pop()
        listener.stop()
        file_handler.close()
    if _previous_root_level is not None:
        root.setLevel(_previous_root_level)
        _previous_root_level = None
