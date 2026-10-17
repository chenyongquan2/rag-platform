"""Spec P0-11: structured JSON logs with trace and request ids, rotation, idempotent setup."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from rag.observe import request_span, shutdown_observability
from rag.observe.logs import OWN_HANDLER_ATTR, setup_logging, teardown_logging
from rag.testing import memory_observability, observe_settings


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    shutdown_observability()
    teardown_logging()
    yield
    shutdown_observability()
    teardown_logging()


def _own_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, OWN_HANDLER_ATTR, False)]


def _flush() -> None:
    for handler in _own_handlers():
        handler.flush()


def test_json_log_lines_carry_trace_and_request_ids(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log = logging.getLogger("rag.test.observe")
    with memory_observability(
        service_name="svc",
        configure_logging=True,
        log_dir=str(log_dir),
        log_stderr_level="OFF",
    ):
        log.info("outside")
        with request_span("search_docs", kb="k") as scope:
            log.info("inside %s", "x", extra={"chunk_count": 3})
            try:
                1 / 0  # noqa: B018
            except ZeroDivisionError:
                log.exception("failed")
        log.debug("not at INFO")
        _flush()
        rows = [
            json.loads(line) for line in (log_dir / "svc.jsonl").read_text("utf-8").splitlines()
        ]
        lines = [row for row in rows if row["logger"] == log.name]
    assert _own_handlers() == []  # shutdown removes the handlers it added

    by_message = {row["message"]: row for row in lines}
    assert set(by_message) == {"outside", "inside x", "failed"}
    outside, inside, failed = by_message["outside"], by_message["inside x"], by_message["failed"]
    for row in lines:
        assert set(row) >= {
            "time",
            "level",
            "logger",
            "message",
            "service",
            "trace_id",
            "span_id",
            "request_id",
        }
        assert row["service"] == "svc"
        assert row["logger"] == "rag.test.observe"
        assert datetime.fromisoformat(row["time"]).utcoffset().total_seconds() == 0
    assert (outside["trace_id"], outside["span_id"], outside["request_id"]) == (None, None, None)
    assert outside["level"] == "INFO"
    assert inside["trace_id"] == scope.trace_id
    assert inside["request_id"] == scope.request_id
    assert len(inside["span_id"]) == 16
    assert inside["chunk_count"] == 3
    assert failed["level"] == "ERROR"
    assert "ZeroDivisionError" in failed["exc"]


def test_log_rotation_and_idempotent_setup(tmp_path: Path) -> None:
    settings = observe_settings(
        log_dir=str(tmp_path),
        log_max_bytes=500,
        log_backups=2,
        log_stderr_level="WARNING",
    )
    root = logging.getLogger()
    level_before = root.level
    first = setup_logging(settings, "svc")
    second = setup_logging(settings, "svc")
    assert first == second == tmp_path / "svc.jsonl"
    assert len(_own_handlers()) == 2  # one file handler, one stderr handler

    log = logging.getLogger("rag.test.rotation")
    for i in range(50):
        log.info("line %d %s", i, "y" * 50)
    _flush()
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["svc.jsonl", "svc.jsonl.1", "svc.jsonl.2"]
    assert all((tmp_path / name).stat().st_size <= 500 for name in files)
    last = (tmp_path / "svc.jsonl").read_text("utf-8").splitlines()[-1]
    assert json.loads(last)["message"].startswith("line 49 ")

    teardown_logging()
    assert _own_handlers() == []
    assert root.level == level_before

    assert setup_logging(observe_settings(log_dir="", log_stderr_level="OFF"), "svc") is None
    assert _own_handlers() == []
