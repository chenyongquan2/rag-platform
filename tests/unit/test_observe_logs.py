"""Spec P0-11: structured JSON logs with trace and request ids, rotation, idempotent setup."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from rag.observe import request_span, shutdown_observability
from rag.observe.logs import (
    OWN_HANDLER_ATTR,
    JsonLogFormatter,
    RotatingJsonlHandler,
    flush_logging,
    setup_logging,
    teardown_logging,
)
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
    flush_logging()  # lines are written by a writer thread


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
    files = sorted(p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock"))
    assert files == ["svc.jsonl", "svc.jsonl.1", "svc.jsonl.2"]
    assert all((tmp_path / name).stat().st_size <= 500 for name in files)
    last = (tmp_path / "svc.jsonl").read_text("utf-8").splitlines()[-1]
    assert json.loads(last)["message"].startswith("line 49 ")

    teardown_logging()
    assert _own_handlers() == []
    assert root.level == level_before

    assert setup_logging(observe_settings(log_dir="", log_stderr_level="OFF"), "svc") is None
    assert _own_handlers() == []


def _all_rows(directory: Path, stem: str) -> list[dict]:
    rows = []
    for path in directory.glob(f"{stem}*"):
        if path.name.endswith(".lock"):
            continue
        rows.extend(json.loads(line) for line in path.read_text("utf-8").splitlines())
    return rows


def test_two_handlers_share_one_log_file_without_errors(tmp_path: Path) -> None:
    """Two handlers on one file stand for two processes of one service (each stdio MCP client
    starts its own server). The stdlib RotatingFileHandler keeps the file open, so on Windows
    the other handler's rename fails and lines are lost."""
    path = tmp_path / "rag.jsonl"
    errors: list[logging.LogRecord] = []
    loggers = []
    for n in range(2):
        handler = RotatingJsonlHandler(path, max_bytes=1500, backups=200)
        handler.setFormatter(JsonLogFormatter())
        handler.handleError = errors.append  # type: ignore[method-assign]
        log = logging.getLogger(f"rag.test.shared{n}")
        log.setLevel(logging.DEBUG)  # filelock logs at DEBUG while our handler holds its lock
        log.propagate = False
        log.addHandler(handler)
        loggers.append((log, handler))
    filelock_log = logging.getLogger("filelock")
    previous = filelock_log.level, list(filelock_log.handlers)
    filelock_log.setLevel(logging.DEBUG)
    filelock_log.addHandler(loggers[0][1])  # nested records from inside emit must not deadlock
    try:
        for i in range(60):
            for n, (log, _) in enumerate(loggers):
                log.info("w%d-%d %s", n, i, "x" * 40)
    finally:
        filelock_log.removeHandler(loggers[0][1])
        filelock_log.setLevel(previous[0])
        for log, handler in loggers:
            log.removeHandler(handler)
            handler.close()
    assert errors == []
    messages = [r["message"].split()[0] for r in _all_rows(tmp_path, "rag.jsonl")]
    expected = [f"w{n}-{i}" for i in range(60) for n in range(2)]
    assert sorted(m for m in messages if m.startswith("w")) == sorted(expected)  # none lost
    assert all(p.stat().st_size <= 1500 for p in tmp_path.glob("rag.jsonl*") if p.suffix != ".lock")


_WRITER = textwrap.dedent(
    """
    import logging, pathlib, sys, time
    from rag.config import ServiceSettings
    from rag.observe.logs import setup_logging
    log_dir, sync_dir, worker = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
    settings = ServiceSettings(
        _env_file=None, log_dir=log_dir, log_max_bytes=2000, log_backups=200,
        log_stderr_level="OFF",
    )
    setup_logging(settings, "rag")
    log = logging.getLogger("rag.test.process")
    (sync_dir / f"ready-{worker}").touch()
    while not (sync_dir / "go").exists():
        time.sleep(0.01)
    for i in range(60):
        log.info("p%s-%d %s", worker, i, "y" * 40)
        time.sleep(0.005)  # interleave with the other process
    # no teardown: the queued lines are written at exit
    """
)


def test_two_processes_share_one_log_file(tmp_path: Path) -> None:
    """Both processes write at the same time (start barrier, paced lines). With the stdlib
    RotatingFileHandler this loses about a third of the lines on Windows."""
    log_dir, sync_dir = tmp_path / "logs", tmp_path / "sync"
    sync_dir.mkdir()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(log_dir), str(sync_dir), str(n)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(2)
    ]
    deadline = time.monotonic() + 60
    while not all((sync_dir / f"ready-{n}").exists() for n in range(2)):
        assert time.monotonic() < deadline, "writer processes did not start"
        assert all(proc.poll() is None for proc in procs), "a writer process exited early"
        time.sleep(0.05)
    (sync_dir / "go").touch()
    outputs = [proc.communicate(timeout=120) for proc in procs]
    for proc, (_, err) in zip(procs, outputs, strict=True):
        assert proc.returncode == 0, err
        assert "Logging error" not in err, err
    rows = _all_rows(log_dir, "rag.jsonl")
    messages = sorted(r["message"].split()[0] for r in rows if r["logger"] == "rag.test.process")
    assert messages == sorted(f"p{n}-{i}" for n in range(2) for i in range(60))
