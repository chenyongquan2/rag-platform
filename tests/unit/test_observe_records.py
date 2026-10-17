"""Spec P0-11: business records (one JSON line per request) and size-based rotation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from filelock import FileLock

from rag.observe import request_span, shutdown_observability, write_business_record
from rag.observe.records import (
    BusinessRecordWriter,
    RankedId,
    RotatingJsonlWriter,
    ThresholdDecision,
    business_record_from_result,
)
from rag.testing import memory_observability
from rag.types import Chunk, RetrievalResult, RouteHit


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _result() -> RetrievalResult:
    return RetrievalResult(
        request_id="req-1",
        kb="hf_docs",
        index_version="kb_hf_docs_v1",
        data_generation=42,
        degraded=True,
        timings_ms={"bm25": 12.5, "vector": 30.0, "rerank": 80.0},
        chunks=[
            Chunk(chunk_id=f"c{i}", doc_id="d", kb="hf_docs", text="t", score=0.9 - i / 10)
            for i in range(3)
        ],
        routes={
            "bm25": [
                RouteHit(chunk_id=f"b{i}", doc_id="d", rank=i + 1, score=20.0 - i)
                for i in range(12)
            ],
            # Out of order on purpose: the record sorts by rank.
            "vector": [
                RouteHit(chunk_id=f"v{i}", doc_id="d", rank=3 - i, score=0.5 + i / 10)
                for i in range(3)
            ],
        },
    )


def test_business_record_fields_from_result(tmp_path: Path) -> None:
    path = tmp_path / "records" / "business.jsonl"
    rerank = [RankedId(chunk_id="c0", score=0.91), RankedId(chunk_id="c1", score=0.42)]
    threshold = ThresholdDecision(
        threshold=0.3, top_score=0.91, applied=True, passed=True, kept=2, dropped=1
    )
    with memory_observability(business_record_path=str(path)) as obs:
        assert obs.records is not None
        with request_span("search_docs", kb="hf_docs", request_id="req-1") as scope:
            record = business_record_from_result(
                _result(),
                tool="search_docs",
                config_version="cfg-sha",
                caller="claude-code",
                rerank=rerank,
                threshold=threshold,
            )
            assert write_business_record(record)
            assert write_business_record(record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one line per write
    row = json.loads(lines[0])
    assert row["schema_version"] == 1
    assert datetime.fromisoformat(row["time"]).utcoffset().total_seconds() == 0
    assert row["request_id"] == "req-1"
    assert row["trace_id"] == scope.trace_id
    assert row["tool"] == "search_docs"
    assert row["caller"] == "claude-code"
    assert row["kb"] == "hf_docs"
    assert row["config_version"] == "cfg-sha"
    assert row["index_version"] == "kb_hf_docs_v1"
    assert row["data_generation"] == 42
    assert row["routes"]["bm25"] == [
        {"chunk_id": f"b{i}", "score": 20.0 - i} for i in range(10)
    ]  # top 10 only
    assert [h["chunk_id"] for h in row["routes"]["vector"]] == ["v2", "v1", "v0"]
    assert row["rerank"] == [
        {"chunk_id": "c0", "score": 0.91},
        {"chunk_id": "c1", "score": 0.42},
    ]
    assert row["threshold"] == {
        "threshold": 0.3,
        "top_score": 0.91,
        "applied": True,
        "passed": True,
        "kept": 2,
        "dropped": 1,
    }
    assert row["final"] == [
        {"chunk_id": f"c{i}", "score": pytest.approx(0.9 - i / 10)} for i in range(3)
    ]
    assert row["flags"] == ["degraded"]
    assert row["timings_ms"] == {"bm25": 12.5, "vector": 30.0, "rerank": 80.0}
    assert row["extra"] == {}


def test_business_records_disabled_without_path() -> None:
    with memory_observability(business_record_path="") as obs:
        assert obs.records is None
        record = business_record_from_result(_result(), tool="search_docs", config_version="x")
        assert record.trace_id is None  # no span open
        assert write_business_record(record) is False


def _files(directory: Path, stem: str) -> list[Path]:
    return sorted(p for p in directory.glob(f"{stem}*") if not p.name.endswith(".lock"))


def test_jsonl_rotation(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    # Two writer objects on one file (two services or two threads) share the rotation.
    writers = [RotatingJsonlWriter(path, max_bytes=200, backups=2) for _ in range(2)]
    for i in range(20):
        assert writers[i % 2].write({"i": i, "pad": "x" * 30})

    files = _files(tmp_path, "r.jsonl")
    assert [p.name for p in files] == ["r.jsonl", "r.jsonl.1", "r.jsonl.2"]
    numbers: list[list[int]] = []
    for p in (path, tmp_path / "r.jsonl.1", tmp_path / "r.jsonl.2"):
        assert p.stat().st_size <= 200
        numbers.append([json.loads(line)["i"] for line in p.read_text("utf-8").splitlines()])
    newest_first = [n for chunk in numbers for n in reversed(chunk)]
    assert newest_first[0] == 19
    assert newest_first == list(range(19, 19 - len(newest_first), -1))  # contiguous, oldest gone
    assert len(newest_first) < 20

    # A line larger than max_bytes still gets written, alone in a fresh file.
    big = {"pad": "y" * 500}
    assert writers[0].write(big)
    assert json.loads(path.read_text("utf-8")) == big

    # Without backups the full file is replaced.
    single = tmp_path / "s.jsonl"
    writer = RotatingJsonlWriter(single, max_bytes=100, backups=0)
    for i in range(10):
        writer.write({"i": i, "pad": "z" * 20})
    assert [p.name for p in _files(tmp_path, "s.jsonl")] == ["s.jsonl"]
    assert single.stat().st_size <= 100


def test_write_failure_does_not_raise(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    writer = BusinessRecordWriter(blocker / "business.jsonl", max_bytes=1000, backups=1)
    record = business_record_from_result(_result(), tool="search_docs", config_version="x")
    with caplog.at_level(logging.WARNING, logger="rag.observe.records"):
        assert writer.write(record) is False
    assert "business record" in caplog.text


def test_lock_timeout_returns_false(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A writer that cannot get the file lock in time gives up instead of blocking the request."""
    path = tmp_path / "business.jsonl"
    writer = BusinessRecordWriter(path, max_bytes=1000, backups=1, lock_timeout_s=0.2)
    record = business_record_from_result(_result(), tool="search_docs", config_version="x")
    with FileLock(str(path) + ".lock"), caplog.at_level(logging.WARNING, "rag.observe.records"):
        assert writer.write(record) is False
    assert "business record" in caplog.text
    assert writer.write(record) is True  # the lock is free again
