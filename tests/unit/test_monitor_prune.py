"""Spec P5-02 AC6: pruning old trace files (the safety net behind the Collector's own rotation)."""

from __future__ import annotations

import os
from pathlib import Path

from rag.monitor.prune import prune_files

NOW = 1_800_000_000.0
DAY = 86400.0


def _file(directory: Path, name: str, *, age_days: float, size: int = 10) -> Path:
    path = directory / name
    path.write_bytes(b"x" * size)
    stamp = NOW - age_days * DAY
    os.utime(path, (stamp, stamp))
    return path


def test_files_older_than_the_age_are_removed_and_newer_kept(tmp_path: Path) -> None:
    old = _file(tmp_path, "traces-old.jsonl", age_days=4)
    edge = _file(tmp_path, "traces-edge.jsonl", age_days=2.9)
    live = _file(tmp_path, "traces.jsonl", age_days=0.1)
    removed = prune_files(tmp_path, "traces*", max_age_s=3 * DAY, max_total_bytes=10_000, now=NOW)
    assert removed == [old]
    assert not old.exists() and edge.exists() and live.exists()


def test_other_files_and_folders_are_never_touched(tmp_path: Path) -> None:
    other = _file(tmp_path, "keep.txt", age_days=30)
    folder = tmp_path / "traces-dir"
    folder.mkdir()
    _file(tmp_path, "traces-old.jsonl", age_days=30)
    _file(tmp_path, "traces.jsonl", age_days=0)
    prune_files(tmp_path, "traces*", max_age_s=DAY, max_total_bytes=10_000, now=NOW)
    assert other.exists() and folder.is_dir()


def test_the_size_cap_removes_the_oldest_first(tmp_path: Path) -> None:
    a = _file(tmp_path, "traces-a.jsonl", age_days=2, size=40)
    b = _file(tmp_path, "traces-b.jsonl", age_days=1, size=40)
    c = _file(tmp_path, "traces.jsonl", age_days=0, size=40)
    removed = prune_files(tmp_path, "traces*", max_age_s=30 * DAY, max_total_bytes=90, now=NOW)
    assert removed == [a]
    assert not a.exists() and b.exists() and c.exists()
    removed = prune_files(tmp_path, "traces*", max_age_s=30 * DAY, max_total_bytes=50, now=NOW)
    assert removed == [b] and c.exists()


def test_the_newest_file_survives_the_age_rule(tmp_path: Path) -> None:
    """The live file may have had no traffic for days; the Collector still has it open."""
    quiet = _file(tmp_path, "traces.jsonl", age_days=10)
    old = _file(tmp_path, "traces-1.jsonl", age_days=20)
    removed = prune_files(tmp_path, "traces*", max_age_s=3 * DAY, max_total_bytes=10_000, now=NOW)
    assert removed == [old] and quiet.exists()


def test_dry_run_lists_without_deleting_and_a_missing_folder_is_empty(tmp_path: Path) -> None:
    old = _file(tmp_path, "traces-old.jsonl", age_days=9)
    _file(tmp_path, "traces.jsonl", age_days=0)
    listed = prune_files(
        tmp_path, "traces*", max_age_s=DAY, max_total_bytes=10_000, now=NOW, dry_run=True
    )
    assert listed == [old] and old.exists()
    assert prune_files(tmp_path / "nope", "*", max_age_s=DAY, max_total_bytes=1, now=NOW) == []
