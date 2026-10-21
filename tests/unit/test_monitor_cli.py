"""Spec P5-02: the `python -m rag.monitor` command line."""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import pytest

from rag.monitor import cli
from rag.observe import shutdown_observability
from rag.testing import observe_settings


def test_prune_removes_old_trace_files_by_the_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_800_000_000.0
    old = tmp_path / "traces-old.jsonl"
    live = tmp_path / "traces.jsonl"
    old.write_text("x", "utf-8")
    live.write_text("y", "utf-8")
    os.utime(old, (now - 10 * 86400, now - 10 * 86400))
    os.utime(live, (now - 60, now - 60))
    settings = observe_settings(trace_retention_dir=str(tmp_path), trace_retention_days=3)
    monkeypatch.setattr(cli.time, "time", lambda: now)
    assert cli.main(["prune", "--dry-run"], settings=settings) == 0
    assert old.exists()
    assert "would remove 1" in capsys.readouterr().out
    assert cli.main(["prune"], settings=settings) == 0
    assert not old.exists() and live.exists()


def test_receiver_command_serves_until_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "alerts.jsonl"
    settings = observe_settings(alert_log_path=str(log), alert_receiver_port=29193)
    seen: list[int] = []

    def wait(stop: threading.Event) -> None:
        with urllib.request.urlopen("http://127.0.0.1:29193/-/healthy", timeout=5) as response:
            seen.append(response.status)

    monkeypatch.setattr(cli, "_wait_for_interrupt", wait)
    assert cli.main(["receiver"], settings=settings) == 0
    assert seen == [200]


def test_probe_command_runs_the_prober_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutdown_observability()
    settings = observe_settings(
        probe_url="http://127.0.0.1:1",  # nothing listens there: every run is an error
        probe_state_path=str(tmp_path / "state.json"),
        probe_interval_s=0.05,
    )
    monkeypatch.setattr(cli, "_wait_for_interrupt", lambda stop: stop.wait(0.3))
    assert cli.main(["probe"], settings=settings) == 0
    assert not (tmp_path / "state.json").exists()
    shutdown_observability()
