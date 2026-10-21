"""Spec P5-02 AC5: the local alert receiver (Alertmanager v2 API subset), its file and webhook."""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rag.monitor.alert_receiver import AlertReceiver, parse_time, serve

NOW = datetime(2026, 10, 21, 10, 0, 0, tzinfo=UTC)
SECRET_URL = "https://hooks.example.test/secret-token-123"


class Clock:
    def __init__(self) -> None:
        self.now = NOW.timestamp()

    def __call__(self) -> float:
        return self.now


def _alert(
    name: str = "SearchProbeFailing", *, firing: bool = True, **labels: str
) -> dict[str, Any]:
    starts = NOW - timedelta(minutes=2)
    ends = NOW + timedelta(minutes=4) if firing else NOW - timedelta(seconds=5)
    return {
        "labels": {"alertname": name, "severity": "critical", **labels},
        "annotations": {"summary": f"{name} summary", "description": "details"},
        "startsAt": starts.strftime("%Y-%m-%dT%H:%M:%S.123456789Z"),
        "endsAt": ends.strftime("%Y-%m-%dT%H:%M:%S.123456789Z"),
        "generatorURL": "http://prometheus:9090/graph",
    }


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _receiver(tmp_path: Path, clock: Clock, **over: Any) -> AlertReceiver:
    return AlertReceiver(tmp_path / "alerts.jsonl", clock=clock, **over)


def test_firing_is_written_once_and_resolved_once(tmp_path: Path) -> None:
    clock = Clock()
    receiver = _receiver(tmp_path, clock)
    # Prometheus sends the same firing alert again every minute or so.
    for _ in range(3):
        receiver.handle([_alert()])
        clock.now += 60
    receiver.handle([_alert(firing=False)])
    receiver.handle([_alert(firing=False)])  # a resolved alert is resent for a while
    rows = _lines(tmp_path / "alerts.jsonl")
    assert [r["status"] for r in rows] == ["firing", "resolved"]
    first = rows[0]
    assert first["alertname"] == "SearchProbeFailing" and first["severity"] == "critical"
    assert first["summary"] == "SearchProbeFailing summary" and first["starts_at"].endswith("Z")
    assert first["labels"]["alertname"] == "SearchProbeFailing"
    assert "received_at" in first


def test_alerts_with_different_labels_are_separate(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path, Clock())
    receiver.handle([_alert("DependencyDown", dependency="redis")])
    receiver.handle([_alert("DependencyDown", dependency="elasticsearch")])
    receiver.handle([_alert("DependencyDown", dependency="redis")])
    rows = _lines(tmp_path / "alerts.jsonl")
    assert [r["labels"]["dependency"] for r in rows] == ["redis", "elasticsearch"]


def test_an_alert_that_fires_again_after_resolving_is_written_again(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path, Clock())
    receiver.handle([_alert()])
    receiver.handle([_alert(firing=False)])
    receiver.handle([_alert()])
    assert [r["status"] for r in _lines(tmp_path / "alerts.jsonl")] == [
        "firing",
        "resolved",
        "firing",
    ]


def test_a_resolved_alert_never_seen_firing_is_ignored(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path, Clock())
    receiver.handle([_alert(firing=False)])
    assert _lines(tmp_path / "alerts.jsonl") == []


def test_webhook_gets_one_post_per_change_and_its_url_is_never_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sent: list[tuple[str, bytes]] = []

    def poster(url: str, body: bytes) -> int:
        sent.append((url, body))
        return 200

    receiver = _receiver(tmp_path, Clock(), webhook_url=SECRET_URL, poster=poster)
    with caplog.at_level(logging.DEBUG):
        receiver.handle([_alert()])
        receiver.handle([_alert()])
    assert len(sent) == 1
    assert sent[0][0] == SECRET_URL
    body = json.loads(sent[0][1])
    assert body["status"] == "firing" and body["alertname"] == "SearchProbeFailing"
    assert "SearchProbeFailing" in body["text"] and "summary" in body["text"]
    assert "secret-token" not in caplog.text and "hooks.example.test" not in caplog.text


def test_a_failing_webhook_does_not_lose_the_line_or_leak_the_url(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def poster(url: str, body: bytes) -> int:
        raise ConnectionError(f"cannot reach {url}")

    receiver = _receiver(tmp_path, Clock(), webhook_url=SECRET_URL, poster=poster)
    with caplog.at_level(logging.DEBUG):
        receiver.handle([_alert()])
    assert [r["status"] for r in _lines(tmp_path / "alerts.jsonl")] == ["firing"]
    assert "secret-token" not in caplog.text and "hooks.example.test" not in caplog.text
    assert "webhook" in caplog.text  # the failure itself is logged


def test_a_bad_status_from_the_webhook_is_logged_without_the_url(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    receiver = _receiver(tmp_path, Clock(), webhook_url=SECRET_URL, poster=lambda u, b: 500)
    with caplog.at_level(logging.DEBUG):
        receiver.handle([_alert()])
    assert "HTTP 500" in caplog.text and "secret-token" not in caplog.text


def test_no_webhook_when_the_url_is_empty(tmp_path: Path) -> None:
    def poster(url: str, body: bytes) -> int:
        raise AssertionError("no request expected")

    receiver = _receiver(tmp_path, Clock(), webhook_url="", poster=poster)
    receiver.handle([_alert()])
    assert len(_lines(tmp_path / "alerts.jsonl")) == 1


def test_the_default_poster_sends_json_and_returns_the_status() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from rag.monitor.alert_receiver import post_json

    got: list[tuple[str, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            got.append((self.headers["Content-Type"], self.rfile.read(length)))
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = post_json(f"http://127.0.0.1:{server.server_address[1]}/hook", b'{"a": 1}')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert status == 202 and got == [("application/json", b'{"a": 1}')]


def test_parse_time_takes_prometheus_timestamps() -> None:
    assert parse_time("2026-10-21T10:00:00.123456789Z") == pytest.approx(NOW.timestamp() + 0.123456)
    assert parse_time("2026-10-21T10:00:00Z") == NOW.timestamp()
    assert parse_time("2026-10-21T18:00:00+08:00") == NOW.timestamp()
    assert parse_time("0001-01-01T00:00:00Z") is None  # "no end"
    assert parse_time("garbage") is None


def test_http_endpoint_takes_a_prometheus_post(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path, Clock())
    server, thread = serve(receiver, "127.0.0.1", 0)
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        assert urllib.request.urlopen(base + "/-/healthy", timeout=5).status == 200
        request = urllib.request.Request(
            base + "/api/v2/alerts",
            data=json.dumps([_alert(), _alert("HighErrorRate")]).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assert urllib.request.urlopen(request, timeout=5).status == 200
        bad = urllib.request.Request(base + "/api/v2/alerts", data=b"not json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(bad, timeout=5)
        assert caught.value.code == 400
        wrong_shape = urllib.request.Request(
            base + "/api/v2/alerts", data=b'{"labels": {}}', method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(wrong_shape, timeout=5)
        assert caught.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(base + "/other", timeout=5)
        assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert {r["alertname"] for r in _lines(tmp_path / "alerts.jsonl")} == {
        "SearchProbeFailing",
        "HighErrorRate",
    }
