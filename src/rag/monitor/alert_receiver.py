"""A small local alert receiver: Prometheus sends alerts here, they land in a file (spec P5-02).

    python -m rag.monitor receiver            # listens on 127.0.0.1:29093

Prometheus delivers alerts through Alertmanager's HTTP API: `POST /api/v2/alerts` with a JSON
list of `{labels, annotations, startsAt, endsAt}`. This receiver speaks just that part, so
`alerting.alertmanagers` in `prometheus.yml` can point at it, with no Alertmanager container
(grouping, silencing and routing are not done here: one alert is one line).

Prometheus repeats each firing alert about once a minute and keeps sending a resolved alert for a
while. The receiver remembers which alerts it has seen firing and writes a line only when an alert
starts firing and when it resolves. Each line is appended to `alerts.jsonl` (rotated by size) and,
when `alert_webhook_url` (env `ALERT_WEBHOOK_URL`) is set, posted to that URL as one JSON body
`{status, alertname, severity, summary, description, labels, starts_at, text}`. The URL is a secret:
it is never logged, not even in a failure message.

Restarts: at start the receiver reads `alerts.jsonl` and its rotated backups and takes the last
line of each alert as its state, so an alert that was firing before a restart is not written
again as new and its resolve is still written.

Webhook: posted from one background thread (Prometheus gets its answer at once), at most
`WEBHOOK_ATTEMPTS` tries with `WEBHOOK_BACKOFF_S` seconds between them for a refused connection,
a timeout, HTTP 429 or a 5xx; a 4xx other than 429 is not tried again. The line in the file is
written first, whatever the webhook does.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rag.observe.rotating import RotatingJsonlWriter

logger = logging.getLogger(__name__)

FIRING = "firing"
RESOLVED = "resolved"
MAX_BODY_BYTES = 1_000_000
WEBHOOK_TIMEOUT_S = 5.0
WEBHOOK_ATTEMPTS = 3
WEBHOOK_BACKOFF_S = (1.0, 2.0)
"""Seconds to wait before the second and the third try."""

Poster = Callable[[str, bytes], int]
"""Posts a JSON body to a URL and returns the HTTP status; raises `OSError` when it cannot."""


def post_json(url: str, body: bytes) -> int:
    """The default poster. urllib, not httpx: httpx logs every request URL at INFO level, and the
    webhook URL is a secret."""
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_S) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


_FRACTION = re.compile(r"(\.\d{1,9})")


def parse_time(text: Any) -> float | None:
    """Unix time of an RFC 3339 timestamp (Prometheus writes nanoseconds); None for garbage and
    for the zero time (`0001-01-01...`, "no end")."""
    if not isinstance(text, str):
        return None
    cleaned = _FRACTION.sub(lambda m: m.group(1)[:7].ljust(7, "0"), text.strip())
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.year < 2000:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _fingerprint(labels: dict[str, str]) -> str:
    return json.dumps(sorted(labels.items()), ensure_ascii=False)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class BadAlertsError(ValueError):
    """The body is not a list of alerts."""


class AlertReceiver:
    def __init__(
        self,
        path: str | Path,
        *,
        webhook_url: str = "",
        poster: Poster = post_json,
        clock: Callable[[], float] = time.time,
        max_bytes: int = 5_000_000,
        backups: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._writer = RotatingJsonlWriter(path, max_bytes=max_bytes, backups=backups)
        self._webhook_url = webhook_url
        self._poster = poster
        self._clock = clock
        self._sleep = sleep
        self._firing: set[str] = _firing_from_files(Path(path), backups)
        self._lock = threading.Lock()
        self._outbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None

    def handle(self, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process one POST body; returns the lines written (for tests and logs)."""
        written: list[dict[str, Any]] = []
        now = self._clock()
        with self._lock:
            for alert in alerts:
                row = self._transition(alert, now)
                if row is not None:
                    written.append(row)
        for row in written:
            self._writer.write(row)
            self._notify(row)
        return written

    def _transition(self, alert: dict[str, Any], now: float) -> dict[str, Any] | None:
        labels = {str(k): str(v) for k, v in (alert.get("labels") or {}).items()}
        key = _fingerprint(labels)
        ends_at = parse_time(alert.get("endsAt"))
        resolved = ends_at is not None and ends_at <= now
        if resolved:
            if key not in self._firing:
                return None
            self._firing.discard(key)
            status = RESOLVED
        else:
            if key in self._firing:
                return None
            self._firing.add(key)
            status = FIRING
        notes = {str(k): str(v) for k, v in (alert.get("annotations") or {}).items()}
        starts_at = parse_time(alert.get("startsAt"))
        name = labels.get("alertname", "unknown")
        summary = notes.get("summary", "")
        return {
            "status": status,
            "alertname": name,
            "severity": labels.get("severity", ""),
            "summary": summary,
            "description": notes.get("description", ""),
            "labels": labels,
            "starts_at": _iso(starts_at) if starts_at is not None else "",
            "ends_at": _iso(ends_at) if resolved and ends_at is not None else "",
            "received_at": _iso(now),
            "text": f"[{status.upper()}] {name}: {summary}".strip(),
        }

    def _notify(self, row: dict[str, Any]) -> None:
        if not self._webhook_url:
            return
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._deliver_forever, name="alert-webhook", daemon=True
                )
                self._worker.start()
        self._outbox.put(row)

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Wait until every queued webhook post is done (tests, shutdown); False on time-out."""
        deadline = time.monotonic() + timeout_s
        while self._outbox.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _deliver_forever(self) -> None:
        while True:
            row = self._outbox.get()
            try:
                self._deliver(json.dumps(row, ensure_ascii=False).encode())
            except Exception:  # the worker must outlive any one post
                logger.warning("alert webhook delivery failed unexpectedly")
            finally:
                self._outbox.task_done()

    def _deliver(self, body: bytes) -> None:
        for attempt in range(WEBHOOK_ATTEMPTS):
            if attempt:
                self._sleep(WEBHOOK_BACKOFF_S[min(attempt, len(WEBHOOK_BACKOFF_S)) - 1])
            try:
                status = self._poster(self._webhook_url, body)
            except (OSError, ValueError) as exc:
                # The exception text may hold the URL, which is a secret: log its class only.
                logger.warning(
                    "alert webhook failed (try %s of %s): %s",
                    attempt + 1,
                    WEBHOOK_ATTEMPTS,
                    type(exc).__name__,
                )
                continue
            if status < 300:
                return
            logger.warning(
                "alert webhook answered HTTP %s (try %s of %s)",
                status,
                attempt + 1,
                WEBHOOK_ATTEMPTS,
            )
            if status < 500 and status != 429:
                return  # the hook refuses this body: sending it again will not help
        logger.warning("alert webhook: gave up after %s tries", WEBHOOK_ATTEMPTS)


def _firing_from_files(path: Path, backups: int) -> set[str]:
    """The alerts whose last line in the alert file (and its rotated backups, oldest first) says
    firing. A damaged line is skipped."""
    last: dict[str, str] = {}
    files = [path.with_name(f"{path.name}.{n}") for n in range(backups, 0, -1)] + [path]
    for file in files:
        try:
            text = file.read_text("utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
                labels = {str(k): str(v) for k, v in row["labels"].items()}
                status = str(row["status"])
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            last[_fingerprint(labels)] = status
    return {key for key, status in last.items() if status == FIRING}


def _handler_for(receiver: AlertReceiver) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "rag-alert-receiver"

        def _send(self, status: int, body: bytes = b"{}") -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/-/healthy":
                self._send(200, b'{"status": "ok"}')
            else:
                self._send(404)

        def do_POST(self) -> None:
            if self.path != "/api/v2/alerts":
                self._send(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY_BYTES:
                    raise BadAlertsError("empty or too large")
                alerts = json.loads(self.rfile.read(length))
                if not isinstance(alerts, list) or not all(isinstance(a, dict) for a in alerts):
                    raise BadAlertsError("not a list of alerts")
            except (ValueError, BadAlertsError):
                self._send(400, b'{"error": "expected a JSON list of alerts"}')
                return
            receiver.handle(alerts)
            self._send(200)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("alert receiver: " + format, *args)

    return Handler


def serve(
    receiver: AlertReceiver, host: str, port: int
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the receiver in a thread; the caller calls `server.shutdown()` to stop it."""
    server = ThreadingHTTPServer((host, port), _handler_for(receiver))
    thread = threading.Thread(target=server.serve_forever, name="alert-receiver", daemon=True)
    thread.start()
    return server, thread
