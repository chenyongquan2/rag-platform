"""Scheduled probe: one fixed request to a running service, as metrics (spec P5-02, design s15-1).

    prober = Prober(ProbeConfig.from_settings(settings), client=httpx.Client(), clock=time.time)
    prober.run_due()            # sends what is due now; call it every second or so

A stopped service, Elasticsearch or the gateway being down produces no trace at all, only a
request that never arrives; so a client outside the service asks one fixed question every
`interval_s` seconds. A run has one of three outcomes:

    ok       HTTP 200, not degraded, not a rerank timeout, and one of the returned passages comes
             from the expected page (`search`), or an answered question with a source from it
             (`ask`). The page may be anywhere in the list: passages tie at a rerank score of 1.0
             and their order changes with the query text.
    insane   HTTP 200 but the content is wrong: nothing found, the expected page missing, degraded
    error    no answer: refused connection, timeout, a non-200 status, a body that is not JSON

The metrics (`rag.observe.attributes`): `rag.probe.runs{kind,outcome}`, `rag.probe.duration{kind}`,
gauge `rag.probe.success{kind}` (1 after an ok run, 0 after any other) and gauge
`rag.probe.last_success_timestamp{kind}` (seconds since the epoch of the last ok run: it stays
where it was while runs fail, so an alert can say "no good run for 3 minutes") and gauge
`rag.probe.consecutive_failures{kind}` (runs in a row that were not ok; the critical alert waits
for two, design s15-1, so one lost request does not wake anybody).

The `ask` run costs model calls. It is off by default, runs at most once per `ask_interval_s`, and
at most `ask_daily_cap` times per UTC day; the count is kept in `state_path`, so restarting the
probe does not reset it. A run that failed still counts: the gateway may have billed it.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from opentelemetry.metrics import Meter

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.observe import attributes as attr
from rag.observe import get_observability
from rag.observe.context import PROBE_REQUEST_PREFIX

logger = logging.getLogger(__name__)

OK = "ok"
ERROR = "error"
INSANE = "insane"


@dataclass(frozen=True)
class ProbeConfig:
    base_url: str
    kb: str
    query: str
    expected: str
    question: str
    interval_s: float
    timeout_s: float
    ask_enabled: bool
    ask_interval_s: float
    ask_daily_cap: int
    state_path: Path
    top_k: int = 3
    unique_query: bool = False
    ask_timeout_s: float = 45.0

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> ProbeConfig:
        return cls(
            base_url=(
                settings.probe_url or f"http://{settings.service_host}:{settings.service_port}"
            ).rstrip("/"),
            kb=settings.probe_kb,
            query=settings.probe_query,
            expected=settings.probe_expected,
            question=settings.probe_ask_question,
            interval_s=settings.probe_interval_s,
            timeout_s=settings.probe_timeout_s,
            ask_enabled=settings.probe_ask_enabled,
            ask_interval_s=settings.probe_ask_interval_s,
            ask_daily_cap=settings.probe_ask_daily_cap,
            state_path=repo_path(settings.probe_state_path),
            top_k=settings.probe_top_k,
            unique_query=settings.probe_unique_query,
            ask_timeout_s=settings.probe_ask_timeout_s,
        )


@dataclass(frozen=True)
class ProbeResult:
    kind: str
    outcome: str
    duration_s: float
    detail: str = ""


@dataclass
class _State:
    day: str = ""
    count: int = 0
    last_ask_at: float = 0.0


class ProbeInstruments:
    """The probe's metric instruments (one place, so a test can list their names)."""

    def __init__(self, meter: Meter) -> None:
        self.runs = meter.create_counter(
            attr.METRIC_PROBE_RUNS, unit="{run}", description="Probe runs by kind and outcome"
        )
        self.duration = meter.create_histogram(
            attr.METRIC_PROBE_DURATION,
            unit="s",
            description="Time of one probe request",
            explicit_bucket_boundaries_advisory=[0.1, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60],
        )
        self.success = meter.create_gauge(
            attr.METRIC_PROBE_SUCCESS, unit="", description="1 when the last probe run was ok"
        )
        self.last_duration = meter.create_gauge(
            attr.METRIC_PROBE_LAST_DURATION,
            unit="s",
            description="Seconds the latest probe request took (an alert reads a slow service)",
        )
        self.last_success = meter.create_gauge(
            attr.METRIC_PROBE_LAST_SUCCESS,
            unit="s",
            description="Unix time of the last ok probe run",
        )
        self.consecutive_failures = meter.create_gauge(
            attr.METRIC_PROBE_CONSECUTIVE_FAILURES,
            unit="{run}",
            description="Probe runs in a row that were not ok (0 after an ok run)",
        )


@dataclass
class Prober:
    config: ProbeConfig
    client: httpx.Client
    clock: Callable[[], float] = time.time
    _next_search: float = field(default=0.0, init=False)
    _state: _State | None = field(default=None, init=False)
    _failures: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._metrics = ProbeInstruments(get_observability().meter)

    # ---- the two checks ---------------------------------------------------------------------

    def probe_search(self) -> ProbeResult:
        cfg = self.config
        query = cfg.query
        if cfg.unique_query:
            # A made-up word at the end: nobody asked this exact question before, so the result
            # cache cannot answer it and the request really reaches Elasticsearch and the gateway.
            query = f"{query} zq{secrets.token_hex(3)}"
        body = {"kb": cfg.kb, "query": query, "top_k": cfg.top_k}
        return self._run("search", "/v1/search_docs", body, cfg.timeout_s, self._judge_search)

    def probe_ask(self) -> ProbeResult:
        cfg = self.config
        # A call with a session id does not read the result cache (spec P5-01): the answer is
        # really computed. One session per run, so no run sees the history of an earlier one.
        body = {"kb": cfg.kb, "question": cfg.question, "session_id": f"probe-{int(self.clock())}"}
        return self._run("ask", "/v1/ask_docs", body, cfg.ask_timeout_s, self._judge_ask)

    def _expected_in(self, chunk: Any) -> bool:
        if not isinstance(chunk, dict):
            return False
        title = " ".join(str(part) for part in chunk.get("title_path") or [])
        where = f"{chunk.get('source_rel') or ''} {chunk.get('source_url') or ''} {title}"
        return self.config.expected.lower() in where.lower()

    def _judge_search(self, body: dict[str, Any]) -> tuple[str, str]:
        if body.get("degraded") or body.get("rerank_timeout"):
            return INSANE, "degraded or rerank timeout"
        chunks = body.get("chunks") or []
        if not chunks:
            return INSANE, "no passage returned"
        if not any(self._expected_in(c) for c in chunks):
            return INSANE, "no returned passage is from the expected page"
        return OK, ""

    def _judge_ask(self, body: dict[str, Any]) -> tuple[str, str]:
        if body.get("degraded") or body.get("rerank_timeout"):
            return INSANE, "degraded or rerank timeout"
        if body.get("status") != "answered" or not body.get("found"):
            return INSANE, f"status {body.get('status')}"
        if not str(body.get("answer") or "").strip():
            return INSANE, "empty answer"
        if not any(self._expected_in(c) for c in body.get("sources") or []):
            return INSANE, "no source from the expected page"
        return OK, ""

    def _run(
        self,
        kind: str,
        path: str,
        body: dict[str, Any],
        timeout_s: float,
        judge: Callable[[dict[str, Any]], tuple[str, str]],
    ) -> ProbeResult:
        began = self.clock()
        outcome, detail = ERROR, ""
        try:
            response = self.client.post(
                self.config.base_url + path,
                json=body,
                timeout=timeout_s,
                headers={"X-Request-Id": PROBE_REQUEST_PREFIX + secrets.token_hex(8)},
            )
            if response.status_code != 200:
                detail = f"HTTP {response.status_code}"
            else:
                payload = response.json()
                if isinstance(payload, dict):
                    outcome, detail = judge(payload)
                else:
                    detail = "body is not an object"
        except (httpx.HTTPError, ValueError) as exc:  # refused, timed out, not JSON
            detail = type(exc).__name__
        duration = max(0.0, self.clock() - began)
        self._record(kind, outcome, duration)
        if outcome != OK:
            logger.warning("probe %s: %s (%s) after %.1f s", kind, outcome, detail, duration)
        return ProbeResult(kind, outcome, duration, detail)

    def _record(self, kind: str, outcome: str, duration_s: float) -> None:
        failures = 0 if outcome == OK else self._failures.get(kind, 0) + 1
        self._failures[kind] = failures
        try:
            attrs = {attr.PROBE_KIND: kind}
            m = self._metrics
            m.runs.add(1, {**attrs, attr.PROBE_OUTCOME: outcome})
            m.duration.record(duration_s, attrs)
            m.success.set(1 if outcome == OK else 0, attrs)
            m.last_duration.set(duration_s, attrs)
            m.consecutive_failures.set(failures, attrs)
            if outcome == OK:
                m.last_success.set(self.clock(), attrs)
        except Exception:
            logger.warning("recording probe metrics failed", exc_info=True)

    # ---- schedule and the daily cap ----------------------------------------------------------

    def run_due(self) -> list[ProbeResult]:
        """The checks that are due at the clock's time; each is sent once."""
        now = self.clock()
        results: list[ProbeResult] = []
        if now >= self._next_search:
            self._next_search = now + self.config.interval_s
            results.append(self.probe_search())
        if self.config.ask_enabled and self._ask_allowed(self.clock()):
            self._count_ask(self.clock())
            results.append(self.probe_ask())
        return results

    def _today(self, now: float) -> str:
        return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")

    def _load(self) -> _State:
        if self._state is None:
            state = _State()
            try:
                raw = json.loads(self.config.state_path.read_text("utf-8"))
                state = _State(
                    day=str(raw["day"]),
                    count=int(raw["count"]),
                    last_ask_at=float(raw["last_ask_at"]),
                )
            except (OSError, ValueError, KeyError, TypeError):
                pass  # no file or a damaged one: start the day at zero
            self._state = state
        return self._state

    def _ask_allowed(self, now: float) -> bool:
        state = self._load()
        if now - state.last_ask_at < self.config.ask_interval_s:
            return False
        count = state.count if state.day == self._today(now) else 0
        return count < self.config.ask_daily_cap

    def _count_ask(self, now: float) -> None:
        """Write the count before the request is sent: a crash must not give a free extra run."""
        state = self._load()
        today = self._today(now)
        state.count = state.count + 1 if state.day == today else 1
        state.day = today
        state.last_ask_at = now
        try:
            self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.state_path.write_text(
                json.dumps({"day": state.day, "count": state.count, "last_ask_at": now}), "utf-8"
            )
        except OSError:
            logger.warning("could not write the probe state to %s", self.config.state_path)


def run_forever(prober: Prober, stop: threading.Event, *, tick_s: float = 1.0) -> None:
    """Run the due checks every `tick_s` seconds until `stop` is set."""
    while not stop.is_set():
        try:
            prober.run_due()
        except Exception:  # the probe must outlive any one bad run
            logger.exception("probe run failed")
        stop.wait(tick_s)
