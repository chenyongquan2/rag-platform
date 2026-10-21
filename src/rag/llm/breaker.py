"""Circuit breaker for the chat model gateway (spec P8-01, design s16-8).

Without it, every ask_docs request during a gateway outage waits the full timeout of each model
call (and its retry) before falling back to the passages: a few seconds of nothing per call,
for every user. With it, after `failures` failed calls in a row the breaker is open: calls fail
at once with `CircuitOpenError` for `open_s` seconds (ask_docs then answers with the passages
only, as for any gateway error; search_docs needs no chat model and is unaffected). After that
one call is let through as a trial (half-open): success closes the breaker, failure opens it
for another `open_s`. Calls that arrive while the trial is running fail at once.

What counts: a failed call is one that ended, after the transport's own retries, in a connect
or read error, a timeout, or a 408, 429 or 5xx answer. A 4xx answer means the gateway is up
(the request was wrong) and counts as a success; a call stopped before sending (budget stop,
unknown price) or cancelled counts as nothing.

One breaker per gateway URL in a process (`shared_breaker`): the three chat profiles talk to the
same gateway. `CircuitOpenError` is an `httpx.TransportError`, so the openai SDK turns it into
`APIConnectionError` and `rag.llm.chat_error_category` into `llm_error`, the existing category.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(httpx.TransportError):
    """The breaker is open: the call was not sent."""


def _record(event: str, name: str) -> None:
    try:
        from rag.observe import get_observability

        get_observability().metrics.record_breaker(event=event, breaker=name)
    except Exception:  # metrics must never fail a call
        logger.debug("breaker metric not recorded", exc_info=True)


class CircuitBreaker:
    """Thread-safe closed / open / half-open breaker (module doc)."""

    def __init__(
        self,
        *,
        failures: int,
        open_s: float,
        name: str = "chat",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failures < 1:
            raise ValueError("failures must be at least 1")
        self.failures = failures
        self.open_s = open_s
        self.name = name
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failed = 0
        self._opened_at = 0.0
        self._trial = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == OPEN and self._clock() - self._opened_at >= self.open_s:
                return HALF_OPEN
            return self._state

    def before_call(self) -> None:
        """Raise `CircuitOpenError` when the call may not be sent; else let it go (as the
        half-open trial when the open time is over)."""
        with self._lock:
            if self._state == CLOSED:
                return
            if self._state == OPEN and self._clock() - self._opened_at >= self.open_s:
                self._state = HALF_OPEN
                self._trial = False
            if self._state == HALF_OPEN and not self._trial:
                self._trial = True
                logger.info("chat gateway breaker %s: half-open, sending one trial call", self.name)
                return
            wait = max(0.0, self.open_s - (self._clock() - self._opened_at))
        _record("rejected", self.name)
        raise CircuitOpenError(
            f"chat gateway breaker {self.name} is open ({wait:.1f} s left); call not sent"
        )

    def record_success(self) -> None:
        with self._lock:
            was = self._state
            self._state, self._failed, self._trial = CLOSED, 0, False
        if was != CLOSED:
            logger.warning("chat gateway breaker %s closed: the trial call succeeded", self.name)
            _record("closed", self.name)

    def record_failure(self) -> None:
        with self._lock:
            if self._state == HALF_OPEN:
                opened = True
            else:
                self._failed += 1
                opened = self._state == CLOSED and self._failed >= self.failures
            if opened:
                self._state, self._opened_at, self._trial = OPEN, self._clock(), False
            failed = self._failed
        if opened:
            logger.warning(
                "chat gateway breaker %s opened after %d failed calls: calls fail at once "
                "for %.0f s",
                self.name,
                failed,
                self.open_s,
            )
            _record("opened", self.name)

    def release(self) -> None:
        """A call that was let through but neither failed nor succeeded (not sent, cancelled):
        a half-open breaker may send another trial."""
        with self._lock:
            if self._state == HALF_OPEN:
                self._trial = False


_SHARED: dict[str, CircuitBreaker] = {}
_SHARED_LOCK = threading.Lock()


def shared_breaker(key: str, *, failures: int, open_s: float) -> CircuitBreaker | None:
    """The process's breaker for gateway `key` (its base URL); None when `failures` is 0."""
    if failures <= 0:
        return None
    with _SHARED_LOCK:
        breaker = _SHARED.get(key)
        if breaker is None or (breaker.failures, breaker.open_s) != (failures, open_s):
            breaker = CircuitBreaker(failures=failures, open_s=open_s, name="chat")
            _SHARED[key] = breaker
        return breaker


def reset_shared_breakers() -> None:
    """Forget every shared breaker (tests)."""
    with _SHARED_LOCK:
        _SHARED.clear()
