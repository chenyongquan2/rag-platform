"""Per-caller request limits and daily model budgets of the search service (spec P8-01, design
s16-4).

Rate limit. A fixed one-minute window per caller: the counter `<prefix><caller>:<minute>` is
incremented by every search_docs, ask_docs and get_context request (`INCR`, then `EXPIRE` 120 s
in one pipeline round trip); over the caller's limit the request is refused with
`rate_limited` and the seconds left in the minute (HTTP sends them as Retry-After). The design
puts the counters in the cache Redis, which may evict keys: an evicted counter only lets a few
more requests through. In the same spirit a Redis error lets the request through (fail-open):
the limit protects the service from one noisy caller, while money has its own hard stop (the
daily budget below and the ledger's `budget_stop_cny`), so a Redis outage should not turn every
search into an error. A fixed window lets a caller send up to twice its limit across a minute
boundary; that is accepted for a limit meant against loops, not for billing.

Daily budget. The cost ledger rows of the current UTC day (`time` is UTC) that carry a `caller`,
i.e. that were written inside a service request (`rag.cost.ledger` adds the current request's
caller). Summed per caller and in total; ask_docs is refused with `budget_exceeded` before any
model call when spent + `budget_ask_estimate_cny` would pass the caller's or the service's
limit; search_docs is still served (design: over the budget only search_docs). The ledger is
read incrementally: only the bytes appended since the last check. Spending of the request in
flight is not reserved: several concurrent asks of one caller can pass together and overshoot
by their own cost.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rag.config import ServiceSettings

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

WINDOW_S = 60
_KEY_TTL_S = 2 * WINDOW_S
_LOG_EVERY_S = 60.0  # at most one warning per minute while the counter keeps failing


class WindowCounter(Protocol):
    def incr(self, caller: str, window: int, ttl_s: int) -> int:
        """The caller's count in `window` after adding this request."""
        ...


class MemoryWindowCounter:
    """Counts in this process (tests, a single instance). Old windows are dropped."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def incr(self, caller: str, window: int, ttl_s: int) -> int:
        with self._lock:
            for key in [k for k in self._counts if k[1] < window - 1]:
                del self._counts[key]
            count = self._counts.get((caller, window), 0) + 1
            self._counts[(caller, window)] = count
            return count


class RedisWindowCounter:
    """`INCR` + `EXPIRE` of `<prefix><caller>:<window>` in one round trip."""

    def __init__(self, client: Redis | Any, *, prefix: str) -> None:
        self.client = client
        self.prefix = prefix

    def key(self, caller: str, window: int) -> str:
        return f"{self.prefix}{caller}:{window}"

    def incr(self, caller: str, window: int, ttl_s: int) -> int:
        key = self.key(caller, window)
        pipe = self.client.pipeline(transaction=False)
        pipe.incr(key)
        pipe.expire(key, ttl_s)
        count, _ = pipe.execute()
        return int(count)


@dataclass(frozen=True)
class Admission:
    allowed: bool
    count: int
    limit: int
    retry_after_s: float
    counter_error: bool = False


class RateLimiter:
    """Fixed one-minute window per caller (module doc). `per_caller` overrides `per_minute`;
    0 means no limit."""

    def __init__(
        self,
        counter: WindowCounter,
        *,
        per_minute: int,
        per_caller: Mapping[str, int],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.counter = counter
        self.per_minute = per_minute
        self.per_caller = dict(per_caller)
        self._clock = clock
        self._last_log = -math.inf

    def limit_for(self, caller: str) -> int:
        return self.per_caller.get(caller, self.per_minute)

    def admit(self, caller: str) -> Admission:
        limit = self.limit_for(caller)
        now = self._clock()
        retry_after = WINDOW_S - (now % WINDOW_S)
        if limit <= 0:
            return Admission(True, 0, 0, retry_after)
        window = int(now // WINDOW_S)
        try:
            count = self.counter.incr(caller, window, _KEY_TTL_S)
        except Exception as exc:  # fail-open: see the module doc
            if time.monotonic() - self._last_log >= _LOG_EVERY_S:
                self._last_log = time.monotonic()
                logger.warning(
                    "rate limit counter failed, requests are let through: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            return Admission(True, 0, limit, retry_after, counter_error=True)
        return Admission(count <= limit, count, limit, retry_after)


def make_rate_limiter(settings: ServiceSettings) -> RateLimiter | None:
    """The limiter the settings ask for (`rate_limit_mode`), or None when off. The Redis client
    connects at the first request, with `rate_limit_timeout_ms` for connecting and reading."""
    if settings.rate_limit_mode == "off":
        return None
    counter: WindowCounter
    if settings.rate_limit_mode == "memory":
        counter = MemoryWindowCounter()
    else:
        from rag.factories import make_redis_client

        timeout_s = settings.rate_limit_timeout_ms / 1000.0
        client = make_redis_client(settings.redis_url, socket_timeout=timeout_s)
        counter = RedisWindowCounter(client, prefix=settings.rate_limit_key_prefix)
    return RateLimiter(
        counter, per_minute=settings.rate_limit_per_minute, per_caller=settings.rate_limit_callers
    )


@dataclass(frozen=True)
class BudgetRefusal:
    scope: str  # `caller` or `service`
    spent_cny: float
    limit_cny: float
    retry_after_s: float


def _row_day_caller_cny(line: bytes) -> tuple[date, str, float] | None:
    if not line.strip():
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    if not isinstance(row, dict) or not isinstance(row.get("caller"), str):
        return None
    try:
        when = datetime.fromisoformat(str(row["time"])).astimezone(UTC)
        cny = float(row.get("cny") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    return when.date(), row["caller"], cny


class DailyBudget:
    """Today's (UTC) service spending per caller and in total from the cost ledger (module doc).
    A limit of 0 is no limit."""

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        total_cny: float,
        per_caller_cny: float,
        callers: Mapping[str, float],
        estimate_cny: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.path = Path(ledger_path)
        self.total_cny = total_cny
        self.per_caller_cny = per_caller_cny
        self.callers = dict(callers)
        self.estimate_cny = estimate_cny
        self._clock = clock
        self._lock = threading.Lock()
        self._offset = 0
        self._spent: dict[tuple[date, str], float] = {}

    def _read(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            self._offset, self._spent = 0, {}
            return
        if size < self._offset:  # rewritten, not appended: read again
            self._offset, self._spent = 0, {}
        if size == self._offset:
            return
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            new = fh.read(size - self._offset)
        complete = new[: new.rfind(b"\n") + 1]
        for line in complete.splitlines():
            parsed = _row_day_caller_cny(line)
            if parsed is not None:
                day, caller, cny = parsed
                self._spent[(day, caller)] = self._spent.get((day, caller), 0.0) + cny
        self._offset += len(complete)

    def spent_today(self, caller: str | None = None) -> float:
        """CNY spent today by `caller`, or by every caller together."""
        with self._lock:
            self._read()
            today = self._clock().astimezone(UTC).date()
            return sum(
                cny
                for (day, who), cny in self._spent.items()
                if day == today and (caller is None or who == caller)
            )

    def refusal(self, caller: str) -> BudgetRefusal | None:
        """Why one more ask_docs of `caller` would pass a limit, or None."""
        now = self._clock().astimezone(UTC)
        tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        retry_after = (tomorrow - now).total_seconds()
        limit = self.callers.get(caller, self.per_caller_cny)
        if limit > 0:
            spent = self.spent_today(caller)
            if spent + self.estimate_cny > limit:
                return BudgetRefusal("caller", spent, limit, retry_after)
        if self.total_cny > 0:
            spent = self.spent_today()
            if spent + self.estimate_cny > self.total_cny:
                return BudgetRefusal("service", spent, self.total_cny, retry_after)
        return None


def make_daily_budget(settings: ServiceSettings) -> DailyBudget | None:
    if (
        settings.budget_daily_cny <= 0
        and settings.budget_daily_caller_cny <= 0
        and not any(v > 0 for v in settings.budget_daily_callers.values())
    ):
        return None
    return DailyBudget(
        settings.cost_ledger_path,
        total_cny=settings.budget_daily_cny,
        per_caller_cny=settings.budget_daily_caller_cny,
        callers=settings.budget_daily_callers,
        estimate_cny=settings.budget_ask_estimate_cny,
    )
