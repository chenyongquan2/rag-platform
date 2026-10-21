"""App-side reachability of Elasticsearch and Redis, as one gauge (spec P5-02, design s15-1).

    checker = DependencyChecker.from_settings(settings)
    register_dependency_gauge(obs.meter, checker)     # gauge rag.dependency.up{rag.dependency}

The gauge is 1 when a check passed and 0 when it failed or timed out. It is read from the app's
own side, with the app's own connection strings, so it answers "can this process reach its
stores", which Prometheus scraping the Collector cannot tell. A verdict is reused for `ttl_s`
seconds: the metric reader calls the gauge at every export and a store that is down must not be
pinged more often than that. A check never raises.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping

from opentelemetry.metrics import CallbackOptions, Meter, Observation

from rag.config import ServiceSettings
from rag.observe import attributes as attr

logger = logging.getLogger(__name__)

Check = Callable[[], None]
"""Returns when the dependency answers; raises when it does not."""


class DependencyChecker:
    """Runs named checks and keeps each verdict for `ttl_s` seconds."""

    def __init__(
        self,
        checks: Mapping[str, Check],
        *,
        ttl_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._checks = dict(checks)
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._verdicts: dict[str, bool] = {}
        self._at: float | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._checks)

    @classmethod
    def from_settings(
        cls, settings: ServiceSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> DependencyChecker:
        """Checks built with the factories (design s16-7), so `user:password@` in a connection
        string becomes authentication exactly as for the app's own clients. Nothing here logs a
        URL: a connection string may hold a password."""
        from rag import factories

        timeout = settings.dependency_check_timeout_s

        def elasticsearch() -> None:
            client = factories.make_es_client(
                settings.es_url, request_timeout=timeout, max_retries=0
            )
            try:
                if not client.ping():  # False, not an exception, when it cannot connect
                    raise OSError("Elasticsearch did not answer the ping")
            finally:
                client.close()

        def redis() -> None:
            client = factories.make_redis_client(settings.redis_url, socket_timeout=timeout)
            try:
                client.ping()
            finally:
                client.close()

        return cls(
            {"elasticsearch": elasticsearch, "redis": redis},
            ttl_s=settings.dependency_check_ttl_s,
            clock=clock,
        )

    def status(self) -> dict[str, bool]:
        """The latest verdicts; the checks run again when the last run is older than the TTL."""
        with self._lock:
            now = self._clock()
            if self._at is None or now - self._at >= self._ttl_s:
                for name, check in self._checks.items():
                    try:
                        check()
                        self._verdicts[name] = True
                    except Exception as exc:  # any failure of a check means "not reachable"
                        # Only the type: a client's message can quote its URL (and password).
                        logger.debug("dependency %s check failed: %s", name, type(exc).__name__)
                        self._verdicts[name] = False
                self._at = now
            return dict(self._verdicts)


def register_dependency_gauge(meter: Meter, checker: DependencyChecker) -> None:
    """Gauge `rag.dependency.up`: 1 or 0 per dependency, read at every metric export."""

    def observe(options: CallbackOptions) -> Iterable[Observation]:
        try:
            verdicts = checker.status()
        except Exception:  # a gauge callback must never break the export
            logger.warning("dependency check failed unexpectedly", exc_info=True)
            return []
        return [
            Observation(1 if up else 0, {attr.DEPENDENCY: name}) for name, up in verdicts.items()
        ]

    meter.create_observable_gauge(
        attr.METRIC_DEPENDENCY_UP,
        callbacks=[observe],
        unit="",
        description="1 when the app can reach the dependency, 0 when it cannot",
    )
