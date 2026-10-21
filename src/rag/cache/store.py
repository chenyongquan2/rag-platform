"""Where cache entries live: Redis, memory (tests) or nowhere; and the circuit breaker (s12-1,
s16-8).

A backend stores strings under keys with an expiry, takes short locks (`SET NX PX`), reads the
data generation of a knowledge base and drops keys by prefix. Every failure of the store (Redis
down, a time-out, a refused connection) is a `CacheUnavailable`: the cache layer turns it into a
bypass and never into a failed request. `CircuitBreaker` stops the layer from calling a store
that keeps failing, so a dead Redis costs one time-out every few seconds, not one per request.

Compression is off (values are small JSON documents), and nothing here uses `KEYS` or
`FLUSHDB`: dropping a prefix walks the keys with `SCAN` and removes them with `UNLINK`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from redis import Redis

# Compare-and-delete: only the owner of a lock (its token) removes it.
_RELEASE_SCRIPT = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"
)
_GLOB_SPECIAL = "\\*?[]"


class CacheUnavailable(Exception):  # noqa: N818 - named for what it says, not as an error class
    """The cache store cannot be used right now (down, too slow, refused)."""

    def __init__(self, reason: str = "unavailable") -> None:
        super().__init__(reason)
        self.reason = reason


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_s: int) -> None: ...

    def set_nx(self, key: str, token: str, ttl_ms: int) -> bool:
        """Take a lock: True when `key` was free and now holds `token` for `ttl_ms`."""
        ...

    def release(self, key: str, token: str) -> None:
        """Free the lock `key`, only when it still holds `token`."""
        ...

    def delete_prefix(self, prefix: str, *, batch: int = 500) -> int:
        """Remove every key that starts with `prefix`; the number removed."""
        ...

    def generation(self, kb: str) -> int:
        """The data generation of `kb` (`rag.index.generation`)."""
        ...


class NullCacheBackend:
    """A store that is never there: every call is `CacheUnavailable` (cache off)."""

    def _no(self) -> CacheUnavailable:
        return CacheUnavailable("no_store")

    def get(self, key: str) -> str | None:
        raise self._no()

    def set(self, key: str, value: str, ttl_s: int) -> None:
        raise self._no()

    def set_nx(self, key: str, token: str, ttl_ms: int) -> bool:
        raise self._no()

    def release(self, key: str, token: str) -> None:
        raise self._no()

    def delete_prefix(self, prefix: str, *, batch: int = 500) -> int:
        raise self._no()

    def generation(self, kb: str) -> int:
        raise self._no()


class MemoryCacheBackend:
    """Process-local store for tests, with a clock the test moves and switches to make it fail
    (`fail_with`) or count calls (`calls`)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._data: dict[str, tuple[str, float]] = {}
        self._generations: dict[str, int] = {}
        self._lock = threading.Lock()
        self.fail_with: Exception | None = None
        self.delay_s: float = 0.0
        self.calls: dict[str, int] = {}

    def _enter(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with

    def _live(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry[1] <= self._clock():
            del self._data[key]
            return None
        return entry[0]

    def get(self, key: str) -> str | None:
        self._enter("get")
        with self._lock:
            return self._live(key)

    def set(self, key: str, value: str, ttl_s: int) -> None:
        self._enter("set")
        with self._lock:
            self._data[key] = (value, self._clock() + ttl_s)

    def set_nx(self, key: str, token: str, ttl_ms: int) -> bool:
        self._enter("set_nx")
        with self._lock:
            if self._live(key) is not None:
                return False
            self._data[key] = (token, self._clock() + ttl_ms / 1000.0)
            return True

    def release(self, key: str, token: str) -> None:
        self._enter("release")
        with self._lock:
            if self._live(key) == token:
                del self._data[key]

    def delete_prefix(self, prefix: str, *, batch: int = 500) -> int:
        self._enter("delete_prefix")
        with self._lock:
            doomed = [k for k in self._data if k.startswith(prefix)]
            for key in doomed:
                del self._data[key]
            return len(doomed)

    def generation(self, kb: str) -> int:
        self._enter("generation")
        with self._lock:
            return self._generations.setdefault(kb, 1)

    def bump_generation(self, kb: str) -> int:
        """What an ingestion run does after it has written and refreshed."""
        with self._lock:
            self._generations[kb] = self._generations.get(kb, 1) + 1
            return self._generations[kb]

    def live_keys(self) -> list[str]:
        with self._lock:
            return [k for k in list(self._data) if self._live(k) is not None]


def _glob_escape(text: str) -> str:
    return "".join(f"\\{c}" if c in _GLOB_SPECIAL else c for c in text)


class RedisCacheBackend:
    """Redis through the factory's client (`rag.factories.make_redis_client`, str responses,
    short socket time-out). `key_prefix` goes in front of every key (a test uses its own, so
    it never touches the keys of anything else)."""

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = "rag:l1:",
        generation_key_prefix: str | None = None,
        generation: Callable[[str], int] | None = None,
        generation_client: Redis | None = None,
    ) -> None:
        self.client = client
        # The counter lives in the Redis ingestion writes to (`redis_url`), which may not be the
        # one the entries live in (`cache_url`).
        self.generation_client = generation_client if generation_client is not None else client
        self.key_prefix = key_prefix
        if generation is None:
            from rag.index.generation import DEFAULT_KEY_PREFIX, get_data_generation

            gen_prefix = (
                generation_key_prefix if generation_key_prefix is not None else (DEFAULT_KEY_PREFIX)
            )

            def generation(kb: str) -> int:
                return get_data_generation(self.generation_client, kb, key_prefix=gen_prefix)

        self._generation = generation

    @staticmethod
    def _wrap(exc: Exception) -> CacheUnavailable:
        return CacheUnavailable(type(exc).__name__)

    def get(self, key: str) -> str | None:
        from redis.exceptions import RedisError

        try:
            value = self.client.get(self.key_prefix + key)
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc
        return None if value is None else str(value)

    def set(self, key: str, value: str, ttl_s: int) -> None:
        from redis.exceptions import RedisError

        try:
            self.client.set(self.key_prefix + key, value, ex=ttl_s)
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc

    def set_nx(self, key: str, token: str, ttl_ms: int) -> bool:
        from redis.exceptions import RedisError

        try:
            return bool(self.client.set(self.key_prefix + key, token, px=ttl_ms, nx=True))
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc

    def release(self, key: str, token: str) -> None:
        from redis.exceptions import RedisError

        try:
            self.client.eval(_RELEASE_SCRIPT, 1, self.key_prefix + key, token)
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc

    def delete_prefix(self, prefix: str, *, batch: int = 500) -> int:
        """`SCAN` (never `KEYS`) in steps of `batch` keys and `UNLINK`; glob characters of
        `prefix` are escaped, so it is a prefix and nothing wider."""
        from redis.exceptions import RedisError

        pattern = _glob_escape(self.key_prefix + prefix) + "*"
        removed = 0
        pending: list[str] = []
        try:
            for key in self.client.scan_iter(match=pattern, count=batch):
                pending.append(key)
                if len(pending) >= batch:
                    removed += int(self.client.unlink(*pending))
                    pending = []
            if pending:
                removed += int(self.client.unlink(*pending))
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc
        return removed

    def generation(self, kb: str) -> int:
        from redis.exceptions import RedisError

        try:
            return int(self._generation(kb))
        except (RedisError, OSError) as exc:
            raise self._wrap(exc) from exc


class CircuitBreaker:
    """Closed: calls go through. After `failures` failures in a row it opens: no call for
    `open_s` seconds. Then one call may probe (half open): its success closes the breaker, its
    failure opens it for another `open_s`. A success resets the count of failures."""

    def __init__(
        self, *, failures: int, open_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if failures < 1:
            raise ValueError("failures must be at least 1")
        if open_s <= 0:
            raise ValueError("open_s must be positive")
        self._failures_to_open = failures
        self._open_s = open_s
        self._clock = clock
        self._lock = threading.Lock()
        self._count = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self._probing else "open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._probing:
                return False
            if self._clock() - self._opened_at >= self._open_s:
                self._probing = True
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._count = 0
            self._opened_at = None
            self._probing = False

    def failure(self) -> None:
        with self._lock:
            self._count += 1
            if self._probing or self._count >= self._failures_to_open:
                self._opened_at = self._clock()
                self._probing = False
