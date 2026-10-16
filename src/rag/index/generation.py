"""Data generation of each knowledge base, kept in Redis (design s7-4, s7-6, s12-1).

Every cache key carries the generation; ingestion adds 1 after it has written and refreshed, and
so does an alias switch. Old cache entries then stop matching.

The cache Redis evicts keys (`allkeys-lru`) and has no persistence, so the counter can be lost.
A lost counter must not restart at 0 or 1, or entries cached under those old values would match
again. So a missing counter is created with the current Unix time in milliseconds: it is larger
than any earlier value as long as a knowledge base gets fewer increments than milliseconds pass
(always true in practice) and the clock does not go back.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis

DEFAULT_KEY_PREFIX = "rag:data_generation:"
_KB_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _key(kb: str, key_prefix: str) -> str:
    if not _KB_RE.fullmatch(kb):
        raise ValueError(f"invalid knowledge base name: {kb!r}")
    return f"{key_prefix}{kb}"


def _seed() -> int:
    return time.time_ns() // 1_000_000


def get_data_generation(client: Redis, kb: str, *, key_prefix: str = DEFAULT_KEY_PREFIX) -> int:
    """Current generation; creates it (current time in ms) when missing, so reads are stable."""
    key = _key(kb, key_prefix)
    pipe = client.pipeline(transaction=True)
    pipe.set(key, _seed(), nx=True)
    pipe.get(key)
    _, value = pipe.execute()
    return int(value)


def incr_data_generation(client: Redis, kb: str, *, key_prefix: str = DEFAULT_KEY_PREFIX) -> int:
    """Add 1 and return the new generation (a missing counter is created first, see module doc)."""
    key = _key(kb, key_prefix)
    pipe = client.pipeline(transaction=True)
    pipe.set(key, _seed(), nx=True)
    pipe.incr(key)
    _, value = pipe.execute()
    return int(value)
