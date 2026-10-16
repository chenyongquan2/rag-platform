"""Data generation of each knowledge base, kept in Redis (design s7-4, s7-6, s12-1).

Every cache key carries the generation; ingestion adds 1 after it has written and refreshed, and
so does an alias switch. Old cache entries then stop matching.

The cache Redis evicts keys (`allkeys-lru`) and has no persistence, so the counter can be lost.
A lost counter must not restart at 0 or 1, or entries cached under those old values would match
again. So a missing counter is created with the Redis server's current Unix time in
milliseconds (`TIME`, inside a Lua script, so the check and the write are one atomic step and
every process uses the same clock): it is larger than any earlier value as long as a knowledge
base gets fewer increments than milliseconds pass (always true in practice) and the server
clock does not go back.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis

DEFAULT_KEY_PREFIX = "rag:data_generation:"
_KB_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")

# KEYS[1] = counter key, ARGV[1] = "get" or "incr". The millisecond value is built as a string
# (seconds .. three-digit milliseconds) so no Lua number formatting is involved.
_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  local now = redis.call('TIME')
  local ms = string.format('%03d', math.floor(tonumber(now[2]) / 1000))
  redis.call('SET', KEYS[1], now[1] .. ms)
end
if ARGV[1] == 'incr' then
  return redis.call('INCR', KEYS[1])
end
return redis.call('GET', KEYS[1])
"""


def _key(kb: str, key_prefix: str) -> str:
    if not _KB_RE.fullmatch(kb):
        raise ValueError(f"invalid knowledge base name: {kb!r}")
    return f"{key_prefix}{kb}"


def _run(client: Redis, kb: str, key_prefix: str, op: str) -> int:
    script = client.register_script(_SCRIPT)
    return int(script(keys=[_key(kb, key_prefix)], args=[op]))


def get_data_generation(client: Redis, kb: str, *, key_prefix: str = DEFAULT_KEY_PREFIX) -> int:
    """Current generation; creates it (server time in ms) when missing, so reads are stable."""
    return _run(client, kb, key_prefix, "get")


def incr_data_generation(client: Redis, kb: str, *, key_prefix: str = DEFAULT_KEY_PREFIX) -> int:
    """Add 1 and return the new generation (a missing counter is created first, see module doc)."""
    return _run(client, kb, key_prefix, "incr")
