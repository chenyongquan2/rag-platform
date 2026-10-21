"""The environment of a server process started by a test (spec P5-01).

A test that starts `python -m rag.serve ...` gives it `server_env(...)`: the result cache is off,
because a cache entry lives an hour and a second run of the test would be served the first run's
answer (no retrieval, no trace). A test about the cache itself passes `CACHE_MODE=l1` and a key
prefix of its own.
"""

from __future__ import annotations


def server_env(**variables: str) -> dict[str, str]:
    """`variables` with `CACHE_MODE=off` unless the test sets it."""
    return {"CACHE_MODE": "off", **variables}
