"""Secrets for the optional Langfuse stack, written to the git-ignored `.env` (spec P5-02).

    uv run python -m rag.monitor langfuse-env

docker-compose.yml reads these names for the `langfuse` profile and for the Collector overlay. Only
keys that are missing are added (never overwritten), each value is random, and the command prints
the names it wrote, never a value.
"""

from __future__ import annotations

import base64
import secrets
from collections.abc import Callable
from pathlib import Path

KEYS: tuple[str, ...] = (
    "LANGFUSE_DB_PASSWORD",
    "LANGFUSE_REDIS_PASSWORD",
    "LANGFUSE_CLICKHOUSE_PASSWORD",
    "LANGFUSE_MINIO_PASSWORD",
    "LANGFUSE_SALT",
    "LANGFUSE_ENCRYPTION_KEY",
    "LANGFUSE_NEXTAUTH_SECRET",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_INIT_USER_PASSWORD",
    "LANGFUSE_BASIC_AUTH",
)

_MAKERS: dict[str, Callable[[], str]] = {
    "LANGFUSE_ENCRYPTION_KEY": lambda: secrets.token_hex(32),  # Langfuse wants 64 hex characters
    "LANGFUSE_PUBLIC_KEY": lambda: "pk-lf-" + secrets.token_hex(16),
    "LANGFUSE_SECRET_KEY": lambda: "sk-lf-" + secrets.token_hex(16),
}


def _parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, value = stripped.partition("=")
            values[key.strip()] = value.strip()
    return values


def ensure_env(path: str | Path) -> list[str]:
    """Add the missing keys of `KEYS` to the file; returns the names added, in file order."""
    target = Path(path)
    text = target.read_text("utf-8") if target.exists() else ""
    values = _parse(text)
    added: list[str] = []
    lines: list[str] = []
    for key in KEYS:
        if values.get(key):
            continue
        if key == "LANGFUSE_BASIC_AUTH":
            public = values.get("LANGFUSE_PUBLIC_KEY") or _find(lines, "LANGFUSE_PUBLIC_KEY")
            private = values.get("LANGFUSE_SECRET_KEY") or _find(lines, "LANGFUSE_SECRET_KEY")
            value = base64.b64encode(f"{public}:{private}".encode()).decode()
        else:
            value = _MAKERS.get(key, lambda: secrets.token_urlsafe(24))()
        lines.append(f"{key}={value}")
        added.append(key)
    if lines:
        prefix = "" if not text or text.endswith("\n") else "\n"
        header = "# Langfuse (rag.monitor langfuse-env)\n" if "LANGFUSE_" not in text else ""
        with target.open("a", encoding="utf-8", newline="") as fh:
            fh.write(prefix + header + "\n".join(lines) + "\n")
    return added


def _find(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key + "="):
            return line.partition("=")[2]
    return ""
