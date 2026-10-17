"""Append-only JSON-lines files shared by several processes, rotated by size (spec P0-11).

Used for business records and for log files. Each line is written under a file lock with the
file opened only for that line, so writers in several processes (each stdio MCP client starts
its own server process) share one rotation, and on Windows no open handle blocks the renames.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from filelock import FileLock

logger = logging.getLogger(__name__)

# Longest wait for the file lock before a line is given up. Writes take milliseconds; a
# request must not hang behind a stuck writer.
LOCK_TIMEOUT_S = 2.0


class RotatingJsonlWriter:
    """Appends lines to `path`, rotating by size: before a line would push the file past
    `max_bytes`, `path` becomes `path.1`, `path.1` becomes `path.2`, ..., and files beyond
    `backups` are deleted (`backups=0`: the file is simply replaced). A line longer than
    `max_bytes` is written alone into a fresh file. The lock file is `path.lock`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int,
        backups: int,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        if max_bytes <= 0 or backups < 0 or lock_timeout_s <= 0:
            raise ValueError("max_bytes and lock_timeout_s must be > 0, backups >= 0")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self.lock_timeout_s = lock_timeout_s
        self._lock = threading.Lock()

    def append_line(self, line: str) -> None:
        """Append one line (a newline is added). Raises `OSError` when the file cannot be
        written, including `filelock.Timeout` (a `TimeoutError`) when the lock stays taken."""
        data = (line + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lock.acquire(timeout=self.lock_timeout_s):
            raise TimeoutError(f"thread lock of {self.path} not free in {self.lock_timeout_s} s")
        try:
            with FileLock(str(self.path) + ".lock", timeout=self.lock_timeout_s):
                try:
                    size = self.path.stat().st_size
                except FileNotFoundError:
                    size = 0
                if size > 0 and size + len(data) > self.max_bytes:
                    self._rotate()
                with self.path.open("ab") as fh:
                    fh.write(data)
        finally:
            self._lock.release()

    def write(self, obj: Mapping[str, Any]) -> bool:
        """Append `obj` as one JSON line; on failure log a warning and return False."""
        try:
            self.append_line(json.dumps(obj, ensure_ascii=False, default=str))
        except OSError as exc:
            logger.warning("could not append to %s: %s", self.path, exc)
            return False
        return True

    def _backup(self, n: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{n}")

    def _rotate(self) -> None:
        if self.backups == 0:
            self.path.unlink()
            return
        self._backup(self.backups).unlink(missing_ok=True)
        for n in range(self.backups - 1, 0, -1):
            if self._backup(n).exists():
                os.replace(self._backup(n), self._backup(n + 1))
        os.replace(self.path, self._backup(1))
