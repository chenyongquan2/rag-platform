"""Ledger lines written while one piece of eval work runs (spec P1-04).

    with collect_ledger_rows() as rows:
        answer = await generate_answer(...)
    usage = Usage.from_rows(rows)

The rows are kept per context: an asyncio task or a `to_thread` call started inside the block
sees the same list, concurrent tasks with their own block do not mix. The ledger listener is
registered once per process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from rag.cost import add_record_listener

_ROWS: ContextVar[list[dict[str, Any]] | None] = ContextVar("eval_ledger_rows", default=None)
_LOCK = threading.Lock()
_installed = False


def _collect(row: dict[str, Any]) -> None:
    rows = _ROWS.get()
    if rows is not None:
        rows.append(row)


def _install() -> None:
    global _installed
    with _LOCK:
        if not _installed:
            add_record_listener(_collect)
            _installed = True


@contextmanager
def collect_ledger_rows() -> Iterator[list[dict[str, Any]]]:
    _install()
    rows: list[dict[str, Any]] = []
    token = _ROWS.set(rows)
    try:
        yield rows
    finally:
        _ROWS.reset(token)
