"""Append-only cost ledger of real model calls and the budget check (spec P0-05).

One JSON object per line in the ledger file (`cost_ledger_path`, by default one file shared by
every checkout): time (UTC), kind (`embedding`, `chat`), model, prompt/completion/total tokens
from the response's `usage`, and the cost in CNY from the price table in the service settings.
Callers add their own fields (profile, number of inputs).

The amount spent is a running total per file and process that reads only the lines appended
since the last check (spec P1-01). Requests in flight hold a reservation of their estimated
cost (`reserve`), so concurrent
requests in one process cannot together pass the stop. Other processes do not see these
reservations; with a shared ledger they can overshoot the stop by at most what they have in
flight at that moment.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from rag.config import ModelPrice, ServiceSettings

logger = logging.getLogger(__name__)

_PER_MILLION = 1_000_000

# Estimated cost of requests in flight, per resolved ledger path, shared by every CostLedger
# object of this process that writes the same file.
_RESERVED: dict[str, float] = {}
_RESERVED_LOCK = threading.Lock()

# Running total per resolved ledger path: (bytes already read, CNY in those bytes).
_TOTALS: dict[str, tuple[int, float]] = {}
_TOTALS_LOCK = threading.Lock()


def _line_cny(line: bytes, path: Path) -> float:
    if not line.strip():
        return 0.0
    try:
        row = json.loads(line)
    except ValueError:
        logger.warning("cost ledger %s: skipping unreadable line %r", path, line[:80])
        return 0.0
    return float(row.get("cny") or 0.0) if isinstance(row, dict) else 0.0


class BudgetExceeded(RuntimeError):  # noqa: N818 - the name the spec and plan use
    """Spent plus the estimated cost of the next job would pass `budget_stop_cny`."""


class UnknownModelPriceError(KeyError):
    """The model has no entry in the price table, so its cost cannot be recorded or checked."""

    def __str__(self) -> str:  # KeyError would print the message with quotes around it
        return str(self.args[0])


@dataclass
class ModelTotals:
    calls: int = 0
    tokens: int = 0
    cny: float = 0.0


@dataclass
class LedgerSummary:
    path: Path
    calls: int
    spent_cny: float
    budget_cny: float | None
    budget_stop_cny: float
    by_kind_model: dict[tuple[str, str], ModelTotals] = field(default_factory=dict)


class CostLedger:
    """Writes and reads one ledger file; all methods are safe across threads and processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        prices: Mapping[str, ModelPrice],
        budget_stop_cny: float,
        budget_cny: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.prices = dict(prices)
        self.budget_stop_cny = budget_stop_cny
        self.budget_cny = budget_cny
        self._thread_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: ServiceSettings | None = None) -> CostLedger:
        settings = settings or ServiceSettings()
        return cls(
            settings.cost_ledger_path,
            prices=settings.model_prices,
            budget_stop_cny=settings.budget_stop_cny,
            budget_cny=settings.budget_cny,
        )

    def price(self, model: str) -> ModelPrice:
        try:
            return self.prices[model]
        except KeyError:
            raise UnknownModelPriceError(
                f"no price for model {model!r}; add it to model_prices in the service settings "
                f"(known: {', '.join(sorted(self.prices)) or 'none'})"
            ) from None

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
        """Cost in CNY of a call with these token counts."""
        price = self.price(model)
        return (
            prompt_tokens * price.input_cny_per_m + completion_tokens * price.output_cny_per_m
        ) / _PER_MILLION

    def record(
        self,
        *,
        kind: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append one line for a real call and return it."""
        cny = self.estimate(model, prompt_tokens, completion_tokens)
        row: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": kind,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cny": round(cny, 10),
            **extra,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self._file_lock(), self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return row

    def entries(self) -> list[dict[str, Any]]:
        """All complete lines. A line cut off by a crash in mid-write is skipped with a warning."""
        if not self.path.exists():
            return []
        with self._thread_lock, self._file_lock():
            text = self.path.read_text(encoding="utf-8")
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("cost ledger %s: skipping unreadable line %d", self.path, number)
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def total_spent(self) -> float:
        """CNY spent according to the ledger file.

        The budget check runs before every request, also on an asyncio event loop (spec P1-01),
        so this keeps a running total per file in the process and reads only the bytes appended
        since the last call, by this or any other process. A line still being written counts
        once its newline is there. A file shorter than what was already read (rewritten rather
        than appended) is read again from the start.
        """
        key = self._key()
        with _TOTALS_LOCK:
            offset, total = _TOTALS.get(key, (0, 0.0))
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                _TOTALS.pop(key, None)
                return 0.0
            if size < offset:
                offset, total = 0, 0.0
            if size > offset:
                with self.path.open("rb") as fh:
                    fh.seek(offset)
                    new = fh.read(size - offset)
                complete = new[: new.rfind(b"\n") + 1]
                total += sum(_line_cny(line, self.path) for line in complete.splitlines())
                offset += len(complete)
            _TOTALS[key] = (offset, total)
            return total

    def reserved(self) -> float:
        """Estimated cost of this process's requests in flight against this ledger file."""
        with _RESERVED_LOCK:
            return _RESERVED.get(self._key(), 0.0)

    def check_budget(self, estimated_cny: float) -> float:
        """Return the amount spent so far; raise `BudgetExceeded` if spent + reserved +
        `estimated_cny` would pass the stop."""
        spent = self.total_spent()
        self._raise_if_over(spent, self.reserved(), estimated_cny)
        return spent

    @contextmanager
    def reserve(self, estimated_cny: float) -> Iterator[None]:
        """Hold `estimated_cny` against the stop while a request is in flight.

        The check and the reservation happen under one lock, so two threads cannot both pass a
        check that only one of them fits into. Record the call's real cost before leaving.
        """
        key = self._key()
        with _RESERVED_LOCK:
            held = _RESERVED.get(key, 0.0)
            self._raise_if_over(self.total_spent(), held, estimated_cny)
            _RESERVED[key] = held + estimated_cny
        try:
            yield
        finally:
            with _RESERVED_LOCK:
                left = _RESERVED.get(key, 0.0) - estimated_cny
                if left > 1e-12:
                    _RESERVED[key] = left
                else:
                    _RESERVED.pop(key, None)

    def _raise_if_over(self, spent: float, reserved: float, estimated_cny: float) -> None:
        if spent + reserved + estimated_cny > self.budget_stop_cny:
            raise BudgetExceeded(
                f"budget stop: spent {spent:.6f} CNY + reserved {reserved:.6f} CNY + "
                f"estimated {estimated_cny:.6f} CNY > stop {self.budget_stop_cny:.2f} CNY "
                f"(ledger {self.path}); switch to the fake models (EMBED_BACKEND=fake, "
                "CHAT_BACKEND=fake) and leave the real run for the owner"
            )

    def _key(self) -> str:
        return str(self.path.resolve())

    def summary(self) -> LedgerSummary:
        rows = self.entries()
        totals: dict[tuple[str, str], ModelTotals] = {}
        for row in rows:
            key = (str(row.get("kind", "?")), str(row.get("model", "?")))
            t = totals.setdefault(key, ModelTotals())
            t.calls += 1
            t.tokens += int(row.get("total_tokens") or 0)
            t.cny += float(row.get("cny") or 0.0)
        return LedgerSummary(
            path=self.path,
            calls=len(rows),
            spent_cny=sum(t.cny for t in totals.values()),
            budget_cny=self.budget_cny,
            budget_stop_cny=self.budget_stop_cny,
            by_kind_model=dict(sorted(totals.items())),
        )

    def _file_lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.path) + ".lock")


def total_spent(settings: ServiceSettings | None = None) -> float:
    """CNY spent so far according to the ledger in the settings."""
    return CostLedger.from_settings(settings).total_spent()


def check_budget(estimated_cny: float, settings: ServiceSettings | None = None) -> float:
    """Raise `BudgetExceeded` if spent + `estimated_cny` > `budget_stop_cny`, else return spent."""
    return CostLedger.from_settings(settings).check_budget(estimated_cny)
