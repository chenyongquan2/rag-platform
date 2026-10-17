"""A point in time by which a query must be answered (design s16-2).

The caller's total time limit (`deadline_ms`, s16-1) becomes one `Deadline` at the start of the
request and is passed down: the query embedding waits at most until the time the search needs,
the search gets what is left, and later steps (rerank, P0-08) stop by it too.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Deadline:
    """`at` is a reading of `clock` (monotonic seconds) after which the request has failed."""

    at: float
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    @classmethod
    def after_ms(cls, ms: float, *, clock: Callable[[], float] = time.monotonic) -> Deadline:
        if ms <= 0:
            raise ValueError(f"a deadline needs a positive time limit, got {ms} ms")
        return cls(at=clock() + ms / 1000.0, clock=clock)

    def remaining_s(self) -> float:
        """Seconds left, 0 when the deadline has passed (never negative)."""
        return max(0.0, self.at - self.clock())

    def remaining_ms(self) -> float:
        return self.remaining_s() * 1000.0

    @property
    def expired(self) -> bool:
        return self.remaining_s() <= 0.0
