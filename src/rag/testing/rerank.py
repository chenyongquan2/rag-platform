"""A scripted cross-encoder for rerank tests (spec P0-08), never the real model.

`FakeCrossEncoder` implements `rag.retrieve.rerank.PairScorer`. Scores come from a dict keyed by
the passage text (or a function of question and passage), with `default` for anything else.
Time: `per_pair_s` advances a `ManualClock` (deterministic deadline tests), `sleep_s` really
sleeps per batch (semaphore and queueing tests), `gate` blocks a batch until it is set (only
for passages in `gate_texts` when given). `batches`, `started`, `active` and `max_active`
record what happened.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence


class ManualClock:
    """A monotonic clock that only moves when told to (`advance`)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class FakeCrossEncoder:
    def __init__(
        self,
        scores: Mapping[str, float] | Callable[[str, str], float] | None = None,
        *,
        default: float = 0.5,
        clock: ManualClock | None = None,
        per_pair_s: float = 0.0,
        sleep_s: float = 0.0,
        gate: threading.Event | None = None,
        gate_texts: Collection[str] | None = None,
    ) -> None:
        self.scores = scores if scores is not None else {}
        self.default = default
        self.clock = clock
        self.per_pair_s = per_pair_s
        self.sleep_s = sleep_s
        self.gate = gate
        self.gate_texts = set(gate_texts) if gate_texts is not None else None
        self.batches: list[list[tuple[str, str]]] = []
        self.started = threading.Event()
        """Set when the first batch starts (before any gate or sleep)."""
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _score(self, question: str, text: str) -> float:
        if callable(self.scores):
            return float(self.scores(question, text))
        return float(self.scores.get(text, self.default))

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            gated = self.gate_texts is None or any(t in self.gate_texts for _, t in pairs)
            if self.gate is not None and gated:
                self.gate.wait(timeout=30)
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.clock is not None:
                self.clock.advance(self.per_pair_s * len(pairs))
            with self._lock:
                self.batches.append(list(pairs))
            return [self._score(q, t) for q, t in pairs]
        finally:
            with self._lock:
                self.active -= 1
