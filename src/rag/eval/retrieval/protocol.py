"""What the eval harness needs from a retriever (spec P0-10).

The retrieval chain of P0-07/P0-08 (and anything else, e.g. a baseline or a fake) is evaluated
through this protocol only, so the harness does not depend on how retrieval is built.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag.types import Chunk, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, question: str, kb: str, top_k: int) -> RetrievalResult:
        """Search knowledge base `kb`; called with keyword arguments `kb` and `top_k`.

        - `chunks`: the final list, best first, at most `top_k`. The harness scores it in list
          order and never re-sorts by score, so ties must already be broken (deterministically).
        - `routes`: each route's own ranked list before fusion (`bm25`, `vector`), for the
          per-route metrics; may be empty. A failed route is left out and `degraded` is set.
        - `index_version`, `data_generation`: recorded in the run (placeholders are fine).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    def __call__(self, question: str, candidates: Sequence[Chunk]) -> Sequence[Chunk]:
        """Return `candidates` reordered best first (it may drop some, never add any)."""
        ...
