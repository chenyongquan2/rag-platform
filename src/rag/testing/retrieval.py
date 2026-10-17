"""A scripted retriever for eval and tool tests (implements `rag.eval.retrieval.Retriever`)."""

from __future__ import annotations

from collections.abc import Mapping

from rag.types import RetrievalResult


class FakeRetriever:
    """Returns the result scripted for each question text; an exception is raised instead.

    The final list is cut to `top_k` like a real retriever's; route lists are returned as
    scripted. Every call is recorded in `calls` as `(question, kb, top_k)`.
    """

    def __init__(self, results: Mapping[str, RetrievalResult | Exception]) -> None:
        self.results = dict(results)
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, question: str, kb: str, top_k: int) -> RetrievalResult:
        self.calls.append((question, kb, top_k))
        if question not in self.results:
            raise KeyError(f"FakeRetriever has no result for question {question!r}")
        result = self.results[question]
        if isinstance(result, Exception):
            raise result
        return result.model_copy(update={"chunks": result.chunks[:top_k]})
