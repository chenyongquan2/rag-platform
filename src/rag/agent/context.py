"""What the nodes of one run need besides the state (spec P3-01).

`AgentContext` is passed to `graph.ainvoke(..., context=...)`. It holds things that cannot go
into a checkpoint (the retriever, the model, the clock) or that belong to this call only (the
deadline, the token budget, the caller's ACL groups). It is not stored, so a resumed thread
(P3-02) gets a fresh one from the new call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag.agent.budget import TokenBudget, guard_model
from rag.agent.hooks import AgentHooks
from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.retrieve.deadline import Deadline
from rag.types import RetrievalResult


class AgentRetriever(Protocol):
    """The part of a knowledge base's `SearchPipeline` the graph uses (blocking call)."""

    def retrieve(
        self,
        question: str,
        kb: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> RetrievalResult: ...


@dataclass
class AgentContext:
    settings: ServiceSettings
    cfg: KnowledgeBaseConfig
    retriever: AgentRetriever
    deadline: Deadline
    """The whole run's time limit, started when the call came in (waiting for the session lock
    counts)."""
    total_ms: int
    budget: TokenBudget
    model: Any
    """The chat model that writes the answer (`BaseChatModel`), or a function that makes it
    (called only when there is a fragment to answer from); guarded by `budget` when used."""
    hooks: AgentHooks
    request_id: str
    session_id: str | None
    thread_id: str
    acl_groups: tuple[str, ...] = ("public",)
    filters: Mapping[str, Any] | None = None
    used_history: bool = False
    """Earlier turns of the session were loaded into this run (set by the runtime)."""

    @property
    def reserve_ms(self) -> int:
        """Time kept for writing the answer: no round starts, and a search gets no time, that
        would eat into it. At most half the total, so a short deadline still searches."""
        return min(self.settings.agent_answer_reserve_ms, self.total_ms // 2)

    def retrieval_deadline_ms(self) -> int:
        """Time limit given to one search: what is left minus the answer's time, and never more
        than a plain search_docs would get."""
        left = self.deadline.remaining_ms() - self.reserve_ms
        return max(1, int(min(self.settings.search_deadline_ms, left)))

    def answer_model(self) -> Any:
        """The answer model, counted against the token cap (it may use all of what is left)."""
        model = self.model if hasattr(self.model, "astream") else self.model()
        return guard_model(model, self.budget)

    def guard(self, model: Any, *, auxiliary: bool = True) -> Any:
        """`model` counted against the token cap. A call before the answer (`auxiliary`) leaves
        the answer's tokens (`agent_answer_reserve_tokens`) untouched."""
        reserve = self.settings.agent_answer_reserve_tokens if auxiliary else 0
        return guard_model(model, self.budget, reserve=reserve)
