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
    helper_model: Any
    """The small model for the route and rewrite steps (P3-02): a `BaseChatModel`, or a function
    that makes it (called when a step needs it; a failure there is that step's fallback)."""
    hooks: AgentHooks
    request_id: str
    session_id: str | None
    thread_id: str
    acl_groups: tuple[str, ...] = ("public",)
    filters: Mapping[str, Any] | None = None
    used_history: bool = False
    """Earlier turns of the session were loaded into this run (set by the runtime)."""
    resumed: bool = False
    """This call continues a run that stopped to ask the caller (set by the runtime)."""
    judge_model: Any = None
    """The model of the sufficiency judge (P3-03), same kinds of value as `helper_model`; None:
    the helper model serves."""

    @property
    def reserve_ms(self) -> int:
        """Time kept for writing the answer: no round starts, and a search gets no time, that
        would eat into it. At most half the total, so a short deadline still searches."""
        return min(self.settings.agent_answer_reserve_ms, self.total_ms // 2)

    def search_time_ms(self) -> float:
        """Time a search could get now: what is left minus the answer's time (may be negative)."""
        return self.deadline.remaining_ms() - self.reserve_ms

    def can_search(self) -> bool:
        """Whether a search of round two or later has the least time worth spending
        (`agent_search_min_ms`, spec P3-06): with less it would embed the question and then have
        no time left to rerank."""
        return self.search_time_ms() >= self.settings.agent_search_min_ms

    def retrieval_deadline_ms(self) -> int:
        """Time limit given to one search: what is left minus the answer's time, and never more
        than a plain search_docs would get."""
        left = self.deadline.remaining_ms() - self.reserve_ms
        return max(1, int(min(self.settings.search_deadline_ms, left)))

    def answer_model(self) -> Any:
        """The answer model, counted against the token cap (it may use all of what is left)."""
        model = self.model if hasattr(self.model, "astream") else self.model()
        return guard_model(model, self.budget)

    @staticmethod
    def model_instance(source: Any) -> Any:
        """A chat model from a model or from the function that makes one."""
        return source if hasattr(source, "with_structured_output") else source()

    def helper_model_instance(self) -> Any:
        """The small model (not yet counted against the cap: `guard` does that)."""
        return self.model_instance(self.helper_model)

    def judge_model_source(self) -> Any:
        """The judge's model, or the function that makes it: its own when there is one."""
        return self.judge_model if self.judge_model is not None else self.helper_model

    def guard(self, model: Any, *, auxiliary: bool = True) -> Any:
        """`model` (a chat model, or a runnable built on one) counted against the token cap. A
        call before the answer (`auxiliary`) leaves the answer's tokens
        (`agent_answer_reserve_tokens`) untouched."""
        reserve = self.settings.agent_answer_reserve_tokens if auxiliary else 0
        return guard_model(model, self.budget, reserve=reserve)
