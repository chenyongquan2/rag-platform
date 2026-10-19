"""The route rules and the list of hooks behind the graph's nodes (spec P3-01, P3-02).

The graph (`rag.agent.graph`) has a `clarify`, a `rewrite` and a `judge` node, the edges between
them and the guards around them (maximum rounds, deadline, token cap). What those nodes decide
is a function in `AgentHooks`:

    hook           | default does                                  | where
    ---------------+-----------------------------------------------+-------------------------
    decide_route   | rules: smalltalk, clear / vague / undecided,   | here, `rag.agent.rules`
                   | two-hop hint (no model)                        |
    route_fallback | asks the small model about what the rules      | `rag.agent.routing`
                   | left undecided (only then)                     |
    clarify        | the question to ask the caller (the node        | here
                   | stops the run with `interrupt`)                |
    rewrite        | standalone question, declarative query,        | `rag.agent.rewrite`
                   | sub-questions; round 2 is P3-03's              |
    judge          | the results are enough (stub)                  | P3-03

so adding behaviour is a change of one function, not of the graph. `decide_route` is the `route`
node's rule set: rules only, no model (design s8-1, s13-3). It returns signals; the node reads
them with the knowledge base's `agent` settings and asks `route_fallback` when it must.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag.agent.decisions import RewriteOutcome, RouteDecision, Verdict
from rag.agent.rewrite import QueryRewriter
from rag.agent.routing import route_with_model
from rag.agent.rules import clarity, default_clarification, multi_hop_hint
from rag.agent.session import HistoryTurn
from rag.answer.prompts import answer_language

if TYPE_CHECKING:
    from rag.agent.context import AgentContext
    from rag.agent.state import AgentState

__all__ = [
    "AgentHooks",
    "RewriteOutcome",
    "RouteDecision",
    "Verdict",
    "clarification_question",
    "decide_route",
    "stub_judge",
]


# A message that is only a greeting or thanks is not a question about the documents: nothing to
# retrieve and no model to call. Anything with more words is a question.
_SMALLTALK = re.compile(
    r"^[\W_]*(?:hi|hello|hey|thanks|thank you|thx|ok|okay|你好|您好|谢谢|多谢|好的)[\W_]*$",
    re.IGNORECASE,
)


def decide_route(question: str, *, kb: str, history: Sequence[HistoryTurn]) -> RouteDecision:
    """Rules of the `route` node. The knowledge base is the one the caller chose (picking one
    for a caller that names none is s8-3, not done here); retrieval is skipped for an empty
    message and for plain greetings and thanks. Otherwise the rules' signals: whether the
    question can be searched as it stands (`clarity`) and whether it may need two searches
    (`multi_hop`); what to do with them is the node's, with the knowledge base's settings."""
    text = question.strip()
    if not text:
        return RouteDecision(retrieve=False, kb=kb, reason="empty")
    if _SMALLTALK.match(text):
        return RouteDecision(retrieve=False, kb=kb, reason="smalltalk")
    return RouteDecision(
        retrieve=True,
        kb=kb,
        reason="rules",
        clarity=clarity(text, has_history=bool(history)),
        multi_hop=multi_hop_hint(text),
    )


# ---- the clarify hook and the stubs ----------------------------------------------------------


async def clarification_question(state: AgentState, ctx: AgentContext) -> str | None:
    """The question to ask the caller, when the route flagged the question as unclear: the
    one the small model wrote, else a fixed one in the question's language. No model call:
    the clarify node runs again when the caller answers, and must not spend tokens twice."""
    if not state.get("needs_clarification"):
        return None
    return state.get("clarify_question") or default_clarification(
        answer_language(state["question"], ctx.cfg.answer.language)
    )


async def stub_judge(state: AgentState, ctx: AgentContext) -> Verdict:
    """P3-03: are the chunks found so far enough for every sub-question? The stub says yes, so
    there is exactly one round."""
    return "sufficient"


@dataclass(frozen=True)
class AgentHooks:
    """The functions behind the nodes `route`, `clarify`, `rewrite` and `judge`."""

    decide_route: Callable[..., RouteDecision | Awaitable[RouteDecision]] = decide_route
    route_fallback: Callable[
        [AgentState, AgentContext, RouteDecision, frozenset[str]], Awaitable[RouteDecision]
    ] = route_with_model
    clarify: Callable[[AgentState, AgentContext], Awaitable[str | None]] = clarification_question
    rewrite: Callable[[AgentState, AgentContext], Awaitable[RewriteOutcome]] = field(
        default_factory=QueryRewriter
    )
    judge: Callable[[AgentState, AgentContext], Awaitable[str]] = stub_judge
