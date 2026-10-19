"""Rules of the `route` node and the extension points P3-02 and P3-03 fill in (spec P3-01).

The graph (`rag.agent.graph`) is complete: it has a `clarify`, a `rewrite` and a `judge` node,
the edges between them and the guards around them (maximum rounds, deadline, token cap). What
those nodes decide is a function in `AgentHooks`; here the three are stubs with no behaviour:

    hook      | stub does                      | filled in by
    ----------+--------------------------------+---------------------------------------------
    clarify   | no question to ask             | P3-02: `interrupt` with a question, resume
    rewrite   | no rewriting, one sub-question | P3-02: standalone question, sub-questions;
              |                                | P3-03: another wording for the next round
    judge     | the results are enough         | P3-03: sufficient / insufficient per sub-question

so adding one is a change of this table's function (and, for `clarify`, of the rule that sends
the run there), not of the graph. `decide_route` is the `route` node's rule set: rules only,
no model (design s8-1, s13-3); P3-02 adds the small-model fallback for what rules cannot decide.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rag.agent.session import HistoryTurn

if TYPE_CHECKING:
    from rag.agent.context import AgentContext
    from rag.agent.state import AgentState

Verdict = Literal["sufficient", "insufficient"]


@dataclass(frozen=True)
class RouteDecision:
    """What the rules decided: whether to retrieve at all, which knowledge base, whether the
    question must be clarified first (P3-02), and which rule decided."""

    retrieve: bool = True
    kb: str = ""
    needs_clarification: bool = False
    reason: str = "rules"


@dataclass(frozen=True)
class RewriteOutcome:
    """What a rewrite changed: the question made standalone (P3-02) and the sub-questions to
    search, one search each. Empty means "keep what the state has"."""

    standalone_question: str | None = None
    sub_questions: tuple[str, ...] = ()


# A message that is only a greeting or thanks is not a question about the documents: nothing to
# retrieve and no model to call. Anything with more words is a question.
_SMALLTALK = re.compile(
    r"^[\W_]*(?:hi|hello|hey|thanks|thank you|thx|ok|okay|你好|您好|谢谢|多谢|好的)[\W_]*$",
    re.IGNORECASE,
)


def decide_route(question: str, *, kb: str, history: Sequence[HistoryTurn]) -> RouteDecision:
    """Rules of the `route` node. The knowledge base is the one the caller chose (picking one
    for a caller that names none is s8-3, not done here); retrieval is skipped for an empty
    message and for plain greetings and thanks."""
    text = question.strip()
    if not text:
        return RouteDecision(retrieve=False, kb=kb, reason="empty")
    if _SMALLTALK.match(text):
        return RouteDecision(retrieve=False, kb=kb, reason="smalltalk")
    return RouteDecision(retrieve=True, kb=kb, reason="rules")


# ---- stubs: extension points for P3-02 and P3-03 (no behaviour) -----------------------------


async def stub_clarify(state: AgentState, ctx: AgentContext) -> str | None:
    """P3-02: return the question to ask the caller when the question is unclear. The stub
    asks nothing, so the run goes on to `rewrite`."""
    return None


async def stub_rewrite(state: AgentState, ctx: AgentContext) -> RewriteOutcome:
    """P3-02 / P3-03: the standalone question and sub-questions (first round), or another
    wording of what is still missing (later rounds). The stub changes nothing: the question
    is searched as asked."""
    return RewriteOutcome()


async def stub_judge(state: AgentState, ctx: AgentContext) -> Verdict:
    """P3-03: are the chunks found so far enough for every sub-question? The stub says yes, so
    there is exactly one round."""
    return "sufficient"


@dataclass(frozen=True)
class AgentHooks:
    """The functions behind the nodes `route`, `clarify`, `rewrite` and `judge`."""

    decide_route: Callable[..., RouteDecision | Awaitable[RouteDecision]] = decide_route
    clarify: Callable[[AgentState, AgentContext], Awaitable[str | None]] = stub_clarify
    rewrite: Callable[[AgentState, AgentContext], Awaitable[RewriteOutcome]] = stub_rewrite
    judge: Callable[[AgentState, AgentContext], Awaitable[str]] = stub_judge
