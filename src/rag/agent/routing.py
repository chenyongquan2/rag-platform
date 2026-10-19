"""The small model's part of the route step (design s13-3, s8-1; spec P3-02).

The route node asks the rules first (`rag.agent.rules`). Only when a signal is `undecided`
(the question points at something it does not name) or `maybe` (it joins two parts), and the
knowledge base allows it (`agent.route_model`), this hook asks the small model one structured
question: must the caller be asked, and does the question need two separate searches. A failed
call (timeout, error, garbage, token cap) leaves the rules' signals as they were, so nothing is
asked back and nothing is split, and sets `fallback` so the result says so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag.agent.decisions import RouteDecision
from rag.agent.helper import HelperFailed, ask_helper, history_lines
from rag.agent.state import working_question

if TYPE_CHECKING:
    from rag.agent.context import AgentContext
    from rag.agent.state import AgentState

CLARIFY = "clarify"
# The longest question asked back that the model may write: it goes to the caller, the
# checkpoint and the history.
CLARIFY_QUESTION_CHARS = 300
TWO_HOP = "two_hop"


class RouteJudgement(BaseModel):
    """What to do with a question before searching the documentation."""

    needs_clarification: bool = Field(
        default=False,
        description="True only if the question cannot be searched without asking the user: it "
        "points at something it does not name and the conversation does not say what.",
    )
    clarifying_question: str = Field(
        default="",
        description="One short question that asks the user for the missing detail, in the "
        "language of the user's question. Empty when needs_clarification is false.",
    )
    two_hop: bool = Field(
        default=False,
        description="True if answering needs two or more separate lookups, for example a "
        "comparison, or a question whose second part depends on the first.",
    )


_SYSTEM = (
    "You check a user's question before it is searched in a documentation knowledge base. "
    "Decide only what the fields ask. Do not answer the question. "
    "Never translate: write the clarifying question in the language of the user's question."
)


def _messages(state: AgentState, ctx: AgentContext, wants: frozenset[str]) -> list[BaseMessage]:
    asked = []
    if CLARIFY in wants:
        asked.append("needs_clarification and clarifying_question")
    if TWO_HOP in wants:
        asked.append("two_hop")
    lines = [f"Knowledge base: {ctx.cfg.description or ctx.cfg.kb}"]
    history = history_lines(state, ctx)
    if history:
        lines.append("Earlier in the conversation:")
        lines.extend(history)
    lines.append(f"Question: {working_question(state)}")
    lines.append(f"Fill in: {', '.join(asked)}.")
    return [SystemMessage(_SYSTEM), HumanMessage("\n".join(lines))]


async def route_with_model(
    state: AgentState, ctx: AgentContext, decision: RouteDecision, wants: frozenset[str]
) -> RouteDecision:
    """`decision` with the model's answer to the questions in `wants` (`clarify`, `two_hop`)."""
    try:
        judgement = await ask_helper(ctx, RouteJudgement, _messages(state, ctx, wants))
    except HelperFailed as exc:
        return replace(decision, fallback=exc.reason)
    update: dict[str, object] = {"reason": "rules+model"}
    if CLARIFY in wants:
        ask = judgement.needs_clarification
        update["needs_clarification"] = ask
        question = judgement.clarifying_question.strip()[:CLARIFY_QUESTION_CHARS]
        update["clarify_question"] = (question or None) if ask else None
    if TWO_HOP in wants:
        update["two_hop"] = judgement.two_hop
    return replace(decision, **update)
