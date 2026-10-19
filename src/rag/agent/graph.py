"""The ask_docs `StateGraph` (design s13-2, s13-3, s4-2; spec P3-01).

    START -> route
    route    -> clarify   (the rules flag the question as unclear; P3-02)
             -> answer    (no retrieval: a greeting)
             -> rewrite   (otherwise)
    clarify  -> verify    (a question for the caller: the run ends)
             -> rewrite   (nothing to ask)
    rewrite  -> retrieve
             -> answer    (token cap reached while rewriting)
    retrieve -> judge
    judge    -> rewrite   (another round: not enough found, and no limit reached)
             -> answer    (enough, or max rounds, deadline or token cap reached)
    answer -> verify -> END

Our own graph rather than `create_agent`: the steps are fixed and only the number of rounds
varies, so rounds, cost and the path on failure are decided by these edges (s13-2). The edge
functions read plain state fields (`needs_clarification`, `needs_retrieval`, `clarification`,
`stop_reason`, `loop`) that the nodes set, so a hook that fills in a stub changes what it sets,
not the wiring.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag.agent.context import AgentContext
from rag.agent.nodes import (
    answer_node,
    clarify_node,
    judge_node,
    retrieve_node,
    rewrite_node,
    route_node,
    verify_node,
)
from rag.agent.state import AgentState


def _after_route(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarify"
    return "rewrite" if state.get("needs_retrieval", True) else "answer"


def _after_clarify(state: AgentState) -> str:
    if state.get("clarification"):
        return "verify"  # the run ends with the question for the caller
    return "rewrite" if state.get("needs_retrieval", True) else "answer"


def _after_rewrite(state: AgentState) -> str:
    return "answer" if state.get("stop_reason") else "retrieve"


def _after_judge(state: AgentState) -> str:
    return "rewrite" if state.get("loop") else "answer"


def build_graph(checkpointer: Any) -> Any:
    """The compiled graph over `checkpointer` (a LangGraph saver). Cheap to build: the service
    compiles one per request, over the saver it just opened."""
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("route", route_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("judge", judge_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", _after_route, {"clarify": "clarify", "rewrite": "rewrite", "answer": "answer"}
    )
    graph.add_conditional_edges(
        "clarify", _after_clarify, {"verify": "verify", "rewrite": "rewrite", "answer": "answer"}
    )
    graph.add_conditional_edges(
        "rewrite", _after_rewrite, {"retrieve": "retrieve", "answer": "answer"}
    )
    graph.add_edge("retrieve", "judge")
    graph.add_conditional_edges("judge", _after_judge, {"rewrite": "rewrite", "answer": "answer"})
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=checkpointer)
