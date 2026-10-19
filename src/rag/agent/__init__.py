"""The ask_docs agent graph (spec P3-01, design s13-2, s13-3, s4-2).

    result = await ask_docs(question, cfg=cfg, retriever=pipeline, settings=settings,
                            runtime=runtime, session_id="s1")   # -> rag.types.AskResult

Our own LangGraph `StateGraph`: route (rules) -> rewrite -> retrieve -> judge -> answer ->
verify, one checkpoint thread per request (`session_id:run_id`), chat history per session,
requests of one session one after another, a total deadline, a recursion limit and a token cap.
Rewrite, clarify and the sufficiency judge are stubs (`AgentHooks`) that P3-02 and P3-03 fill in.
"""

from rag.agent.budget import RequestTokenCapExceeded, TokenBudget, guard_model
from rag.agent.context import AgentContext, AgentRetriever
from rag.agent.graph import build_graph
from rag.agent.hooks import AgentHooks, RewriteOutcome, RouteDecision, Verdict, decide_route
from rag.agent.runtime import AgentDeadlineExceeded, AgentRuntime, ask_docs
from rag.agent.session import (
    HistoryTurn,
    MemorySessionStore,
    SessionLocks,
    SessionStore,
    SqliteSessionStore,
)
from rag.agent.state import AgentState, RetrievalRound, thread_id, valid_session_id

__all__ = [
    "AgentContext",
    "AgentDeadlineExceeded",
    "AgentHooks",
    "AgentRetriever",
    "AgentRuntime",
    "AgentState",
    "HistoryTurn",
    "MemorySessionStore",
    "RequestTokenCapExceeded",
    "RetrievalRound",
    "RewriteOutcome",
    "RouteDecision",
    "SessionLocks",
    "SessionStore",
    "SqliteSessionStore",
    "TokenBudget",
    "Verdict",
    "ask_docs",
    "build_graph",
    "decide_route",
    "guard_model",
    "thread_id",
    "valid_session_id",
]
