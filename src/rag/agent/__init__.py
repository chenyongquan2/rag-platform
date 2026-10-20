"""The ask_docs agent graph (spec P3-01, design s13-2, s13-3, s4-2).

    result = await ask_docs(question, cfg=cfg, retriever=pipeline, settings=settings,
                            runtime=runtime, session_id="s1")   # -> rag.types.AskResult

Our own LangGraph `StateGraph`: route (rules) -> rewrite -> retrieve -> judge -> answer ->
verify, one checkpoint thread per request (`session_id:run_id`), chat history per session,
requests of one session one after another, a total deadline, a recursion limit and a token cap.
Rewrite (standalone question, declarative query, sub-questions), the small-model fallback of the
route and the clarify step (`interrupt`, resumed with a token) are P3-02. The sufficiency judge
(enough / partly enough / not enough, rules first, small model second) and the correction rounds
(multi-query, HyDE, then a real "not found") are P3-03 (`rag.agent.judge`,
`rag.agent.correction`).
"""

from rag.agent.budget import RequestTokenCapExceeded, TokenBudget, guard_model
from rag.agent.context import AgentContext, AgentRetriever
from rag.agent.graph import build_graph
from rag.agent.helper import HelperFailed, ask_helper
from rag.agent.hooks import (
    AgentHooks,
    JudgeOutcome,
    RewriteOutcome,
    RouteDecision,
    Verdict,
    decide_route,
)
from rag.agent.judge import SufficiencyJudge, judge_rounds
from rag.agent.resume import MemoryResumeStore, ResumeStore, ResumeTokenInvalid, SqliteResumeStore
from rag.agent.rewrite import QueryRewriter, RewritePlan
from rag.agent.routing import RouteJudgement, route_with_model
from rag.agent.runtime import AgentDeadlineExceeded, AgentRuntime, ask_docs
from rag.agent.session import (
    HistoryTurn,
    MemorySessionStore,
    SessionLocks,
    SessionStore,
    SqliteSessionStore,
)
from rag.agent.state import (
    AgentState,
    RetrievalRound,
    SearchQuery,
    thread_id,
    valid_session_id,
)

__all__ = [
    "AgentContext",
    "AgentDeadlineExceeded",
    "AgentHooks",
    "AgentRetriever",
    "AgentRuntime",
    "AgentState",
    "HelperFailed",
    "HistoryTurn",
    "JudgeOutcome",
    "MemoryResumeStore",
    "MemorySessionStore",
    "QueryRewriter",
    "RequestTokenCapExceeded",
    "ResumeStore",
    "ResumeTokenInvalid",
    "RetrievalRound",
    "RewriteOutcome",
    "RewritePlan",
    "RouteDecision",
    "RouteJudgement",
    "SearchQuery",
    "SessionLocks",
    "SessionStore",
    "SqliteResumeStore",
    "SqliteSessionStore",
    "SufficiencyJudge",
    "TokenBudget",
    "Verdict",
    "ask_docs",
    "ask_helper",
    "build_graph",
    "decide_route",
    "guard_model",
    "judge_rounds",
    "route_with_model",
    "thread_id",
    "valid_session_id",
]
