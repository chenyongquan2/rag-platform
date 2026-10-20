"""State of the ask_docs graph and the rules for its checkpoint thread (design s13-3; spec P3-01).

No `from __future__ import annotations` here: LangGraph reads the `Annotated[...]` reducers of
`AgentState` from the real annotations.

Rules that keep one request from seeing another's retrieved chunks:

- `thread_id(session_id, run_id)` is `"{session_id}:{run_id}"` with a run id made by the server:
  every run is its own checkpoint thread, whatever request id the caller sent. The session id
  has no colon, so two different pairs never give one id.
- Retrieval fields use a reducer that appends (`accumulate`): the answer can use the results of
  every round of this request. The `route` node starts every run by overwriting them with empty
  values (`Overwrite`), so a run on a thread that already has state (a resumed thread, P3-02,
  or a caller that passes a run id twice) accumulates only its own results.
- The conversation does not live in the state's checkpoint: the turns of the session are stored
  per session key (`rag.agent.session`) and given to the graph as `history` input.
"""

import re
from typing import Annotated, Any, Literal, Self, TypedDict

from pydantic import BaseModel, Field

from rag.agent.session import HistoryTurn
from rag.types import (
    Answer,
    AnswerSentence,
    AnswerStatus,
    AskResult,
    Chunk,
    Citation,
    CitationCheck,
    ContextPiece,
    ErrorCategory,
    RetrievalResult,
)

SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
NO_SESSION = "-"


def valid_session_id(value: object) -> bool:
    """`-` is the no-session name in a thread id, so a real session may not use it."""
    return isinstance(value, str) and value != NO_SESSION and SESSION_ID.match(value) is not None


def thread_id(session_id: str | None, run_id: str) -> str:
    """The checkpoint thread of one run; `-` stands for a call without a session."""
    if session_id is None:
        return f"{NO_SESSION}:{run_id}"
    if not valid_session_id(session_id):
        raise ValueError("session_id must be 1 to 64 letters, digits, dot, dash or underscore")
    return f"{session_id}:{run_id}"


QueryKind = Literal["original", "multi_query", "hyde"]


class SearchQuery(BaseModel):
    """One search of a round: the text sent to the retrieval layer, the sub-question it is
    for (`owner`, the key the judge and the answer's context budget group by), and how the
    text was made. A `hyde` text is a model-written passage, used as a query only: it is never
    given to the judge or the answer as if it were a source (s8-2, spec P3-03)."""

    text: str
    owner: str
    kind: QueryKind = "original"


class RetrievalRound(BaseModel):
    """What one search of one sub-question returned (the part of `RetrievalResult` the graph
    keeps). Several of these are appended per request: one per search of each round. In round
    one `sub_question` is the query itself; later rounds search other phrasings (`query`) of
    the same `sub_question`."""

    round: int = Field(ge=1)
    sub_question: str
    query: str = ""
    """The text searched, when it is not the sub-question (a phrasing, a HyDE passage)."""
    kind: QueryKind = "original"
    """How `query` was made; a `hyde` text is generated and is kept out of prompts and spans."""
    chunks: list[Chunk] = Field(default_factory=list)
    below_threshold: list[Chunk] = Field(default_factory=list)
    """With `not_found`: the nearest chunks the retrieval layer kept apart (s10-2)."""
    not_found: bool = False
    degraded: bool = False
    rerank_timeout: bool = False
    index_version: str = ""
    data_generation: int = 0
    config_version: str = ""
    top_rerank_score: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        round_no: int,
        sub_question: str,
        result: RetrievalResult,
        *,
        query: str = "",
        kind: QueryKind = "original",
    ) -> Self:
        scores = [h.score for h in result.rerank if h.score is not None]
        return cls(
            round=round_no,
            sub_question=sub_question,
            query=query if query != sub_question else "",
            kind=kind,
            chunks=list(result.chunks),
            below_threshold=list(result.below_threshold),
            not_found=result.not_found,
            degraded=result.degraded,
            rerank_timeout=result.rerank_timeout,
            index_version=result.index_version,
            data_generation=result.data_generation,
            config_version=result.config_version,
            top_rerank_score=max(scores) if scores else None,
            timings_ms=dict(result.timings_ms),
        )


def accumulate(old: list[Any] | None, new: list[Any] | None) -> list[Any]:
    """Reducer of the retrieval fields: results of later rounds are appended to earlier ones."""
    return [*(old or []), *(new or [])]


def add_timings(old: dict[str, float] | None, new: dict[str, float] | None) -> dict[str, float]:
    """Reducer of the step timings: a step that runs again (a second round) adds its time."""
    merged = dict(old or {})
    for step, ms in (new or {}).items():
        merged[step] = round(merged.get(step, 0.0) + ms, 3)
    return merged


class AgentState(TypedDict, total=False):
    """Everything the nodes pass to each other. Input: `request_id`, `session_id`, `kb`,
    `question`, `history`."""

    request_id: str
    session_id: str | None
    kb: str
    question: str
    history: list[HistoryTurn]
    # Set by route (rules, and the small model when the rules cannot decide), clarify and the
    # rewrite hook (P3-02).
    needs_retrieval: bool
    needs_clarification: bool
    two_hop: bool
    clarify_question: str | None
    """The question to ask the caller (set by route, read by clarify)."""
    clarification: str | None
    """Set only on the way out of a run that stopped at the interrupt (the question asked)."""
    clarification_answer: str | None
    clarified: bool
    """The caller was asked and answered: at most once per request."""
    clarified_question: str | None
    """The question and the caller's answer as one text, searched and answered instead of the
    question alone."""
    tokens_before_clarify: int
    """Model tokens spent before the run stopped, so the resumed call keeps counting."""
    standalone_question: str | None
    sub_questions: list[str]
    """The search queries of the first round (rewrite) or of the next one (P3-03)."""
    missing: list[str]
    """The sub-questions (first-round queries) the last judgement found without grounds: set by
    the judge, read by the rewrite hook of the next round and the judge after it."""
    sufficiency: str | None
    """The judge's last verdict: enough, partly_enough or not_enough (None: no judge)."""
    search_plan: list[SearchQuery]
    """The searches of the next round, set by the rewrite hook from round two on (empty: search
    `sub_questions`, as round one does)."""
    fallbacks: Annotated[list[str], accumulate]
    """Steps that fell back to the rules or the original question: `route:timeout`,
    `rewrite:parse_failed` ... (any entry sets the result's `degraded`)."""
    # Retrieval rounds of this request (reducers; cleared by route).
    round: int
    rounds: Annotated[list[RetrievalRound], accumulate]
    timings_ms: Annotated[dict[str, float], add_timings]
    loop: bool
    stop_reason: str | None
    answer: Answer | None
    result: AskResult | None


def working_question(state: AgentState) -> str:
    """The question the later steps work on: the one the caller asked, with the clarification
    added when the caller was asked back. (The standalone question of the rewrite, when there
    is one, is what the answer layer gets.)"""
    return state.get("clarified_question") or state["question"]


# Types the state (and so a checkpoint) may hold. The serializer refuses a type that is not
# listed here, so a checkpoint file cannot make it import anything else.
MSGPACK_ALLOWED: tuple[type, ...] = (
    Answer,
    AnswerSentence,
    AnswerStatus,
    AskResult,
    Chunk,
    Citation,
    CitationCheck,
    ContextPiece,
    ErrorCategory,
    HistoryTurn,
    RetrievalRound,
    SearchQuery,
)
