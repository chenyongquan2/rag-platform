"""State of the ask_docs graph and the rules for its checkpoint thread (design s13-3; spec P3-01).

No `from __future__ import annotations` here: LangGraph reads the `Annotated[...]` reducers of
`AgentState` from the real annotations.

Rules that keep one request from seeing another's retrieved chunks:

- `thread_id(session_id, request_id)` is `"{session_id}:{request_id}"`: every request is its own
  checkpoint thread. The session id has no colon, so two different pairs never give one id.
- Retrieval fields use a reducer that appends (`accumulate`): the answer can use the results of
  every round of this request. The `route` node starts every run by overwriting them with empty
  values (`Overwrite`), so a run on a thread that already has state (a resumed thread, P3-02,
  or a caller that reuses a request id) accumulates only its own results.
- The conversation does not live in the state's checkpoint: the turns of the session are stored
  per session id (`rag.agent.session`) and given to the graph as `history` input.
"""

import re
from typing import Annotated, Any, Self, TypedDict

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
    return isinstance(value, str) and SESSION_ID.match(value) is not None


def thread_id(session_id: str | None, request_id: str) -> str:
    """The checkpoint thread of one request; `-` stands for a call without a session."""
    if session_id is None:
        return f"{NO_SESSION}:{request_id}"
    if not valid_session_id(session_id):
        raise ValueError("session_id must be 1 to 64 letters, digits, dot, dash or underscore")
    return f"{session_id}:{request_id}"


class RetrievalRound(BaseModel):
    """What one search of one sub-question returned (the part of `RetrievalResult` the graph
    keeps). Several of these are appended per request: one per sub-question and round."""

    round: int = Field(ge=1)
    sub_question: str
    chunks: list[Chunk] = Field(default_factory=list)
    not_found: bool = False
    degraded: bool = False
    rerank_timeout: bool = False
    index_version: str = ""
    data_generation: int = 0
    config_version: str = ""
    top_rerank_score: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, round_no: int, sub_question: str, result: RetrievalResult) -> Self:
        scores = [h.score for h in result.rerank if h.score is not None]
        return cls(
            round=round_no,
            sub_question=sub_question,
            chunks=list(result.chunks),
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
    # Set by route (rules) and the rewrite hook (P3-02).
    needs_retrieval: bool
    needs_clarification: bool
    clarification: str | None
    standalone_question: str | None
    sub_questions: list[str]
    # Retrieval rounds of this request (reducers; cleared by route).
    round: int
    rounds: Annotated[list[RetrievalRound], accumulate]
    timings_ms: Annotated[dict[str, float], add_timings]
    loop: bool
    stop_reason: str | None
    answer: Answer | None
    result: AskResult | None


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
)
