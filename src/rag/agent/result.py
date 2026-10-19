"""The `AskResult` of a run: built from the final state, or from the last checkpoint of a run that
was cut off (design s4-2, s13-3, s16-8; spec P3-01).

`compose_result` is the one place that turns state into the caller's result, so the `verify`
node and the cut-off path (total deadline, recursion limit) set the same fields the same way:
the retrieval flags (`degraded`, `rerank_timeout`, index version, data generation) are copied
from the rounds, because the answer layer does not know them (spec P1-02 handoff); the
answer's own failure flags are kept; the stop reason becomes the flag "not confirmed that
everything was searched" with a notice in the answer's language.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.agent.context import AgentContext
from rag.agent.state import RetrievalRound
from rag.answer.prompts import AnswerLanguage, answer_language
from rag.types import (
    Answer,
    AnswerSentence,
    AnswerStatus,
    AskResult,
    Chunk,
    ErrorCategory,
    StopReason,
)

NOTICE: dict[AnswerLanguage, str] = {
    "en": (
        "Not confirmed that every relevant page was searched. "
        "This answer uses only what was found before the run stopped."
    ),
    "zh": "没有确认是否查全。这个回答只用了运行停止前已经查到的段。",
}

# Fixed reply to a greeting or thanks: nothing is retrieved and no model is called.
SMALLTALK: dict[AnswerLanguage, str] = {
    "en": "Hello. Ask a question about the documentation and I will answer from it.",
    "zh": "你好。请直接问文档里的问题。我会根据文档回答。",
}


def smalltalk_answer(question: str, language_setting: Any) -> Answer:
    language = answer_language(question, language_setting)
    return Answer(
        request_id="",
        kb="",
        status=AnswerStatus.ANSWERED,
        found=False,
        sentences=[AnswerSentence(text=SMALLTALK[language])],
    )


def group_by_sub_question(rounds: Sequence[RetrievalRound]) -> list[list[Chunk]]:
    """One chunk list per sub-question, in the order the sub-questions first appeared: later
    rounds of the same sub-question add to its list, so the context budget (s10-3) is split
    between different sub-questions, not between rounds."""
    lists: dict[str, list[Chunk]] = {}
    for r in rounds:
        lists.setdefault(r.sub_question, []).extend(r.chunks)
    return list(lists.values())


def _unique_chunks(rounds: Sequence[RetrievalRound]) -> list[Chunk]:
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for chunk in (c for r in rounds for c in r.chunks):
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def compose_result(
    values: Mapping[str, Any],
    ctx: AgentContext,
    *,
    timings_ms: Mapping[str, float],
    stop_reason: StopReason | None = None,
) -> AskResult:
    """`AskResult` from the state `values` (the final state, or the last checkpoint of a run
    that was cut off: then there may be no `answer`, and `stop_reason` says why)."""
    rounds: list[RetrievalRound] = list(values.get("rounds") or [])
    answer: Answer | None = values.get("answer")
    stop_reason = stop_reason or values.get("stop_reason")
    question = values.get("question", "")
    last = rounds[-1] if rounds else None
    meta: dict[str, Any] = {
        "request_id": ctx.request_id,
        "kb": ctx.cfg.kb,
        "session_id": ctx.session_id,
        "thread_id": ctx.thread_id,
        "index_version": last.index_version if last else "",
        "data_generation": last.data_generation if last else 0,
        "config_version": last.config_version if last else "",
        "degraded": any(r.degraded for r in rounds),
        "rerank_timeout": any(r.rerank_timeout for r in rounds),
        "rounds": values.get("round") or 0,
        "timings_ms": dict(timings_ms),
        "tokens_used": ctx.budget.spent,
        "used_history": ctx.used_history,
    }

    clarification = values.get("clarification")
    if clarification:  # P3-02: the run ended with a question for the caller
        return AskResult(
            **meta,
            status=AnswerStatus.NEED_CLARIFICATION,
            answer=clarification,
            resume_token=ctx.thread_id,
            cache_allowed=False,
        )

    if ctx.budget.cap_hit and stop_reason is None:
        stop_reason = "token_cap"
    if answer is None:  # cut off before the answer: the chunks found so far, no text
        fields: dict[str, Any] = {
            "status": AnswerStatus.NOT_FOUND,
            "found": False,
            "sources": _unique_chunks(rounds),
        }
    else:
        generation_error = answer.generation_error
        if ctx.budget.cap_hit and generation_error == ErrorCategory.BUDGET_EXCEEDED:
            generation_error = None  # our own token cap, reported as `token_cap_reached`
        fields = {
            "status": answer.status,
            "found": answer.found,
            "answer": answer.text,
            "sentences": answer.sentences,
            "sources": answer.chunks,
            "answer_parse_failed": answer.answer_parse_failed,
            "generation_error": generation_error,
            "citation_check": answer.citation_check,
        }
    notice = None
    if stop_reason is not None:
        notice = NOTICE[answer_language(question, ctx.cfg.answer.language)]
    result = AskResult(
        **meta,
        **fields,
        stop_reason=stop_reason,
        coverage_unconfirmed=stop_reason is not None,
        notice=notice,
    )
    # The exact cache (P5) takes only complete answers to a question asked on its own: not one
    # that used history, stopped at a limit or carries a flag (s12-1).
    complete = not (ctx.used_history or result.coverage_unconfirmed or result.error_categories)
    return result.model_copy(update={"cache_allowed": complete})
