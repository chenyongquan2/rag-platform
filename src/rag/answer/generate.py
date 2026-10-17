"""Answer generation: stream text deltas, then parse and check (design s11-1, s11-2, s16-8).

    async for event in stream_answer(question, ctx, kb=kb, config=cfg.answer):
        match event:
            case AnswerDelta(text=text): ...      # show the text as it comes (`token` event, s16-9)
            case AnswerRetry(): ...               # the text so far is void, a second attempt starts
            case AnswerDone(answer=answer): ...   # parsed and checked (`answer` event)

    answer = await generate_answer(question, ctx, kb=kb, config=cfg.answer)  # same, no deltas

One attempt streams the model's answer (`online` profile by default) and parses it. An answer
that cites no fragment at all is tried once more (`parse_retries`) with a reminder; after that it
is `answer_parse_failed` and the caller gets the chunks only. The fixed not-found sentence gives
`found=false`. A gateway timeout, error or budget refusal also gives the chunks only, with the
category in `generation_error`; any other exception is a bug and propagates.

The model is always streamed, also for `generate_answer`: the profile's read timeout then limits
the gap between two chunks, not the whole generation (spec P1-01 open issue).

The `generate` step span carries the context, the answer and the citation check numbers, and the
tokens and CNY of the ledger lines written while the model was working for this answer. The span
is current only during the model's own steps, not while the caller handles an event, so the
caller's spans and model calls between two events are not counted here.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from opentelemetry.trace import Span

from rag.answer.context import AnswerContext
from rag.answer.parse import CheckedAnswer, ParsedAnswer, check_citations, parse_answer
from rag.answer.prompts import (
    NOT_FOUND,
    AnswerLanguage,
    answer_language,
    build_messages,
    retry_messages,
)
from rag.config import AnswerConfig, ServiceSettings
from rag.cost import add_record_listener
from rag.llm import chat_error_category, make_chat_model
from rag.observe import (
    activate_span,
    current_request_id,
    layer_span,
    new_request_id,
    set_error_category,
)
from rag.observe import attributes as attr
from rag.types import (
    Answer,
    AnswerSentence,
    AnswerStatus,
    CitationCheck,
    ErrorCategory,
)

logger = logging.getLogger(__name__)

STEP = "generate"


@dataclass(frozen=True)
class AnswerDelta:
    """A piece of answer text as the model produced it (markers like `[1]` included)."""

    text: str
    attempt: int


@dataclass(frozen=True)
class AnswerRetry:
    """Attempt `attempt` starts: the text of the earlier attempts is void."""

    attempt: int
    reason: str


@dataclass(frozen=True)
class AnswerDone:
    """The parsed and checked answer; always the last event."""

    answer: Answer


AnswerEvent = AnswerDelta | AnswerRetry | AnswerDone


# Ledger lines written while an answer is generated, collected per context (task) so that
# concurrent answers do not mix; the listener is registered once per process.
_LEDGER_ROWS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "answer_ledger_rows", default=None
)
_LISTENER_LOCK = threading.Lock()
_listener_installed = False


def _collect_ledger_row(row: dict[str, Any]) -> None:
    rows = _LEDGER_ROWS.get()
    if rows is not None:
        rows.append(row)


def _install_ledger_listener() -> None:
    global _listener_installed
    with _LISTENER_LOCK:
        if not _listener_installed:
            add_record_listener(_collect_ledger_row)
            _listener_installed = True


def _content_text(content: Any) -> str:
    """Text of a message chunk's content (a string, or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


@dataclass
class _Outcome:
    attempts: int = 0
    parsed: ParsedAnswer | None = None
    checked: CheckedAnswer | None = None
    parse_failed: bool = False
    error: ErrorCategory | None = None


def _context_attributes(context: AnswerContext, language: AnswerLanguage) -> dict[str, Any]:
    return {
        attr.CONTEXT_FRAGMENTS: len(context.fragments),
        attr.CONTEXT_TOKENS: context.tokens,
        attr.CONTEXT_BUDGET: context.budget_tokens,
        attr.CONTEXT_DROPPED: len(context.dropped),
        attr.CONTEXT_DUPLICATES: len(context.duplicates),
        attr.CONTEXT_SUSPICIOUS: len(context.suspicious),
        attr.ANSWER_LANGUAGE: language,
    }


def _build_answer(
    outcome: _Outcome,
    context: AnswerContext,
    language: AnswerLanguage,
    *,
    kb: str,
    request_id: str,
    elapsed_ms: float,
) -> Answer:
    checked = outcome.checked
    check = CitationCheck(
        removed_markers=list(checked.removed_markers) if checked else [],
        unsupported_sentences=checked.unsupported_sentences if checked else 0,
        parse_attempts=outcome.attempts,
    )
    base: dict[str, Any] = {
        "request_id": request_id,
        "kb": kb,
        "chunks": context.chunks,
        "citation_check": check,
        "timings_ms": {STEP: round(elapsed_ms, 3)},
    }
    if outcome.error is not None or outcome.parse_failed:
        return Answer(
            **base,
            status=AnswerStatus.NOT_FOUND,
            found=False,
            answer_parse_failed=outcome.parse_failed,
            generation_error=outcome.error,
        )
    if checked is None:  # the not-found sentence, or no fragment at all
        return Answer(
            **base,
            status=AnswerStatus.NOT_FOUND,
            found=False,
            sentences=[AnswerSentence(text=NOT_FOUND[language])],
        )
    return Answer(**base, status=AnswerStatus.ANSWERED, found=True, sentences=checked.sentences)


def _set_answer_attributes(
    span: Span, answer: Answer, outcome: _Outcome, rows: list[dict[str, Any]]
) -> None:
    span.set_attributes(
        {
            attr.ANSWER_FOUND: answer.found,
            attr.ANSWER_SENTENCES: len(answer.sentences),
            attr.ANSWER_UNSUPPORTED: answer.citation_check.unsupported_sentences,
            attr.ANSWER_REMOVED_MARKERS: len(answer.citation_check.removed_markers),
            attr.ANSWER_ATTEMPTS: outcome.attempts,
            attr.ANSWER_PARSE_FAILED: answer.answer_parse_failed,
            attr.ANSWER_MODEL_CALLS: len(rows),
            attr.ANSWER_COST_CNY: round(sum(float(r.get("cny") or 0.0) for r in rows), 10),
            attr.GEN_AI_USAGE_INPUT_TOKENS: sum(int(r.get("prompt_tokens") or 0) for r in rows),
            attr.GEN_AI_USAGE_OUTPUT_TOKENS: sum(
                int(r.get("completion_tokens") or 0) for r in rows
            ),
            attr.USAGE_ESTIMATED: any(bool(r.get("usage_estimated")) for r in rows),
        }
    )
    for category in answer.error_categories:
        set_error_category(category, span)


@contextlib.contextmanager
def _active(span: Span, rows: list[dict[str, Any]]) -> Iterator[None]:
    """Our own work inside the step: the span is current (framework spans nest under it) and
    ledger lines written now are this answer's. Never held across a yield to the caller."""
    token = _LEDGER_ROWS.set(rows)
    try:
        with activate_span(span):
            yield
    finally:
        _LEDGER_ROWS.reset(token)


async def _model_text(
    chat: BaseChatModel, messages: list[BaseMessage], span: Span, rows: list[dict[str, Any]]
) -> AsyncIterator[str]:
    """Text pieces of one streamed model call. Each step of the model's stream runs inside
    `_active`; closing this generator closes the model's stream at once (`aclosing`)."""
    async with contextlib.aclosing(chat.astream(messages)) as stream:
        while True:
            with _active(span, rows):
                try:
                    chunk = await anext(stream)
                except StopAsyncIteration:
                    return
            text = _content_text(chunk.content)
            if text:
                yield text


async def stream_answer(
    question: str,
    context: AnswerContext,
    *,
    kb: str,
    config: AnswerConfig,
    model: BaseChatModel | None = None,
    request_id: str | None = None,
    settings: ServiceSettings | None = None,
) -> AsyncIterator[AnswerEvent]:
    """Text deltas of the answer as they come, then `AnswerDone` with the parsed answer.

    `model` defaults to `make_chat_model("online", settings)`; it is only built when there is a
    fragment to answer from. `request_id` defaults to the current request's or a new one.
    The `generate` span is current, and ledger lines count for this answer, only while the model
    is working; the caller's code between two events runs outside both. Closing the stream early
    closes the model's stream and ends the span as cancelled.
    """
    language = answer_language(question, config.language)
    request_id = request_id or current_request_id() or new_request_id()
    start = time.perf_counter()
    outcome = _Outcome()
    rows: list[dict[str, Any]] = []
    _install_ledger_listener()
    attributes = _context_attributes(context, language)
    with layer_span(STEP, attributes=attributes, activate=False) as span:
        if context.fragments:
            chat = model if model is not None else make_chat_model("online", settings)
            messages = build_messages(question, context, language)
            while True:
                outcome.attempts += 1
                if outcome.attempts > 1:
                    yield AnswerRetry(outcome.attempts, ErrorCategory.ANSWER_PARSE_FAILED)
                parts: list[str] = []
                try:
                    async with contextlib.aclosing(
                        _model_text(chat, messages, span, rows)
                    ) as pieces:
                        async for text in pieces:
                            parts.append(text)
                            yield AnswerDelta(text, outcome.attempts)
                except Exception as exc:
                    category = chat_error_category(exc)
                    if category is None:
                        raise
                    logger.warning("answer generation failed (%s): %s", category, exc)
                    outcome.error = category
                    break
                output = "".join(parts)
                outcome.parsed = parse_answer(output)
                if outcome.parsed.not_found:
                    outcome.checked = None
                    break
                outcome.checked = check_citations(outcome.parsed, context)
                if outcome.checked.has_valid_citation:
                    break
                if outcome.attempts > config.parse_retries:
                    outcome.parse_failed = True
                    break
                logger.info("answer cites no fragment; retrying (attempt %d)", outcome.attempts)
                messages = retry_messages(
                    messages,
                    output,
                    language,
                    fragment_count=len(context.fragments),
                    unknown_markers=outcome.checked.removed_markers,
                )
        answer = _build_answer(
            outcome,
            context,
            language,
            kb=kb,
            request_id=request_id,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        _set_answer_attributes(span, answer, outcome, rows)
    yield AnswerDone(answer)


async def generate_answer(
    question: str,
    context: AnswerContext,
    *,
    kb: str,
    config: AnswerConfig,
    model: BaseChatModel | None = None,
    request_id: str | None = None,
    settings: ServiceSettings | None = None,
) -> Answer:
    """`stream_answer` consumed to the end: the final answer only."""
    answer: Answer | None = None
    async with contextlib.aclosing(
        stream_answer(
            question,
            context,
            kb=kb,
            config=config,
            model=model,
            request_id=request_id,
            settings=settings,
        )
    ) as events:
        async for event in events:
            if isinstance(event, AnswerDone):
                answer = event.answer
    assert answer is not None  # stream_answer always ends with AnswerDone
    return answer
