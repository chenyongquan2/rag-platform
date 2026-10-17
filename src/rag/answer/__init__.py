"""Answer generation with numbered citations (spec P1-02, design s10-3, s11-1 to s11-3, s8-4).

Question plus ranked chunks in, `Answer` out; no retrieval here:

    ctx = build_context([result.chunks], cfg.answer.context_tokens, metadata_keys=cfg.metadata.llm)
    answer = await generate_answer(question, ctx, kb=cfg.kb, config=cfg.answer)
"""

from rag.answer.context import (
    AnswerContext,
    Fragment,
    build_context,
    escape_delimiters,
    estimate_tokens,
)
from rag.answer.generate import (
    AnswerDelta,
    AnswerDone,
    AnswerEvent,
    AnswerRetry,
    generate_answer,
    stream_answer,
)
from rag.answer.parse import CheckedAnswer, ParsedAnswer, check_citations, parse_answer
from rag.answer.prompts import (
    NOT_FOUND,
    SYSTEM_PROMPTS,
    AnswerLanguage,
    answer_language,
    build_messages,
    detect_question_language,
)

__all__ = [
    "NOT_FOUND",
    "SYSTEM_PROMPTS",
    "AnswerContext",
    "AnswerDelta",
    "AnswerDone",
    "AnswerEvent",
    "AnswerLanguage",
    "AnswerRetry",
    "CheckedAnswer",
    "Fragment",
    "ParsedAnswer",
    "answer_language",
    "build_context",
    "build_messages",
    "check_citations",
    "detect_question_language",
    "escape_delimiters",
    "estimate_tokens",
    "generate_answer",
    "parse_answer",
    "stream_answer",
]
