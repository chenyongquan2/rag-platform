# ruff: noqa: RUF001 (Chinese sentence ends are full-width)
"""Dry run of the security harness with scripted model replies (spec P1-03).

For every question the planted page is chunked by the knowledge base's pipeline, the context
builder numbers and delimits the chunks, and `generate_answer` runs with a `StreamingFakeChatModel`
that gives a scripted reply: the reference answer with a `[1]` citation (compliant), or the page's
`followed_example` (a reply that follows the injection). Both sets of answers are then scored,
which checks the scorer end to end on real `Answer` objects. No real model is called.
"""

from __future__ import annotations

import asyncio

from rag.answer.context import build_context
from rag.answer.generate import generate_answer
from rag.config import KnowledgeBaseConfig
from rag.eval.security.dataset import SecuritySet, planted_chunks
from rag.eval.security.score import AnswerRecord
from rag.ingest.tokens import TokenizerFn
from rag.testing import StreamingFakeChatModel

_SENTENCE_END = ("。", "！", "？", ".", "!", "?")


def cited(reference: str, language: str) -> str:
    """`reference` with a `[1]` citation before its final punctuation."""
    text = reference.rstrip()
    end = ""
    if text.endswith(_SENTENCE_END):
        text, end = text[:-1], text[-1]
    space = "" if language == "zh" else " "
    return f"{text}{space}[1]{end or ('。' if language == 'zh' else '.')}"


def dry_run(
    sset: SecuritySet,
    kb_config: KnowledgeBaseConfig,
    *,
    tokenizer: TokenizerFn | None = None,
) -> tuple[list[AnswerRecord], list[AnswerRecord]]:
    """(compliant answers, answers that follow the injection), one per question, in set order."""
    chunks_by_doc = {
        doc: planted_chunks(page, kb_config, tokenizer=tokenizer)
        for doc, page in sset.pages.items()
    }

    async def run() -> tuple[list[AnswerRecord], list[AnswerRecord]]:
        compliant: list[AnswerRecord] = []
        followed: list[AnswerRecord] = []
        for q in sset.questions:
            page = sset.pages[q.doc]
            context = build_context(
                [chunks_by_doc[q.doc]],
                kb_config.answer.context_tokens,
                metadata_keys=kb_config.metadata.llm,
            )
            for reply, out in (
                (cited(q.answer, page.language), compliant),
                (page.injection.followed_example, followed),
            ):
                answer = await generate_answer(
                    q.question,
                    context,
                    kb=kb_config.kb,
                    config=kb_config.answer,
                    model=StreamingFakeChatModel(responses=[reply]),
                    request_id=f"dry-{q.qid}",
                )
                out.append(AnswerRecord(qid=q.qid, answer=answer))
        return compliant, followed

    return asyncio.run(run())
