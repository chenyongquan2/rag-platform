"""Security questions answered by the standard answer path (spec P1-04, P1-03, design s11-3).

The planted pages are not in any index, so the contexts are the planted page's own chunks (what
retrieval would return for it, `planted_chunks`), numbered and delimited by `build_context` and
answered by `generate_answer` with the real model; the P1-03 rule scorer then checks each answer.
"""

from __future__ import annotations

from rag.config import KnowledgeBaseConfig
from rag.data import make_doc_id
from rag.eval.answer.record import AnswerItem, AnswerRunRecord
from rag.eval.answer.runner import RetrievedContext
from rag.eval.security import (
    AnswerRecord,
    SecurityReport,
    SecuritySet,
    planted_chunks,
    score_answers,
)
from rag.ingest.tokens import TokenizerFn
from rag.types import Chunk


def security_items(sset: SecuritySet, *, kb: str = "hf_docs") -> list[AnswerItem]:
    return [
        AnswerItem(
            qid=q.qid,
            question=q.question,
            reference=q.answer,
            relevant_pages=[make_doc_id(kb, sset.pages[q.doc].source_rel)],
        )
        for q in sset.questions
    ]


class PlantedContexts:
    """The chunks of the planted page each security question is about (chunked once per page)."""

    def __init__(
        self,
        sset: SecuritySet,
        kb_config: KnowledgeBaseConfig,
        *,
        tokenizer: TokenizerFn | None = None,
    ) -> None:
        self._doc_of = {q.qid: q.doc for q in sset.questions}
        self._chunks: dict[str, list[Chunk]] = {
            doc: planted_chunks(page, kb_config, tokenizer=tokenizer)
            for doc, page in sset.pages.items()
        }

    def __call__(self, item: AnswerItem) -> RetrievedContext:
        return RetrievedContext(chunks=list(self._chunks[self._doc_of[item.qid]]))


def security_report(sset: SecuritySet, run: AnswerRunRecord) -> SecurityReport:
    """P1-03 rule scores of the run's answers; a question without an answer is scored on ""."""
    records = [
        AnswerRecord(qid=q.qid, answer=q.answer)
        if q.answer is not None
        else AnswerRecord(qid=q.qid, text="")
        for q in run.questions
    ]
    return score_answers(sset, records)
