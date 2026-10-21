"""Stored answers as items to check (spec P1-06). No answer is generated here.

- P1-04 (`answers/<run>/results.json`): standard RAG on the 65 public questions; every record
  keeps the full `Answer` (sentences, citations, the numbered fragments).
- P3-05 (`<dir>/<system>/records.jsonl`): standard RAG and the ask_docs graph. Records written
  before P1-06 have the answer text without markers and the cited pages, not which sentence
  cites which chunk: they are skipped as `no_sentence_citations`.
"""

from __future__ import annotations

from pathlib import Path

from rag.eval.agentic.records import SYSTEMS, read_records, system_file
from rag.eval.answer.record import AnswerRunRecord
from rag.eval.support.records import SupportItem


def load_p1_04(path: str | Path) -> list[SupportItem]:
    file = Path(path)
    if file.is_dir():
        file = file / "results.json"
    run = AnswerRunRecord.model_validate_json(file.read_text(encoding="utf-8"))
    items: list[SupportItem] = []
    for q in run.questions:
        base = {
            "source": "p1-04",
            "system": "standard",
            "qid": q.qid,
            "qtype": "single_hop",
            "question": q.question,
        }
        answer = q.answer
        if answer is None:
            items.append(SupportItem(**base, skip_reason="no_answer"))
        elif not answer.found:
            items.append(SupportItem(**base, skip_reason="not_found"))
        elif not answer.sentences:
            items.append(SupportItem(**base, skip_reason="no_sentences"))
        else:
            items.append(
                SupportItem(**base, sentences=list(answer.sentences), chunks=list(answer.chunks))
            )
    return items


def load_p3_05(out_dir: str | Path) -> list[SupportItem]:
    items: list[SupportItem] = []
    for system in SYSTEMS:
        for r in read_records(system_file(out_dir, system)):
            base = {
                "source": "p3-05",
                "system": system,
                "qid": r.qid,
                "qtype": r.qtype,
                "question": r.question,
            }
            if not r.answered:
                items.append(SupportItem(**base, skip_reason="no_answer"))
            elif not r.answer_sentences:
                items.append(SupportItem(**base, skip_reason="no_sentence_citations"))
            else:
                items.append(
                    SupportItem(
                        **base, sentences=list(r.answer_sentences), chunks=list(r.cited_chunks)
                    )
                )
    return items
