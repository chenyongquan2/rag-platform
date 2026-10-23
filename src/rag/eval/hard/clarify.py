"""The rule check of the ambiguous questions (spec EVAL-2, AC7): did the system ask back?

No judge is used. The graph reports a clarification itself (`clarification` or status
`need_clarification`). Standard RAG has no clarification path, so for it a plain-text reply counts
only when it is mostly a question: its last sentence ends with a question mark and the whole reply
is short (at most `MAX_SENTENCES` sentences and `MAX_CHARS` characters) and the sentences before
the question contain no digit or code span (a concrete value is an answer). A reply that answers one
reading, or every reading, does not count as asking back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.eval.agentic.records import SYSTEMS, SystemRecord
from rag.eval.answer.record import Proportion

MAX_SENTENCES = 3
MAX_CHARS = 400
_SENTENCE_END = re.compile(r"(?<=[.!?？。])\s+")  # noqa: RUF001 (Chinese marks on purpose)


def asked_back(record: SystemRecord) -> bool:
    if record.clarification or record.status == "need_clarification":
        return True
    if not record.answered:
        return False
    text = record.text.strip()
    if not text or len(text) > MAX_CHARS:
        return False
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences or len(sentences) > MAX_SENTENCES:
        return False
    if not sentences[-1].rstrip(" \"')]»”").endswith(("?", "？")):  # noqa: RUF001
        return False
    # Earlier sentences may frame the question, but a concrete value or code means it answered.
    return not any(re.search(r"\d|`", s) for s in sentences[:-1])


def clarification_summary(records: Sequence[SystemRecord]) -> dict[str, Proportion]:
    """Per system: ambiguous questions asked back about, with the 95% Wilson interval."""
    out: dict[str, Proportion] = {}
    for system in SYSTEMS:
        mine = [r for r in records if r.system == system and r.qtype == "ambiguous"]
        if mine:
            out[system] = Proportion.of(sum(asked_back(r) for r in mine), len(mine))
    return out


def clarification_section(records: Sequence[SystemRecord]) -> list[str]:
    """Markdown lines for the report; empty when no ambiguous question was run."""
    summary = clarification_summary(records)
    if not summary:
        return []
    lines = [
        "",
        "## Ambiguous questions: did the system ask back? (rule check, no judge)",
        "",
        "Counted: the graph stopped with a clarification, or a plain reply that is mostly a "
        "question. A reply that answers one reading (or all readings) counts as not asking back.",
        "",
    ]
    for system, p in summary.items():
        lines.append(f"- {system}: asked back {p.text()}")
    answered = sorted(
        {
            r.qid
            for r in records
            if r.qtype == "ambiguous" and r.system == "agentic" and not asked_back(r)
        }
    )
    if answered:
        lines.append(f"- the graph answered instead of asking: {', '.join(answered)}")
    return lines
