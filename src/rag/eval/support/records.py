"""Items to check and the judge's lines (spec P1-06).

An item is one stored answer: its sentences with their citations and the chunks they cite. A
line is one judge pass over one item, appended to `lines.jsonl` as soon as it exists; the item is
copied into the line, so a line file is enough for the report, the planted errors and the human
sheet. A key (source, system, qid, run) written twice (`--retry-unknown`) keeps its newest line.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.answer.support import SupportResult
from rag.eval.agentic.records import _read_lines, append_line, repair_tail
from rag.eval.answer.record import Usage
from rag.types import AnswerSentence, Chunk

LINES_NAME = "lines.jsonl"
SkipReason = Literal["not_found", "no_answer", "no_sentence_citations", "no_sentences"]


class Planted(BaseModel):
    """A deliberately unsupported sentence in a copy of a real answer (judge sanity check)."""

    kind: Literal["swap_fragment", "change_number", "reviewer_hard"]
    """`reviewer_hard`: a hand-made near miss of the P1-06 review (only on the human sheet)."""
    original_qid: str
    sentence: int = Field(ge=1)
    """1-based index of the changed sentence."""
    before: str
    after: str
    chunk_id: str
    """The chunk the changed sentence now cites (the donor for a swap)."""


class SupportItem(BaseModel):
    source: str
    """`p1-04`, `p3-05` or `planted`."""
    system: str
    qid: str
    qtype: str
    question: str
    sentences: list[AnswerSentence] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    skip_reason: str | None = None
    """Why the item is not judged (no answer, not found, a record without sentence citations)."""
    planted: Planted | None = None

    def key_fields(self) -> dict[str, Any]:
        return {"source": self.source, "system": self.system, "qid": self.qid, "qtype": self.qtype}


class SupportLine(BaseModel):
    source: str
    system: str
    qid: str
    qtype: str
    run: int = Field(ge=1)
    item: SupportItem
    result: SupportResult | None = None
    """None for a skipped item."""
    usage: Usage = Field(default_factory=Usage)

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.source, self.system, self.qid, self.run)

    @property
    def has_unknown(self) -> bool:
        return self.result is not None and any(
            p.label == "unknown" for s in self.result.sentences for p in s.pairs
        )


def lines_file(out_dir: str | Path) -> Path:
    return Path(out_dir) / LINES_NAME


def read_lines(path: str | Path) -> list[SupportLine]:
    """Lines in file order; a cut-off last line is ignored, a bad line elsewhere raises."""
    return _read_lines(path, SupportLine)


def latest_lines(lines: Iterable[SupportLine]) -> list[SupportLine]:
    """One line per key, the newest, in the order the keys first appeared."""
    out: dict[tuple[str, str, str, int], SupportLine] = {}
    for line in lines:
        out[line.key] = line  # a dict keeps the position of the first insertion
    return list(out.values())


def write_line(path: Path, line: SupportLine) -> None:
    append_line(path, line)


__all__ = [
    "LINES_NAME",
    "Planted",
    "SupportItem",
    "SupportLine",
    "latest_lines",
    "lines_file",
    "read_lines",
    "repair_tail",
    "write_line",
]
