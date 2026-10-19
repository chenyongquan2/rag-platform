"""The score file `collect` writes and `sweep` reads (spec P2-02).

It holds, per question, the rerank score of every candidate the reranker scored (before any
threshold or top_k cut), so any threshold can be applied afterwards without running the chain
again. The file is the record of a collection run: question set hash, provenance, rerank_n.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.eval.retrieval.record import Provenance, QuestionSetInfo

Kind = Literal["answerable", "no_answer"]


class Candidate(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    """Rerank score in [0, 1] (sigmoid, design s10-2)."""


class ScoreRecord(BaseModel):
    qid: str
    question: str
    kind: Kind
    relevant_pages: list[str] = Field(default_factory=list)
    """Labelled correct pages; empty for no-answer questions."""
    candidates: list[Candidate]
    """Every reranked candidate, best first."""


class ScoreFile(BaseModel):
    schema_version: int = 1
    created_at: str
    kb: str
    rerank_n: int
    question_set: QuestionSetInfo
    provenance: Provenance
    records: list[ScoreRecord]

    @property
    def kind(self) -> Kind:
        """The kind of the questions in the file (`collect` writes one kind per file)."""
        kinds = {r.kind for r in self.records}
        if len(kinds) != 1:
            raise ValueError(f"expected one question kind in the file, found {sorted(kinds)}")
        return next(iter(kinds))


def write_scores(scores: ScoreFile, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scores.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def load_scores(path: str | Path) -> ScoreFile:
    return ScoreFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
