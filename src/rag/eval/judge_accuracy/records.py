"""The record file of a judge-accuracy run (spec P3-03): one row per question with the label, the
retrieval facts the baselines need and what the judge said, so every number of the report can be
recomputed without a model or a search."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.eval.retrieval.record import Provenance
from rag.types import Sufficiency

Kind = Literal["answerable", "no_answer"]


class RescueRecord(BaseModel):
    """What another round would have found for a question that was not enough (run with
    `--rescue`): round two asks for several phrasings, round three (only when round two did not
    find the gold page) for a hypothetical passage. Found = the labelled page(s) are among the
    chunks of all rounds so far."""

    round2_found: bool
    round3_found: bool | None = None
    """None: round three was not needed (round two found it) or could not be made."""
    queries: list[str] = Field(default_factory=list)
    fallback: str | None = None
    cny: float = 0.0


class JudgeRecord(BaseModel):
    set: str
    qid: str
    question: str
    qtype: str
    kind: Kind
    relevant_pages: list[str] = Field(default_factory=list)
    requires: Literal["any", "all"] = "any"
    label_enough: bool
    """The gold page is among the chunks round one returned (after the rerank threshold)."""
    label_in_rerank_top_k: bool
    """Sensitivity: the gold page is among the first k reranked candidates, threshold or not."""
    n_chunks: int
    """Chunks that passed the threshold: the rules alone call the question enough if any did."""
    top_score: float | None = None
    returned_pages: list[str] = Field(default_factory=list)
    verdict: Sufficiency
    missing: list[str] = Field(default_factory=list)
    fallback: str | None = None
    used_model: bool = False
    tokens: int = 0
    cny: float = 0.0
    latency_ms: float = 0.0
    rescue: RescueRecord | None = None


class SetInfo(BaseModel):
    name: str
    path: str
    sha256: str
    count: int


class JudgeFile(BaseModel):
    schema_version: int = 1
    created_at: str = ""
    kb: str
    threshold: float
    k: int
    max_rounds: int = 3
    """The KB's `agent.max_rounds`: the most rounds the real loop runs."""
    judge_profile: str = ""
    provenance: Provenance | None = None
    question_sets: list[SetInfo] = Field(default_factory=list)
    partial: bool = False
    """The run stopped at its spend limit before the last question."""
    spent_cny: float = 0.0
    records: list[JudgeRecord]


def write_records(file: JudgeFile, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(file.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def load_records(path: str | Path) -> JudgeFile:
    return JudgeFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
