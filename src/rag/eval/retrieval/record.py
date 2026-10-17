"""The record of one eval run, written as `results.json` (design s14-4: record everything)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag.types import RouteHit

FINAL = "final"
"""Name of the final list (after fusion, rerank and threshold) next to the route names."""

# metrics[list][level][metric] -> value; a level is absent when the question has no labels there.
Metrics = dict[str, dict[str, dict[str, float]]]


class HitRecord(BaseModel):
    chunk_id: str
    doc_id: str
    score: float | None = None
    route_ranks: dict[str, int | None] = Field(default_factory=dict)
    """Rank of the same chunk in each route's own list (None: that route did not return it)."""


class QuestionRecord(BaseModel):
    qid: str
    question: str
    relevant_pages: list[str]
    relevant_chunks: list[str] = Field(default_factory=list)
    hits: list[HitRecord]
    """The final list, cut at `top_k`."""
    routes: dict[str, list[RouteHit]] = Field(default_factory=dict)
    """Each route's list, cut at the largest k."""
    metrics: Metrics
    first_relevant_rank: dict[str, dict[str, int | None]]
    """Rank of the first relevant item in each whole list, per level with labels."""
    latency_ms: float
    """Wall time of the `retrieve()` call measured by the harness."""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False
    rerank_timeout: bool = False
    error: str | None = None


class LevelSummary(BaseModel):
    n: int
    """Questions with labels at this level (the others are left out of the means)."""
    means: dict[str, float]
    hit_wilson: dict[str, tuple[float, float]]
    """95% Wilson interval of each Hit@k proportion."""


class LatencySummary(BaseModel):
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float


class QuestionSetInfo(BaseModel):
    path: str
    sha256: str
    count: int


class Provenance(BaseModel):
    """What the run depended on outside its inputs (code, lock file, models)."""

    git_commit: str | None = None
    git_dirty: bool | None = None
    """Uncommitted changes to tracked files when the run started."""
    lock_sha256: str | None = None
    models: dict[str, str] = Field(default_factory=dict)


class RunInfo(Provenance):
    run_id: str
    created_at: str
    kb: str
    label: str = ""
    public: bool | None = None
    """Set when written: public reports go to `reports/public/`, others to `reports/private/`."""
    kb_config: dict[str, Any]
    """The knowledge-base config in effect, serialized."""
    kb_config_sha256: str
    index_versions: list[str]
    data_generations: list[int]
    """Distinct values seen in the results; more than one means the index changed mid-run."""
    question_set: QuestionSetInfo
    top_k: int
    ks: list[int]
    lists: list[str]
    """`final` and the route names, in order."""


class RunRecord(BaseModel):
    schema_version: int = 1
    run: RunInfo
    summary: dict[str, dict[str, LevelSummary]]
    """summary[list][level]."""
    latency: LatencySummary
    counts: dict[str, int]
    missed_by_all_lists: list[str]
    """qids whose relevant pages no list found: check the label before the system (s14-2)."""
    questions: list[QuestionRecord]
