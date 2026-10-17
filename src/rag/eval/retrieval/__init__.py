"""Retrieval evaluation harness (spec P0-10, design s14-1, s14-4, s14-6).

CLI: `uv run python -m rag.eval.retrieval compare <run_a> <run_b>` and `... export-hf`.
"""

from rag.eval.retrieval.protocol import Reranker, Retriever
from rag.eval.retrieval.record import (
    FINAL,
    HitRecord,
    LatencySummary,
    LevelSummary,
    Provenance,
    QuestionRecord,
    RunInfo,
    RunRecord,
)
from rag.eval.retrieval.report import render_run_report, resolution_sentence
from rag.eval.retrieval.runner import (
    DEFAULT_KS,
    collect_provenance,
    latency_summary,
    load_run,
    make_run_id,
    run_dir,
    run_retrieval_eval,
    write_run,
)

__all__ = [
    "DEFAULT_KS",
    "FINAL",
    "HitRecord",
    "LatencySummary",
    "LevelSummary",
    "Provenance",
    "QuestionRecord",
    "Reranker",
    "Retriever",
    "RunInfo",
    "RunRecord",
    "collect_provenance",
    "latency_summary",
    "load_run",
    "make_run_id",
    "render_run_report",
    "resolution_sentence",
    "run_dir",
    "run_retrieval_eval",
    "write_run",
]
