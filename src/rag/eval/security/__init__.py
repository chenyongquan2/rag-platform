"""Security eval: planted-injection pages, rule-based scorer, harness and corpus scan (spec P1-03).

sset = load_security_set()                         # evals/datasets/security
report = score_answers(sset, read_answer_records("answers.jsonl"))

uv run python -m rag.eval.security run --answers FILE [--set DIR] [--out FILE]
uv run python -m rag.eval.security dry-run [--out-dir DIR]
uv run python -m rag.eval.security scan --kb hf_docs [--limit N | --all] [--samples 20]
"""

from rag.eval.security.dataset import (
    DEFAULT_SET_DIR,
    ExpectedBehaviour,
    Injection,
    SecurityPage,
    SecurityQuestion,
    SecuritySet,
    SecuritySetError,
    load_security_set,
    planted_chunks,
)
from rag.eval.security.harness import dry_run
from rag.eval.security.scan import ScanReport, ScanSample, render_scan, scan_sources
from rag.eval.security.score import (
    AnswerRecord,
    QuestionScore,
    SecurityReport,
    Violation,
    answer_language,
    read_answer_records,
    render_report,
    score_answer,
    score_answers,
    write_answer_records,
)

__all__ = [
    "DEFAULT_SET_DIR",
    "AnswerRecord",
    "ExpectedBehaviour",
    "Injection",
    "QuestionScore",
    "ScanReport",
    "ScanSample",
    "SecurityPage",
    "SecurityQuestion",
    "SecurityReport",
    "SecuritySet",
    "SecuritySetError",
    "Violation",
    "answer_language",
    "dry_run",
    "load_security_set",
    "planted_chunks",
    "read_answer_records",
    "render_report",
    "render_scan",
    "scan_sources",
    "score_answer",
    "score_answers",
    "write_answer_records",
]
