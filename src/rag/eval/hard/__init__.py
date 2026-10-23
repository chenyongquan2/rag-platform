"""Hard answerable questions and ambiguous questions (spec EVAL-2).

`hard.jsonl` holds 25 hand-written questions that need two or three pages and 10 ambiguous
questions whose right reply is a clarifying question. This package checks the file against the
corpus (no single page states all facts of a hard question), applies the rule check to the
ambiguous ones (`asked_back`) and exports a blind sheet for the owner. CLI:
`python -m rag.eval.hard check | self-check | clarify-report`.
"""

from rag.eval.hard.clarify import asked_back, clarification_section, clarification_summary
from rag.eval.hard.corpus import (
    QuestionFinding,
    corpus_file,
    corpus_report,
    load_corpus,
)
from rag.eval.hard.selfcheck import export_self_check
from rag.eval.hard.spec import HARD_FILE, load_hard, validate_questions

__all__ = [
    "HARD_FILE",
    "QuestionFinding",
    "asked_back",
    "clarification_section",
    "clarification_summary",
    "corpus_file",
    "corpus_report",
    "export_self_check",
    "load_corpus",
    "load_hard",
    "validate_questions",
]
