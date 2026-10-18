"""Format compliance of a set of answers (spec P1-04, design s11-1, s11-2 level 1, s14-4)."""

from __future__ import annotations

from collections.abc import Sequence

from rag.eval.answer.record import AnswerQuestionRecord, FormatCompliance, Proportion


def _model_answered(record: AnswerQuestionRecord) -> bool:
    """The model produced text for this question (whatever came of it)."""
    return record.parse_attempts > 0 and record.generation_error is None


def format_compliance(records: Sequence[AnswerQuestionRecord]) -> FormatCompliance:
    """Counts and Wilson intervals over the question records (see `FormatCompliance`)."""
    n = len(records)
    found = [r for r in records if r.found]
    model = [r for r in records if _model_answered(r)]
    fully = [r for r in found if r.unsupported_sentences == 0 and r.sentences > 0]
    sentences = sum(r.sentences for r in found)
    unsupported = sum(r.unsupported_sentences for r in found)
    return FormatCompliance(
        n_questions=n,
        answered=Proportion.of(len(found), n),
        found_false=Proportion.of(n - len(found), n),
        fully_cited=Proportion.of(len(fully), len(found)),
        fully_cited_all=Proportion.of(len(fully), n),
        with_unknown_markers=Proportion.of(sum(bool(r.removed_markers) for r in model), len(model)),
        unknown_markers_total=sum(len(r.removed_markers) for r in records),
        with_unsupported=Proportion.of(sum(r.unsupported_sentences > 0 for r in found), len(found)),
        unsupported_sentences=Proportion.of(unsupported, sentences),
        parse_failed=Proportion.of(sum(r.answer_parse_failed for r in model), len(model)),
        retried=Proportion.of(sum(r.parse_attempts > 1 for r in model), len(model)),
        generation_errors=sum(r.generation_error is not None for r in records),
        retrieval_not_found=sum(r.retrieval_not_found for r in records),
        retrieval_errors=sum(r.error is not None for r in records),
    )
