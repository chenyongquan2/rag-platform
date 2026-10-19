"""Spec P2-02 AC10: the extra no-answer question set that widens the threshold check."""

from __future__ import annotations

from rag.eval.questions import load_question_set

ORIGINAL = "evals/datasets/hf_docs/no_answer.jsonl"
EXTRA = "evals/datasets/hf_docs/no_answer_extra.jsonl"


def test_no_answer_extra_set_is_new_unlabelled_and_typed() -> None:
    original = load_question_set(ORIGINAL, kb="hf_docs")
    extra = load_question_set(EXTRA, kb="hf_docs")
    assert len(extra.questions) == 20
    assert all(not q.relevant_pages and not q.relevant_chunks for q in extra.questions)
    kinds = [q.meta.get("type") for q in extra.questions]
    assert sorted(set(kinds)) == ["near_miss", "off_topic"]
    assert kinds.count("near_miss") == 12 and kinds.count("off_topic") == 8
    # Nothing repeats the original ten (ids or wording), so the two sets can be pooled.
    assert not {q.qid for q in original.questions} & {q.qid for q in extra.questions}
    assert not {q.question for q in original.questions} & {q.question for q in extra.questions}
    # Each question says what was checked against the corpus before it was labelled no-answer.
    assert all(q.meta.get("checked_absent") for q in extra.questions)


def test_false_premise_questions_are_tagged() -> None:
    """Three questions turned out (in review) to rest on a premise the docs touch: they are not
    clean near-miss questions and must stay marked so the report can say so."""
    extra = load_question_set(EXTRA, kb="hf_docs")
    tagged = sorted(q.qid for q in extra.questions if q.meta.get("false_premise"))
    assert tagged == [f"hf_docs_no_answer:{n}" for n in (12, 13, 30)]
