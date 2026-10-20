"""Spec P3-03 AC31-AC33, AC36: the hand-written two-hop and no-answer questions (data checks).

The shape checks always run. The checks against the corpus text need the downloaded Hugging Face
corpus (`data/raw/`, git-ignored): they are skipped without it and run when it is there.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from rag.config import REPO_ROOT
from rag.eval.questions import load_question_set

SETS = "evals/datasets/hf_docs"
TWO_HOP = f"{SETS}/two_hop.jsonl"
HARD = f"{SETS}/no_answer_hard.jsonl"
OTHERS = [
    f"{SETS}/no_answer.jsonl",
    f"{SETS}/no_answer_extra.jsonl",
    f"{SETS}/huggingface_doc_qa_eval.jsonl",
    f"{SETS}/synthetic_dev.jsonl",
    f"{SETS}/synthetic_holdout.jsonl",
]
CORPUS = next(
    iter(Path(REPO_ROOT).glob("data/raw/hf/m-ric/huggingface_doc/*/default/train/0000.parquet")),
    None,
)


def _words(text: str) -> str:
    return " ".join(text.split())


@cache
def _corpus() -> dict[str, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(CORPUS, columns=["text", "source"])
    return {
        f"hf_docs:{s}": t
        for s, t in zip(table["source"].to_pylist(), table["text"].to_pylist(), strict=True)
    }


def test_two_hop_set_loads_and_every_question_needs_two_different_pages() -> None:
    questions = load_question_set(TWO_HOP, kb="hf_docs").questions
    assert len(questions) == 20
    shapes = [q.meta.get("shape") for q in questions]
    assert set(shapes) == {"bridge", "comparison"}
    assert 8 <= shapes.count("bridge") <= 12
    for q in questions:
        assert q.qid.startswith("hf_docs_two_hop:") and q.answer.strip()
        assert len(q.relevant_pages) == 2 and len(set(q.relevant_pages)) == 2
        assert q.meta["type"] == "two_hop" and q.meta["requires"] == "all"
        facts = q.meta["facts"]
        assert [f["page"] for f in facts] == q.relevant_pages
        assert all(f["evidence"].strip() and len(f["evidence"]) <= 200 for f in facts)
        assert q.meta.get("checked_single_page"), q.qid


def test_no_answer_hard_set_loads_unlabelled_typed_and_says_what_was_checked() -> None:
    questions = load_question_set(HARD, kb="hf_docs").questions
    assert len(questions) == 20
    assert all(not q.relevant_pages and not q.relevant_chunks and not q.answer for q in questions)
    kinds = [q.meta.get("type") for q in questions]
    assert set(kinds) == {"near_miss", "off_topic_hard"}
    assert kinds.count("near_miss") == 14 and kinds.count("off_topic_hard") == 6
    assert all(q.qid.startswith("hf_docs_no_answer_hard:") for q in questions)
    assert all(len(q.meta.get("checked_absent", "")) > 20 for q in questions)


def test_the_hard_sets_repeat_nothing_of_the_other_sets() -> None:
    ours = [q for path in (TWO_HOP, HARD) for q in load_question_set(path, kb="hf_docs").questions]
    others = [q for path in OTHERS for q in load_question_set(path, kb="hf_docs").questions]
    assert not {q.qid for q in ours} & {q.qid for q in others}
    normalised = {re.sub(r"\W+", " ", q.question).strip().lower() for q in others}
    assert not {re.sub(r"\W+", " ", q.question).strip().lower() for q in ours} & normalised


def test_unverified_premises_stay_marked() -> None:
    """A question whose absence from the docs could not be fully ruled out says so in `meta`,
    so the report can leave it out or count it apart."""
    questions = load_question_set(HARD, kb="hf_docs").questions
    for q in questions:
        flag = q.meta.get("premise_unverified")
        assert flag is None or flag is True, q.qid
        if flag:
            assert "unverified" in q.meta["checked_absent"].lower(), q.qid


@pytest.mark.skipif(CORPUS is None, reason="the Hugging Face corpus is not downloaded")
def test_every_gold_page_exists_and_its_evidence_is_in_its_text() -> None:
    corpus = _corpus()
    for q in load_question_set(TWO_HOP, kb="hf_docs").questions:
        for fact in q.meta["facts"]:
            assert fact["page"] in corpus, (q.qid, fact["page"])
            page = _words(corpus[fact["page"]])
            for excerpt in [fact["evidence"], *fact.get("evidence_also", [])]:
                assert _words(excerpt) in page, (q.qid, fact["page"], excerpt)


def test_every_no_answer_question_records_at_least_six_patterns() -> None:
    """Review round 1: a question checked with three spellings missed 'storing logs for 30 days'.
    Each question records its regular expressions and its inflection-aware concept pairs."""
    for q in load_question_set(HARD, kb="hf_docs").questions:
        checks, concepts = q.meta.get("checks", []), q.meta.get("concepts", [])
        assert len(checks) + len(concepts) >= 6, q.qid
        assert all(len(pair) == 2 and all(pair) for pair in concepts), q.qid


@pytest.mark.skipif(CORPUS is None, reason="the Hugging Face corpus is not downloaded")
def test_no_answer_hard_questions_match_no_page_by_their_recorded_patterns() -> None:
    """The recorded patterns are searched again over the whole corpus: none may match a page
    (inflections, hyphen, space and underscore variants and either word order included)."""
    from rag.eval.absence import concept_pages, lowered, pages_matching

    corpus = lowered(_corpus())
    for q in load_question_set(HARD, kb="hf_docs").questions:
        for pattern in q.meta["checks"]:
            assert not pages_matching(pattern, corpus, is_lowered=True), (q.qid, pattern)
        for anchor, fact in q.meta["concepts"]:
            hits = concept_pages(anchor, fact, corpus, is_lowered=True)
            assert not hits, (q.qid, anchor, fact, hits[:3])
