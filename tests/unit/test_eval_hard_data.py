"""Spec EVAL-2 AC1-AC5, AC11: the hand-written hard and ambiguous questions (data checks).

The shape checks always run. The checks against the corpus text need the downloaded Hugging Face
corpus (`HF_DATA_DIR`, default `data/raw/hf`, git-ignored): they are skipped without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import cache

import pytest

from rag.config import REPO_ROOT
from rag.eval.hard import (
    HARD_FILE,
    corpus_file,
    corpus_report,
    load_corpus,
    load_hard,
    validate_questions,
)
from rag.eval.questions import load_question_set

SETS = "evals/datasets/hf_docs"
OTHERS = [
    f"{SETS}/two_hop.jsonl",
    f"{SETS}/colloquial.jsonl",
    f"{SETS}/no_answer.jsonl",
    f"{SETS}/no_answer_extra.jsonl",
    f"{SETS}/no_answer_hard.jsonl",
    f"{SETS}/huggingface_doc_qa_eval.jsonl",
    f"{SETS}/synthetic_dev.jsonl",
    f"{SETS}/synthetic_holdout.jsonl",
]


@cache
def _questions():
    return load_hard()


def _hard():
    return [q for q in _questions() if q.meta["type"] == "hard"]


def _ambiguous():
    return [q for q in _questions() if q.meta["type"] == "ambiguous"]


@cache
def _corpus() -> dict[str, str] | None:
    return load_corpus()


def _words(text: str) -> str:
    return " ".join(text.split())


def test_set_loads_with_the_expected_counts_and_ids() -> None:
    assert f"{SETS}/hard.jsonl" == HARD_FILE
    assert len(_hard()) == 25 and len(_ambiguous()) == 10
    assert all(q.qid.startswith("hf_docs_hard:") for q in _hard())
    assert all(q.qid.startswith("hf_docs_ambiguous:") for q in _ambiguous())
    assert len({q.qid for q in _questions()}) == 35
    assert validate_questions(_questions()) == []


def test_nothing_repeats_the_other_sets() -> None:
    def norm(text: str) -> str:
        return re.sub(r"\W+", " ", text).strip().lower()

    others = [q for path in OTHERS for q in load_question_set(path, kb="hf_docs").questions]
    assert not {q.qid for q in _questions()} & {q.qid for q in others}
    assert not {norm(q.question) for q in _questions()} & {norm(q.question) for q in others}
    # A hard question may not reuse the page pair of a two-hop question either.
    old_pairs = {
        frozenset(q.relevant_pages)
        for q in load_question_set(f"{SETS}/two_hop.jsonl", kb="hf_docs").questions
    }
    assert not {frozenset(q.relevant_pages) for q in _hard()} & old_pairs


def test_every_hard_question_has_answer_pages_facts_and_a_template() -> None:
    for q in _hard():
        assert q.answer.strip(), q.qid
        assert 2 <= len(q.relevant_pages) <= 3, q.qid
        meta = q.meta
        assert meta["requires"] == "all"
        assert [f["page"] for f in meta["facts"]] == q.relevant_pages, q.qid
        assert meta["failure_type"] in {"bridge", "comparison", "vocabulary_gap"}, q.qid
        assert meta["modelled_on"].startswith(("hf_docs_two_hop:", "huggingface_doc_qa_eval:"))
        assert meta["checked_single_page"], q.qid
        for fact in meta["facts"]:
            assert fact["evidence"].strip() and len(fact["evidence"]) <= 200, q.qid
            assert fact["key"] and fact["loose"], q.qid


def test_shape_mix() -> None:
    shapes = Counter(q.meta["shape"] for q in _hard())
    assert shapes["bridge"] >= 6 and shapes["comparison"] >= 8
    assert shapes["list_all"] >= 5 and shapes["chain3"] >= 2
    assert set(shapes) == {"bridge", "comparison", "list_all", "chain3"}
    assert all(len(q.relevant_pages) == 3 for q in _hard() if q.meta["shape"] == "chain3")


def test_paraphrase_questions_avoid_the_documents_words() -> None:
    marked = [q for q in _hard() if "paraphrase" in q.meta.get("style", [])]
    assert len(marked) >= 8
    for q in marked:
        avoided = q.meta["avoided_terms"]
        assert len(avoided) >= 1, q.qid
        evidence = " ".join(_words(f["evidence"]) for f in q.meta["facts"]).lower()
        for term in avoided:
            assert term.lower() in evidence or term.lower() in _words(q.answer).lower(), (
                q.qid,
                term,
            )
            assert term.lower() not in q.question.lower(), (q.qid, term)


def test_ambiguous_questions_have_readings_with_evidence() -> None:
    for q in _ambiguous():
        assert not q.relevant_pages and not q.relevant_chunks, q.qid
        meta = q.meta
        assert meta["check"] == "rule:asked_back"
        assert meta["ambiguity"] in {"scope", "missing_referent"}
        readings = meta["readings"]
        assert len(readings) >= 2, q.qid
        assert len({r["answer"] for r in readings}) == len(readings), q.qid
        assert all(r["page"].startswith("hf_docs:") and r["evidence"].strip() for r in readings)
        assert q.answer.lower().startswith("clarification expected"), q.qid
        assert not q.question.strip().endswith("..."), q.qid


@pytest.mark.skipif(corpus_file() is None, reason="the Hugging Face corpus is not downloaded")
def test_evidence_is_verbatim_in_the_gold_pages() -> None:
    corpus = _corpus()
    assert corpus is not None
    for q in _hard():
        for fact in q.meta["facts"]:
            assert fact["page"] in corpus, (q.qid, fact["page"])
            page = _words(corpus[fact["page"]])
            for text in [fact["evidence"], *fact.get("evidence_also", [])]:
                assert _words(text) in page, (q.qid, fact["page"], text)
    for q in _ambiguous():
        for reading in q.meta["readings"]:
            assert _words(reading["evidence"]) in _words(corpus[reading["page"]]), (
                q.qid,
                reading["page"],
            )


@pytest.mark.skipif(corpus_file() is None, reason="the Hugging Face corpus is not downloaded")
def test_no_single_page_states_all_facts() -> None:
    corpus = _corpus()
    assert corpus is not None
    findings = corpus_report(_hard(), corpus)
    assert len(findings) == 25
    bad = {f.qid: f.problems for f in findings if f.problems}
    assert not bad, bad
    # The proof is not vacuous: every fact's patterns were really searched over the corpus.
    assert all(f.searched_pages == len(corpus) for f in findings)


@pytest.mark.skipif(corpus_file() is None, reason="the Hugging Face corpus is not downloaded")
def test_ambiguous_readings_are_in_the_corpus() -> None:
    corpus = _corpus()
    assert corpus is not None
    bad = {f.qid: f.problems for f in corpus_report(_ambiguous(), corpus) if f.problems}
    assert not bad, bad


def test_manifest_and_notice() -> None:
    manifest = json.loads((REPO_ROOT / SETS / "hard_manifest.json").read_text(encoding="utf-8"))
    data = (REPO_ROOT / HARD_FILE).read_bytes().replace(b"\r\n", b"\n")
    assert manifest["file"] == "hard.jsonl"
    assert manifest["version"] >= 1
    assert manifest["sha256"] == hashlib.sha256(data).hexdigest(), (
        "hard.jsonl changed: write a new version into hard_manifest.json (and say why)"
    )
    assert manifest["counts"] == {"hard": 25, "ambiguous": 10}
    assert manifest["authored_by"].startswith("AI")
    assert isinstance(manifest["dropped"], list)
    notice = (REPO_ROOT / SETS / "NOTICE.md").read_text(encoding="utf-8")
    assert "hard.jsonl" in notice
