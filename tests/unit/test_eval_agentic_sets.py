"""Spec P3-05 AC1, AC2: the question sets of the Agentic RAG comparison (data and loader).

The checks that need the downloaded Hugging Face corpus are skipped without it and run when
`HF_DATA_DIR` (or `data/raw/hf`) holds it.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from rag.config import ServiceSettings, repo_path
from rag.eval.agentic.sets import DEFAULT_TYPES, QTYPES, SET_FILES, load_items
from rag.eval.questions import load_question_set

SETS = "evals/datasets/hf_docs"
COLLOQUIAL = f"{SETS}/colloquial.jsonl"
PUBLIC = f"{SETS}/huggingface_doc_qa_eval.jsonl"


def _corpus_file() -> Path | None:
    root = repo_path(ServiceSettings().hf_data_dir)
    return next(iter(root.glob("m-ric/huggingface_doc/*/default/train/0000.parquet")), None)


@cache
def _corpus_pages() -> set[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(_corpus_file(), columns=["source"])
    return {f"hf_docs:{s}" for s in table["source"].to_pylist()}


def test_colloquial_questions_derive_from_public_ones() -> None:
    colloquial = load_question_set(COLLOQUIAL, kb="hf_docs").questions
    public = {q.qid: q for q in load_question_set(PUBLIC, kb="hf_docs").questions}
    assert len(colloquial) == 15
    assert len({q.qid for q in colloquial}) == 15
    sources = [q.meta["source_qid"] for q in colloquial]
    assert len(set(sources)) == 15  # no public question is rewritten twice
    for q in colloquial:
        source = public[q.meta["source_qid"]]
        assert q.qid.startswith("hf_docs_colloquial:")
        assert q.meta["type"] == "colloquial"
        assert q.relevant_pages == source.relevant_pages
        assert q.answer == source.answer
        assert q.question != source.question  # really rewritten
        assert q.meta["source_question"] == source.question
        assert "hand-written" in q.meta["provenance"]
        assert q.meta["style"]
    # Chosen by a fixed rule, not by results: every 4th public question, from the first.
    order = [q.qid for q in load_question_set(PUBLIC, kb="hf_docs").questions]
    assert sources == [order[i] for i in range(0, 60, 4)]


def test_colloquial_gold_pages_exist_in_corpus() -> None:
    if _corpus_file() is None:
        pytest.skip("the Hugging Face corpus is not downloaded")
    pages = _corpus_pages()
    for q in load_question_set(COLLOQUIAL, kb="hf_docs").questions:
        assert all(p in pages for p in q.relevant_pages), q.qid


def test_load_items_types_and_counts() -> None:
    items = load_items("hf_docs")
    by_type = {t: [i for i in items if i.qtype == t] for t in DEFAULT_TYPES}
    assert {t: len(v) for t, v in by_type.items()} == {
        "single_hop": 65,
        "two_hop": 20,
        "colloquial": 15,
        "no_answer": 50,
    }
    assert len({i.qid for i in items}) == len(items)
    assert all(i.answerable and len(i.relevant_pages) == 1 for i in by_type["single_hop"])
    assert all(i.requires == "all" and len(i.relevant_pages) == 2 for i in by_type["two_hop"])
    assert all(i.requires == "any" for i in by_type["single_hop"])
    assert all(i.source_qid and i.answerable for i in by_type["colloquial"])
    assert all(not i.answerable and not i.relevant_pages for i in by_type["no_answer"])
    assert all(i.reference.strip() for i in items if i.answerable)
    assert set(SET_FILES) == set(QTYPES)


def test_load_items_selection() -> None:
    items = load_items("hf_docs", take={"single_hop": 4, "two_hop": 2, "no_answer": 2})
    assert [i.qtype for i in items] == ["single_hop"] * 4 + ["two_hop"] * 2 + ["no_answer"] * 2
    # the first questions of each file, in file order
    assert items[0].qid == "huggingface_doc_qa_eval:000"
    only = load_items("hf_docs", types=["colloquial"])
    assert len(only) == 15 and {i.qtype for i in only} == {"colloquial"}
    with pytest.raises(ValueError, match="unknown question type"):
        load_items("hf_docs", types=["three_hop"])
