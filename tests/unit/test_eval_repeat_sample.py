"""Spec EVAL-3 AC1: the fixed-seed question list of step 3."""

from __future__ import annotations

import random

from rag.eval.agentic.repeat.sample import (
    LIST_PATH,
    OLD_COUNTS,
    build_question_list,
    draw_sample,
    read_question_list,
    write_question_list,
)
from rag.eval.agentic.sets import EvalItem, load_items

SIZES = {
    "single_hop": 65,
    "two_hop": 20,
    "colloquial": 15,
    "hard": 25,
    "ambiguous": 10,
    "no_answer": 50,
}


def _items() -> list[EvalItem]:
    return [
        EvalItem(
            qid=f"{qtype}:{i:02d}",
            qtype=qtype,  # type: ignore[arg-type]
            set="x",
            question=f"q {qtype} {i}",
            answerable=qtype not in ("no_answer", "ambiguous"),
            relevant_pages=["p"] if qtype not in ("no_answer", "ambiguous") else [],
        )
        for qtype, n in SIZES.items()
        for i in range(n)
    ]


def test_draw_is_deterministic_and_order_independent() -> None:
    items = _items()
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    first = build_question_list(items, seed=11, date="2026-10-23")
    again = build_question_list(shuffled, seed=11, date="2026-10-23")
    # The same questions are drawn whatever the order of the files (the list keeps file order).
    assert {t: set(v) for t, v in first.qids.items()} == {t: set(v) for t, v in again.qids.items()}
    assert first == build_question_list(items, seed=11, date="2026-10-23")
    assert "no_answer" not in first.qids  # their repeat adds nothing (plan section 13)


def test_counts_and_seed_change() -> None:
    items = _items()
    ql = build_question_list(items, seed=11, date="2026-10-23")
    assert {t: len(v) for t, v in ql.qids.items()} == {**OLD_COUNTS, "hard": 25, "ambiguous": 10}
    assert len(ql.all_qids()) == 75 and len(set(ql.all_qids())) == 75
    assert ql.old_qids() + ql.new_qids() == ql.all_qids()
    assert ql.seed == 11 and ql.date == "2026-10-23"
    # Every hard and ambiguous question is in; the old ones are a subset of their type.
    assert set(ql.qids["hard"]) == {i.qid for i in items if i.qtype == "hard"}
    assert set(ql.qids["single_hop"]) <= {i.qid for i in items if i.qtype == "single_hop"}
    other = build_question_list(items, seed=12, date="2026-10-23")
    assert other.qids["single_hop"] != ql.qids["single_hop"]
    # Within a type the list keeps the file order.
    for qtype in OLD_COUNTS:
        assert ql.qids[qtype] == sorted(ql.qids[qtype])


def test_draw_takes_at_most_what_exists() -> None:
    items = [i for i in _items() if i.qtype == "two_hop"][:5]
    drawn = draw_sample(items, seed=1, counts={"two_hop": 12})
    assert len(drawn["two_hop"]) == 5


def test_write_and_read_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ql = build_question_list(_items(), seed=3, date="2026-10-23")
    path = tmp_path / "list.json"
    write_question_list(path, ql)
    assert read_question_list(path) == ql
    assert b"\r" not in path.read_bytes()


def test_shipped_list_matches_the_sets() -> None:
    shipped = read_question_list()
    items = load_items(
        "hf_docs", types=["single_hop", "two_hop", "colloquial", "hard", "ambiguous"]
    )
    assert build_question_list(items, seed=shipped.seed, date=shipped.date) == shipped
    assert LIST_PATH.endswith("repeat_step3.json")
