"""Spec EVAL-4 AC1, AC2, AC3: the regression set and its baseline."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from rag.eval.agentic.regress.suite import (
    BASELINE_PATH,
    DEFAULT_MAX_PER_TYPE,
    SET_PATH,
    build_suite,
    check_items,
    fingerprint,
    read_baseline,
    read_suite,
    write_baseline,
    write_suite,
)
from rag.eval.agentic.sets import EvalItem, load_items
from rag.testing.repeat import passes_from, qlist_for

# 30 single-hop questions; the first 20 are stable for the graph in all three passes, the rest flip.
SINGLE = [f"single_hop:{i:02d}" for i in range(30)]
TWO = [f"two_hop:{i:02d}" for i in range(4)]
HARD = [f"hard:{i:02d}" for i in range(6)]
AMBIG = [f"ambiguous:{i:02d}" for i in range(3)]
QIDS = [*SINGLE, *TWO, *HARD, *AMBIG]


def _item(qid: str, *, question: str | None = None) -> EvalItem:
    qtype = qid.split(":")[0]
    ambiguous = qtype == "ambiguous"
    return EvalItem(
        qid=qid,
        qtype=qtype,  # type: ignore[arg-type]
        set="x",
        question=question or f"Question {qid}?",
        reference="" if ambiguous else "Reference.",
        relevant_pages=[] if ambiguous else ["hf_docs:p.md"],
        answerable=not ambiguous,
    )


ITEMS = [_item(q) for q in QIDS]


def _spec() -> dict[str, dict[str, str]]:
    agentic = {q: "ccc" for q in SINGLE[:20]} | {q: "cwc" for q in SINGLE[20:]}
    agentic |= {TWO[0]: "ccc", TWO[1]: "ccc", TWO[2]: "wcc", TWO[3]: "ccc"}
    agentic |= {HARD[0]: "ccc", HARD[1]: "ccc", HARD[2]: "ccc", HARD[3]: "cxc"}
    agentic |= {HARD[4]: "ccc", HARD[5]: "ctc"}
    agentic |= {AMBIG[0]: "ccc", AMBIG[1]: "ccc", AMBIG[2]: "cwc"}
    # The standard system is stable on the first 10 single-hop questions and on two_hop:00, hard:00.
    standard = {q: "ccc" if i < 10 else "cwc" for i, q in enumerate(SINGLE)}
    standard |= {q: "cwc" for q in [*TWO[1:], *HARD[1:], *AMBIG]}
    standard |= {TWO[0]: "ccc", HARD[0]: "ccc"}
    return {"agentic": agentic, "standard": standard}


def _build(**kw):  # type: ignore[no-untyped-def]
    passes = passes_from(_spec())
    params = dict(
        qlist=qlist_for(QIDS),
        items=ITEMS,
        seed=5,
        date="2026-10-23",
        max_per_type={"single_hop": 12},
        sources={"base": "x"},
    )
    params.update(kw)
    return build_suite(passes, **params)  # type: ignore[arg-type]


def test_only_stable_questions_enter() -> None:
    rs, _base = _build()
    got = {q.qid for q in rs.questions}
    assert got <= set(SINGLE[:20]) | {
        TWO[0],
        TWO[1],
        TWO[3],
        HARD[0],
        HARD[1],
        HARD[2],
        HARD[4],
    } | {
        AMBIG[0],
        AMBIG[1],
    }
    # Not stable for the graph: a miss in one pass, a deadline cut, a model timeout.
    for bad in [*SINGLE[20:], TWO[2], HARD[3], HARD[5], AMBIG[2]]:
        assert bad not in got
    # Scarce types are taken whole; single-hop is drawn down to the cap.
    assert [q for q in rs.qids() if q.startswith("two_hop")] == [TWO[0], TWO[1], TWO[3]]
    assert len([q for q in rs.qids() if q.startswith("single_hop")]) == 12
    assert rs.counts == {"single_hop": 12, "two_hop": 3, "hard": 4, "ambiguous": 2}
    assert rs.stable_available["agentic"]["single_hop"] == 20
    assert rs.passes == 3 and rs.seed == 5 and rs.date == "2026-10-23"
    assert rs.sources == {"base": "x"}


def test_cap_draw_is_deterministic() -> None:
    first, _ = _build()
    again, _ = _build()
    assert first == again
    # The draw does not depend on the order of the items.
    shuffled = list(ITEMS)
    random.Random(3).shuffle(shuffled)
    other_order, _ = _build(items=shuffled)
    assert set(first.qids()) == set(other_order.qids())
    # Another seed draws another 12 of the 20.
    other_seed, _ = _build(seed=6)
    singles = lambda rs: [q for q in rs.qids() if q.startswith("single_hop")]  # noqa: E731
    assert singles(other_seed) != singles(first)
    # No cap for a type: all its stable questions.
    full, _ = _build(max_per_type={})
    assert len([q for q in full.qids() if q.startswith("single_hop")]) == 20
    assert DEFAULT_MAX_PER_TYPE == {"single_hop": 12}


def test_cap_draw_takes_the_smallest_hash_ranks() -> None:
    """The draw rule written out again: the 12 smallest sha256('seed:qid') of the stable ones."""
    import hashlib

    rs, _ = _build()
    stable = SINGLE[:20]
    ranked = sorted(stable, key=lambda q: hashlib.sha256(f"5:{q}".encode()).hexdigest())
    expected = sorted(ranked[:12])
    assert [q for q in rs.qids() if q.startswith("single_hop")] == expected


def test_stable_for_marks_standard() -> None:
    rs, _ = _build(max_per_type={})
    marks = {q.qid: q.stable_for for q in rs.questions}
    assert marks[SINGLE[0]] == ["agentic", "standard"]
    assert marks[SINGLE[15]] == ["agentic"]
    assert marks[TWO[0]] == ["agentic", "standard"]
    assert marks[TWO[1]] == ["agentic"]
    assert rs.qids_for("standard") == [
        *SINGLE[:10],
        TWO[0],
        HARD[0],
    ]
    assert rs.qids_for("agentic") == rs.qids()
    assert rs.stable_available["standard"]["single_hop"] == 10


def test_baseline_keeps_every_pass() -> None:
    rs, base = _build(max_per_type={})
    assert base.passes == 3 and base.set_name == rs.name
    for q in rs.questions:
        for system in ("agentic", "standard"):
            rows = base.entries[q.qid][system]
            assert [r.pass_no for r in rows] == [1, 2, 3]
    row = base.entries[SINGLE[0]]["agentic"][0]
    assert row.correct and row.score == 3 and row.status == "answered"
    assert row.cny == pytest.approx(0.01) and row.total_ms == 1000.0
    # A standard miss is stored as it was: SINGLE[15] has a "w" in pass 2.
    row = base.entries[SINGLE[15]]["standard"][1]
    assert row.correct is False and row.score == 2
    assert base.mean_cny("agentic", SINGLE[0]) == pytest.approx(0.01)
    default_set, default_base = _build()
    assert set(default_base.entries) == set(default_set.qids())  # only the set's questions
    amb = base.entries[AMBIG[0]]["agentic"][0]
    assert amb.correct and amb.status == "need_clarification"  # judged by rule, not by score


def test_changed_question_is_noticed() -> None:
    rs, _ = _build()
    check_items(rs, ITEMS)  # unchanged: fine
    changed = [_item(q.qid, question="Another text?") if q.qid == SINGLE[0] else q for q in ITEMS]
    with pytest.raises(ValueError, match=SINGLE[0]):
        check_items(rs, changed)
    with pytest.raises(ValueError, match="not in the question sets"):
        check_items(rs, [i for i in ITEMS if i.qid != rs.qids()[0]])
    assert fingerprint(ITEMS[0]) != fingerprint(_item(ITEMS[0].qid, question="x?"))


def test_round_trip(tmp_path: Path) -> None:
    rs, base = _build()
    write_suite(tmp_path / "set.json", rs)
    write_baseline(tmp_path / "base.json", base)
    assert read_suite(tmp_path / "set.json") == rs
    assert read_baseline(tmp_path / "base.json") == base
    assert b"\r\n" not in (tmp_path / "set.json").read_bytes()


def test_shipped_set_is_consistent() -> None:
    rs = read_suite(SET_PATH)
    base = read_baseline(BASELINE_PATH)
    assert rs.counts == {
        "single_hop": 12,
        "two_hop": 8,
        "colloquial": 6,
        "hard": 12,
        "ambiguous": 2,
    }
    assert len(rs.questions) == 40 and sum(rs.counts.values()) == 40
    assert rs.counts == {t: len([q for q in rs.questions if q.qtype == t]) for t in rs.counts}
    assert rs.system == "agentic" and rs.passes == 3 and rs.seed == 20261023
    items = load_items("hf_docs", types=list(rs.counts))
    check_items(rs, items)  # every question exists, text unchanged
    assert base.set_name == rs.name and base.passes == 3
    assert set(base.entries) == set(rs.qids())
    for q in rs.questions:
        agentic = base.entries[q.qid]["agentic"]
        assert [r.pass_no for r in agentic] == [1, 2, 3] and all(r.correct for r in agentic)
        standard = base.entries[q.qid]["standard"]
        assert ("standard" in q.stable_for) == (
            len(standard) == 3 and all(r.correct for r in standard)
        )
    # The two kinds that the set is about are all there.
    assert {"single_hop", "two_hop", "colloquial", "hard", "ambiguous"} == set(rs.counts)
