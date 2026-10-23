"""Spec EVAL-3 AC2 to AC6: pass^k, pass@k, McNemar, flips, cuts, stable questions."""

from __future__ import annotations

import pytest

from rag.eval.agentic.repeat.analysis import analyze, stable_payload
from rag.testing.repeat import passes_from, qlist_for


def _q(n: int, qtype: str = "single_hop") -> list[str]:
    return [f"{qtype}:{i:02d}" for i in range(n)]


# Six single-hop questions, three passes; one character per pass (see rag.testing.repeat).
STANDARD = ["ccc", "ccc", "ccw", "wcc", "www", "cww"]
AGENTIC = ["ccc", "ccc", "ccc", "ccc", "ccw", "www"]


def _six() -> tuple[dict[str, dict[str, str]], list[str]]:
    qids = _q(6)
    spec = {
        "standard": dict(zip(qids, STANDARD, strict=True)),
        "agentic": dict(zip(qids, AGENTIC, strict=True)),
    }
    return spec, qids


def test_pass_all_and_any_arithmetic() -> None:
    spec, qids = _six()
    result = analyze(passes_from(spec), qlist_for(qids))
    assert result.passes == 3 and result.questions == 6 and not result.incomplete
    group = result.groups["single_hop"]
    std, agent = group.systems["standard"], group.systems["agentic"]
    # pass^3: standard ccc, ccc only; agentic ccc x4.  pass@3: everything but standard www.
    assert (std.pass_all.count, std.pass_all.n) == (2, 6)
    assert (agent.pass_all.count, agent.pass_all.n) == (4, 6)
    assert std.pass_any.count == 5 and agent.pass_any.count == 5
    assert std.pass_all.rate == pytest.approx(2 / 6)
    assert 0.0 < std.pass_all.low < std.pass_all.rate < std.pass_all.high < 1.0  # Wilson interval
    # One number per pass, as a single run would have shown it.
    assert [p.count for p in std.per_pass] == [4, 4, 3]
    assert [p.count for p in agent.per_pass] == [5, 5, 4]
    # The pooled groups exist and equal the single type here.
    assert result.groups["all"].systems["standard"].pass_all.count == 2
    assert result.groups["old"].n == 6


def test_mcnemar_on_pass_all() -> None:
    spec, qids = _six()
    paired = analyze(passes_from(spec), qlist_for(qids)).groups["single_hop"].paired
    assert (paired.improved, paired.worsened, paired.both) == (
        2,
        0,
        2,
    )  # agentic better on q02, q03
    assert paired.p_value == pytest.approx(0.5)
    assert "no reliable difference" in paired.verdict


def test_adjacent_pass_flips() -> None:
    spec, qids = _six()
    group = analyze(passes_from(spec), qlist_for(qids)).groups["single_hop"]
    by = {(f.system, f.from_pass): f for f in group.flips}
    assert by[("standard", 1)].changed == 2  # q03 w->c, q05 c->w
    assert (by[("standard", 1)].to_right, by[("standard", 1)].to_wrong) == (1, 1)
    assert by[("standard", 1)].qids == ["single_hop:03", "single_hop:05"]
    assert by[("standard", 2)].changed == 1  # q02 c->w
    assert by[("agentic", 1)].changed == 0
    assert by[("agentic", 2)].changed == 1  # q04 c->w
    assert (by[("agentic", 2)].to_right, by[("agentic", 2)].to_wrong) == (0, 1)
    assert group.max_flips == 2
    assert group.mean_flips == pytest.approx((2 + 1 + 0 + 1) / 4)


def test_gap_against_noise_verdict() -> None:
    spec, qids = _six()
    group = analyze(passes_from(spec), qlist_for(qids)).groups["single_hop"]
    assert group.gap == 2 and group.max_flips == 2
    assert group.gap_larger_than_noise is False  # a gap of 2 is not more than 2 flips
    assert group.gap_per_pass == [1, 1, 1]  # agentic minus standard correct, pass by pass

    clear = {
        "standard": {q: "www" for q in _q(5)},
        "agentic": {q: "ccc" for q in _q(5)},
    }
    result = analyze(passes_from(clear), qlist_for(_q(5)))
    assert result.groups["single_hop"].gap == 5
    assert result.groups["single_hop"].max_flips == 0
    assert result.groups["single_hop"].gap_larger_than_noise is True

    same = {"standard": {q: "ccc" for q in _q(3)}, "agentic": {q: "ccc" for q in _q(3)}}
    flat = analyze(passes_from(same), qlist_for(_q(3))).groups["single_hop"]
    assert flat.gap == 0 and flat.gap_larger_than_noise is False


def test_ambiguous_uses_asked_back() -> None:
    qids = _q(4, "ambiguous")
    spec = {
        "standard": dict(zip(qids, ["www", "wwc", "www", "www"], strict=True)),
        "agentic": dict(zip(qids, ["ccc", "ccc", "ccw", "ccc"], strict=True)),
    }
    group = analyze(passes_from(spec), qlist_for(qids)).groups["ambiguous"]
    assert group.systems["standard"].pass_all.count == 0
    assert group.systems["standard"].pass_any.count == 1
    assert group.systems["agentic"].pass_all.count == 3
    # no_answer questions are not part of the analysis
    extra = passes_from({"standard": {"no_answer:00": "rr"}, "agentic": {"no_answer:00": "rr"}})
    assert analyze(extra, qlist_for(["no_answer:00"])).groups == {}


def test_deadline_cut_is_listed_and_not_correct() -> None:
    qids = _q(3)
    spec = {
        "standard": dict(zip(qids, ["ccc", "ccc", "ccc"], strict=True)),
        "agentic": dict(zip(qids, ["cxc", "ctc", "cec"], strict=True)),
    }
    result = analyze(passes_from(spec), qlist_for(qids))
    agent = result.groups["single_hop"].systems["agentic"]
    assert agent.pass_all.count == 0  # a cut, a timeout and an error are not correct
    found = {(i.qid, i.pass_no, i.reason, i.system) for i in result.incidents}
    assert found == {
        ("single_hop:00", 2, "deadline_truncated", "agentic"),
        ("single_hop:01", 2, "generation_error", "agentic"),
        ("single_hop:02", 2, "error", "agentic"),
    }
    # Each of the three questions flips c->bad and bad->c; none of the flips is "clean".
    flips = [f for f in result.groups["single_hop"].flips if f.system == "agentic"]
    assert sum(f.changed for f in flips) == 6
    assert sum(f.changed_clean for f in flips) == 0
    # A truncated answer is never counted as a refusal-type answer anywhere: it is just not correct.
    assert result.groups["single_hop"].systems["standard"].pass_all.count == 3


def test_a_cut_ambiguous_answer_is_not_asking_back() -> None:
    qids = _q(1, "ambiguous")
    spec = {"standard": {qids[0]: "ccc"}, "agentic": {qids[0]: "cxc"}}
    result = analyze(passes_from(spec), qlist_for(qids))
    assert result.groups["ambiguous"].systems["agentic"].pass_all.count == 0
    assert [i.reason for i in result.incidents] == ["deadline_truncated"]


def test_incomplete_questions_are_left_out() -> None:
    qids = _q(4)
    spec = {
        "standard": dict(zip(qids, ["ccc", "ccn", "ccc", "ccc"], strict=True)),
        "agentic": dict(zip(qids, ["ccc", "ccc", "ccu", "ccc"], strict=True)),
    }
    result = analyze(passes_from(spec), qlist_for(qids))
    assert result.questions == 2
    assert {x.qid: x.reason for x in result.incomplete} == {
        "single_hop:01": "standard has no record in pass 3",
        "single_hop:02": "agentic answer in pass 3 is not judged",
    }
    assert result.groups["single_hop"].n == 2


def test_stable_questions_file() -> None:
    spec, qids = _six()
    qlist = qlist_for(qids)
    result = analyze(passes_from(spec), qlist)
    assert result.stable["standard"]["single_hop"] == ["single_hop:00", "single_hop:01"]
    assert result.stable["agentic"]["single_hop"] == [
        "single_hop:00",
        "single_hop:01",
        "single_hop:02",
        "single_hop:03",
    ]
    assert result.stable["both"]["single_hop"] == ["single_hop:00", "single_hop:01"]
    payload = stable_payload(result, qlist)
    assert payload["passes"] == 3 and payload["question_list_seed"] == 1
    assert payload["universe"] == 6 and payload["agentic"] == result.stable["agentic"]


def test_two_passes_give_pass_all_of_two() -> None:
    spec, qids = _six()
    two = [
        p for p in passes_from({s: {q: v[:2] for q, v in rows.items()} for s, rows in spec.items()})
    ]
    result = analyze(two, qlist_for(qids))
    assert result.passes == 2
    assert result.groups["single_hop"].systems["standard"].pass_all.count == 3  # q00, q01, q02


def test_needs_two_passes() -> None:
    with pytest.raises(ValueError, match="two passes"):
        analyze(
            passes_from({"standard": {"single_hop:00": "c"}, "agentic": {"single_hop:00": "c"}}),
            qlist_for(["single_hop:00"]),
        )


def test_clean_flips_leave_out_questions_with_a_cut_record() -> None:
    qids = _q(2)
    spec = {
        "standard": dict(zip(qids, ["ccc", "ccc"], strict=True)),
        "agentic": dict(zip(qids, ["cxc", "ccw"], strict=True)),
    }
    flips = {
        (f.system, f.from_pass): f
        for f in analyze(passes_from(spec), qlist_for(qids)).groups["single_hop"].flips
    }
    assert (flips[("agentic", 1)].changed, flips[("agentic", 1)].changed_clean) == (1, 0)
    assert (flips[("agentic", 2)].changed, flips[("agentic", 2)].changed_clean) == (2, 1)
    assert flips[("standard", 1)].changed == 0
