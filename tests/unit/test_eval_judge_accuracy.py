"""Spec P3-03 AC27-AC34: the judge-accuracy script (labels from the question sets, confusion
matrix, Wilson intervals, per question type, the fixed-two-rounds comparison, the spend limit).

A fake retriever with scripted results and scripted judge and helper models: no network, no
Elasticsearch, no real model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.eval.judge_accuracy import cli, metrics
from rag.eval.judge_accuracy.records import JudgeFile, load_records
from rag.eval.judge_accuracy.runner import QuestionSpec, collect_records
from rag.eval.questions import LabeledQuestion, load_question_set
from rag.eval.stats import wilson_interval
from rag.testing import FakeChatModel, observe_settings
from rag.testing.agent import (
    hyde_reply,
    judge_reply,
    kb_config,
    multi_query_reply,
    scored_result,
)
from rag.types import RetrievalResult

KB = "hf_docs"


def _page(n: int) -> str:
    return f"{KB}:p{n}.md"


class MapRetriever:
    """`retrieve(question, kb, top_k)` from a table; records every question asked."""

    def __init__(self, results: dict[str, RetrievalResult], default: RetrievalResult | None = None):
        self.results = results
        self.default = default or scored_result(KB, [0.05], pages=[99])
        self.asked: list[str] = []

    def retrieve(self, question: str, kb: str, top_k: int) -> RetrievalResult:
        self.asked.append(question)
        return self.results.get(question, self.default)


def _q(qid: str, question: str, *pages: int, **meta: Any) -> LabeledQuestion:
    return LabeledQuestion(
        qid=qid, question=question, relevant_pages=[_page(n) for n in pages], meta=meta
    )


# Eight questions with known answers. The judge script below is read in this order; a question
# whose chunks all fail the threshold is never sent to the model.
DEV = [
    _q("d1", "q1 gold in top", 1),  # gold returned -> label enough
    _q("d2", "q2 gold missing", 3),  # gold not returned -> label not enough
    _q("d3", "q3 gold in top", 5),  # label enough
    _q("d4", "q4 gold below threshold", 6),  # label not enough (below the threshold)
]
NA = [
    _q("n1", "n1 near miss", type="near_miss"),
    _q("n2", "n2 off topic", type="off_topic"),
]
HOP = [
    _q("h1", "h1 two pages, one found", 1, 2, type="two_hop", requires="all"),
    _q("h2", "h2 two pages, both found", 1, 2, type="two_hop", requires="all"),
]
RESULTS = {
    "q1 gold in top": scored_result(KB, [0.9, 0.8], pages=[1, 2]),
    "q2 gold missing": scored_result(KB, [0.8], pages=[4]),
    "q3 gold in top": scored_result(KB, [0.7], pages=[5]),
    "q4 gold below threshold": scored_result(KB, [0.1, 0.05], pages=[6, 7]),
    "n1 near miss": scored_result(KB, [0.6], pages=[7]),
    "n2 off topic": scored_result(KB, [0.05], pages=[8]),
    "h1 two pages, one found": scored_result(KB, [0.9], pages=[1]),
    "h2 two pages, both found": scored_result(KB, [0.9, 0.8], pages=[1, 2]),
}
JUDGE_SCRIPT = [True, True, False, False, False, True]  # q1 q2 q3 n1 h1 h2


def _specs() -> list[QuestionSpec]:
    return (
        [QuestionSpec("dev", q, "answerable", "single_hop") for q in DEV]
        + [QuestionSpec("na", q, "no_answer", q.meta["type"]) for q in NA]
        + [QuestionSpec("hop", q, "answerable", "two_hop") for q in HOP]
    )


def _collect(**kwargs: Any) -> tuple[JudgeFile, MapRetriever, FakeChatModel]:
    retriever = MapRetriever(RESULTS)
    judge = FakeChatModel(responses=[judge_reply(g) for g in JUDGE_SCRIPT])
    settings = observe_settings()
    file = collect_records(
        _specs(),
        retriever=retriever,
        cfg=kb_config(KB, judge=True),
        settings=settings,
        judge_model=judge,
        **kwargs,
    )
    return file, retriever, judge


# ---- labels ------------------------------------------------------------------------------------


def test_label_enough_means_the_gold_page_is_among_the_returned_chunks() -> None:
    one = _q("a", "q", 1)
    assert metrics.label_enough(one, [_page(1), _page(2)])
    assert not metrics.label_enough(one, [_page(2)])
    assert not metrics.label_enough(one, [])
    # Several labelled pages are alternatives (the same text on two paths) unless the question
    # says all of them are needed (a two-hop question).
    alternatives = _q("b", "q", 1, 2)
    assert metrics.label_enough(alternatives, [_page(2)])
    both = _q("c", "q", 1, 2, requires="all")
    assert not metrics.label_enough(both, [_page(2)])
    assert metrics.label_enough(both, [_page(2), _page(1)])
    # A no-answer question is never enough on round one.
    assert not metrics.label_enough(_q("d", "q"), [_page(1)])


def test_collect_labels_every_question_from_the_sets_and_records_what_the_judge_said() -> None:
    file, retriever, judge = _collect()
    by_qid = {r.qid: r for r in file.records}
    assert [r.qid for r in file.records] == ["d1", "d2", "d3", "d4", "n1", "n2", "h1", "h2"]
    assert {q for q, r in by_qid.items() if r.label_enough} == {"d1", "d3", "h2"}
    # d4's gold page was among the reranked candidates but under the threshold: not enough for
    # the answer step, and the sensitivity flag says the label would differ without the threshold.
    assert by_qid["d4"].label_in_rerank_top_k and not by_qid["d4"].label_enough
    assert by_qid["d4"].n_chunks == 0 and by_qid["d4"].verdict == "not_enough"
    assert by_qid["d1"].verdict == "enough" and by_qid["d1"].used_model
    assert by_qid["d3"].verdict == "partly_enough" and by_qid["d3"].missing == ["q3 gold in top"]
    assert by_qid["n2"].verdict == "not_enough" and not by_qid["n2"].used_model
    assert (by_qid["h1"].qtype, by_qid["n1"].qtype) == ("two_hop", "near_miss")
    assert by_qid["d1"].returned_pages == [_page(1), _page(2)] and by_qid["d1"].top_score == 0.9
    # The model was asked about the six questions that had chunks above the threshold, once each.
    assert len(judge.calls) == 6 and len(retriever.asked) == 8
    assert not file.partial and file.k == 8 and file.threshold == 0.3


# ---- numbers -----------------------------------------------------------------------------------


def test_confusion_accuracy_precision_recall_and_wilson_intervals() -> None:
    file, _, _ = _collect()
    summary = metrics.summarize(file.records, metrics.POLICIES["judge"])
    c = summary.confusion
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 4)
    assert summary.accuracy.successes == 6 and summary.accuracy.n == 8
    assert summary.accuracy.value == pytest.approx(0.75)
    assert (summary.accuracy.low, summary.accuracy.high) == pytest.approx(wilson_interval(6, 8))
    assert summary.precision is not None and (summary.precision.successes, summary.precision.n) == (
        2,
        3,
    )
    assert summary.recall is not None and (summary.recall.successes, summary.recall.n) == (2, 3)
    assert (summary.precision.low, summary.precision.high) == pytest.approx(wilson_interval(2, 3))
    # Without any "enough" prediction there is no precision, only a recall of 0.
    never = metrics.summarize(file.records, metrics.POLICIES["fixed_two_rounds"])
    assert never.precision is None and never.recall is not None and never.recall.successes == 0


def test_the_three_tiers_against_the_label_and_the_split_by_question_type() -> None:
    file, _, _ = _collect()
    table = metrics.tier_table(file.records)
    assert table["enough"] == {"enough_label": 2, "not_enough_label": 1}
    assert table["partly_enough"] == {"enough_label": 1, "not_enough_label": 2}
    assert table["not_enough"] == {"enough_label": 0, "not_enough_label": 2}
    types = metrics.by_type(file.records, metrics.POLICIES["judge"])
    assert set(types) == {"single_hop", "near_miss", "off_topic", "two_hop"}
    single = types["single_hop"]
    assert (single.confusion.tp, single.confusion.fp, single.confusion.fn, single.confusion.tn) == (
        1,
        1,
        1,
        1,
    )
    assert types["two_hop"].confusion.tp == 1 and types["two_hop"].confusion.tn == 1
    assert types["off_topic"].accuracy.value == 1.0 and types["off_topic"].accuracy.n == 1


def test_the_judge_is_compared_with_the_rules_alone_and_with_fixed_two_rounds() -> None:
    file, _, _ = _collect()
    rows = {row.policy: row for row in metrics.policy_rows(file.records)}
    # Rules alone: any chunk above the threshold is enough. It never wastes a round on a good
    # result, but stops early on every wrong one that scored high (q2, n1, h1).
    assert (rows["rules_only"].accuracy.successes, rows["rules_only"].missed_corrections) == (5, 3)
    assert rows["rules_only"].wasted_rounds == 0
    # The model judge: one wasted round (q3), one missed correction (q2).
    assert (rows["judge"].accuracy.successes, rows["judge"].wasted_rounds) == (6, 1)
    assert rows["judge"].missed_corrections == 1 and rows["judge"].rounds == 13
    # Fixed two rounds: every question gets the second round; the three that were already
    # enough waste it, nothing is missed.
    fixed = rows["fixed_two_rounds"]
    assert (fixed.wasted_rounds, fixed.missed_corrections, fixed.rounds) == (3, 0, 16)
    assert fixed.accuracy.successes == 5
    text = metrics.render_report(file)
    for needle in ("accuracy", "precision", "recall", "fixed_two_rounds", "single_hop", "d2"):
        assert needle in text


def test_wilson_interval_is_the_one_of_the_stats_module() -> None:
    assert metrics.rate(0, 0).low == 0.0 and metrics.rate(0, 0).high == 1.0
    r = metrics.rate(3, 10)
    assert (r.low, r.high) == pytest.approx(wilson_interval(3, 10)) and r.value == 0.3


# ---- rescue: would another round have found the page? ------------------------------------------


def test_the_rescue_run_asks_round_two_and_three_for_questions_that_were_not_enough() -> None:
    # d2: the gold page p3 is found by a phrasing in round two. d4: round two finds nothing, the
    # HyDE passage of round three finds p6. Questions that were already enough are not asked.
    passage = "In gold below threshold mode the page six explains the setting and its gold default."
    results = {
        **RESULTS,
        "q2 gold other words": scored_result(KB, [0.8], pages=[3]),
        "q4 gold other words": scored_result(KB, [0.1], pages=[9]),
        passage: scored_result(KB, [0.7], pages=[6]),
    }
    helper = FakeChatModel(
        responses=[
            multi_query_reply((1, ["q2 gold other words"])),
            multi_query_reply((1, ["q4 gold other words"])),
            hyde_reply((1, passage)),
        ]
    )
    file = collect_records(
        [s for s in _specs() if s.question.qid in {"d2", "d4"}],
        retriever=MapRetriever(results),
        cfg=kb_config(KB, judge=True, rewrite=True),
        settings=observe_settings(),
        judge_model=FakeChatModel(responses=[judge_reply(True)]),
        helper_model=helper,
        rescue=True,
    )
    by_qid = {r.qid: r for r in file.records}
    assert by_qid["d2"].rescue is not None and by_qid["d2"].rescue.round2_found is True
    assert by_qid["d2"].rescue.round3_found is None  # no round three needed
    assert by_qid["d4"].rescue is not None
    assert (by_qid["d4"].rescue.round2_found, by_qid["d4"].rescue.round3_found) == (False, True)
    assert len(helper.calls) == 3
    summary = metrics.rescue_summary(file.records)
    assert summary is not None
    assert (summary.n, summary.round2.successes, summary.round3.successes) == (2, 1, 2)
    assert metrics.rescue_summary([r for r in file.records if r.rescue is None]) is None


# ---- the spend limit and the command line --------------------------------------------------------


def test_the_run_stops_when_the_spend_limit_is_reached() -> None:
    spent = iter(range(100))

    def ledger() -> float:
        return float(next(spent)) * 0.5  # every look at the ledger finds 0.5 CNY more spent

    file, retriever, _ = _collect(max_cny=1.2, spent_cny=ledger)
    assert file.partial and len(file.records) < 8
    assert len(retriever.asked) == len(file.records)  # nothing is asked after the stop


def _write_sets(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name, questions in (("dev", DEV), ("na", NA), ("hop", HOP)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps(q.model_dump(mode="json"), ensure_ascii=False) + "\n" for q in questions
            ),
            encoding="utf-8",
            newline="\n",
        )
        paths[name] = path
    return paths


def test_the_command_line_runs_writes_the_records_and_reports_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_sets(tmp_path)
    judge = FakeChatModel(responses=[judge_reply(g) for g in JUDGE_SCRIPT])
    monkeypatch.setattr(cli, "build_retriever", lambda *a, **k: MapRetriever(RESULTS))
    monkeypatch.setattr(cli, "build_models", lambda settings, rescue: (judge, None))
    monkeypatch.setattr(cli, "spent_cny", lambda settings: 0.0)
    out = tmp_path / "out"
    code = cli.main(
        [
            "run",
            "--kb",
            KB,
            *(arg for name, p in paths.items() for arg in ("--set", f"{name}={p}")),
            "--out",
            str(out),
        ],
        settings=observe_settings(),
    )
    assert code == 0
    shown = capsys.readouterr().out
    assert "accuracy" in shown and "fixed_two_rounds" in shown
    records = load_records(out / "records.json")
    assert len(records.records) == 8 and records.question_sets[0].count == 4
    assert (out / "report.md").read_text("utf-8").startswith("# Judge accuracy")
    # `report` recomputes the same text from the records; no model, no retriever.
    monkeypatch.setattr(cli, "build_models", lambda *a: pytest.fail("no model for a report"))
    assert cli.main(["report", str(out / "records.json")], settings=observe_settings()) == 0
    assert "accuracy" in capsys.readouterr().out
    # The set files are loadable with the ordinary loader (the same schema).
    assert len(load_question_set(paths["dev"], kb=KB).questions) == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--kb", "no_such_kb", "--set", "x=a.jsonl", "--out", "o"],
        ["run", "--kb", KB, "--set", "missing-equals", "--out", "o"],
        ["run", "--kb", KB, "--set", "x=does_not_exist.jsonl", "--out", "o"],
        ["run", "--kb", KB, "--set", "x=a.jsonl", "--out", "o", "--max-cny", "0"],
        ["run", "--kb", KB, "--set", "x=a.jsonl", "--out", "o", "--limit", "0"],
        ["report", "no_such_file.json"],
    ],
)
def test_bad_arguments_exit_with_code_2_and_one_line_on_stderr(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv, settings=observe_settings()) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.strip().splitlines()) == 1


def test_a_set_file_that_mixes_answerable_and_no_answer_questions_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mixed = tmp_path / "mixed.jsonl"
    rows = [_q("a", "has gold", 1), _q("b", "has none")]
    mixed.write_text(
        "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in rows), encoding="utf-8"
    )
    code = cli.main(
        ["run", "--kb", KB, "--set", f"mixed={mixed}", "--out", str(tmp_path / "o")],
        settings=observe_settings(),
    )
    assert code == 2 and "mixes" in capsys.readouterr().err
