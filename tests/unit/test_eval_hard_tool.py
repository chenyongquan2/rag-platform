"""Spec EVAL-2 AC3 (checker logic), AC6-AC10: the sets registration, the clarification rule, the
report, the `check` and `self-check` commands. Fakes and tiny corpora only; no model, no network."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from rag.eval.agentic.analysis import ANSWERABLE_TYPES, loop_report, per_type_summary
from rag.eval.agentic.records import SystemRecord
from rag.eval.agentic.report import project_cost, render_report
from rag.eval.agentic.sets import DEFAULT_TYPES, HARD_TYPES, QTYPES, load_items
from rag.eval.hard import (
    asked_back,
    clarification_summary,
    corpus_report,
    export_self_check,
    load_hard,
    validate_questions,
)
from rag.eval.hard import cli as hard_cli
from rag.eval.questions import LabeledQuestion

A = "hf_docs:org/repo/blob/main/a.md"
B = "hf_docs:org/repo/blob/main/b.md"
C = "hf_docs:org/repo/blob/main/c.md"

CORPUS = {
    A: "The Alpha model is a successor of Beta. Alpha uses a rotary scheme.",
    B: "Beta was pre-trained in two stages with a lightweight querying transformer.",
    C: "Gamma is unrelated. Alpha is mentioned here, and so are two stages of something.",
}


def fact(page: str, evidence: str, key: str, loose: list[list[Any]]) -> dict[str, Any]:
    return {"page": page, "evidence": evidence, "key": key, "loose": loose}


def hard_question(**changes: Any) -> LabeledQuestion:
    meta: dict[str, Any] = {
        "type": "hard",
        "requires": "all",
        "shape": "bridge",
        "failure_type": "bridge",
        "modelled_on": "hf_docs_two_hop:08",
        "style": [],
        "facts": [
            fact(
                A,
                "The Alpha model is a successor of Beta.",
                r"successor of Beta",
                [["Alpha", "Beta"]],
            ),
            fact(
                B,
                "pre-trained in two stages",
                r"pre-trained in two stages",
                [["pre-trained", "two stages"]],
            ),
        ],
        "reasoning": "Alpha names Beta; Beta says how it is trained.",
        "checked_single_page": "test",
    }
    meta.update(changes.pop("meta", {}))
    fields: dict[str, Any] = {
        "qid": "hf_docs_hard:99",
        "question": "Alpha succeeds which model, and how was that model trained?",
        "answer": "Beta; in two stages.",
        "relevant_pages": [A, B],
        "meta": meta,
    }
    fields.update(changes)
    return LabeledQuestion(**fields)


# ---- AC3: the checker --------------------------------------------------------------------------


def test_corpus_report_accepts_a_question_no_single_page_answers() -> None:
    [finding] = corpus_report([hard_question()], CORPUS)
    assert finding.problems == [] and finding.ok
    assert finding.searched_pages == 3


def test_corpus_report_finds_each_kind_of_problem() -> None:
    wrong_evidence = hard_question(
        meta={
            "facts": [
                fact(
                    A, "this text is nowhere on the page", r"successor of Beta", [["Alpha", "Beta"]]
                ),
                fact(
                    B,
                    "pre-trained in two stages",
                    r"pre-trained in two stages",
                    [["pre-trained", "two stages"]],
                ),
            ]
        }
    )
    key_misses_its_page = hard_question(
        meta={
            "facts": [
                fact(
                    A,
                    "The Alpha model is a successor of Beta.",
                    r"no such words",
                    [["Alpha", "Beta"]],
                ),
                fact(
                    B,
                    "pre-trained in two stages",
                    r"pre-trained in two stages",
                    [["pre-trained", "two stages"]],
                ),
            ]
        }
    )
    key_on_another_page = hard_question(
        meta={
            "facts": [
                fact(
                    A,
                    "The Alpha model is a successor of Beta.",
                    r"successor of Beta",
                    [["Alpha", "Beta"]],
                ),
                fact(
                    B, "pre-trained in two stages", r"two stages", [["pre-trained", "two stages"]]
                ),
            ]
        }
    )
    one_page_has_everything = hard_question(
        meta={
            "facts": [
                fact(
                    A,
                    "The Alpha model is a successor of Beta.",
                    r"successor of Beta",
                    [["Alpha", "Alpha"]],
                ),
                fact(
                    C,
                    "two stages of something",
                    r"two stages of something",
                    [["Alpha", "two stages"]],
                ),
            ]
        },
        relevant_pages=[A, C],
    )
    loose_misses_gold = hard_question(
        meta={
            "facts": [
                fact(
                    A,
                    "The Alpha model is a successor of Beta.",
                    r"successor of Beta",
                    [["Zeta", "Beta"]],
                ),
                fact(
                    B,
                    "pre-trained in two stages",
                    r"pre-trained in two stages",
                    [["pre-trained", "two stages"]],
                ),
            ]
        }
    )
    found = {
        name: corpus_report([q], CORPUS)[0].problems
        for name, q in {
            "evidence": wrong_evidence,
            "key": key_misses_its_page,
            "other": key_on_another_page,
            "joint": one_page_has_everything,
            "loose": loose_misses_gold,
        }.items()
    }
    assert any("evidence" in p for p in found["evidence"])
    assert any("key pattern does not match its own page" in p for p in found["key"])
    assert any("also matches" in p and C in p for p in found["other"])
    assert any("single page" in p for p in found["joint"])
    assert any("loose" in p and "gold page" in p for p in found["loose"])


def test_declared_copies_are_allowed_and_must_be_complete() -> None:
    twin = "hf_docs:org/repo/blob/main/b_copy.md"
    corpus = {**CORPUS, twin: CORPUS[B]}
    plain = hard_question()
    assert any("also matches" in p for p in corpus_report([plain], corpus)[0].problems)
    declared = hard_question()
    declared.meta["facts"][1]["also_on"] = [twin]
    assert corpus_report([declared], corpus)[0].problems == []


def test_validate_questions_flags_malformed_ones() -> None:
    assert validate_questions([hard_question()]) == []
    one_page = hard_question(relevant_pages=[A], meta={"facts": [hard_question().meta["facts"][0]]})
    no_key = hard_question()
    no_key.meta["facts"][0]["key"] = ""
    wrong_requires = hard_question(meta={"requires": "any"})
    unknown_shape = hard_question(meta={"shape": "puzzle"})
    for bad in (one_page, no_key, wrong_requires, unknown_shape):
        assert validate_questions([bad]), bad.meta
    assert validate_questions([hard_question(), hard_question()])  # duplicate qid


# ---- AC6: registration in the question sets ------------------------------------------------------


def test_hard_types_load_by_themselves() -> None:
    hard = load_items("hf_docs", types=["hard"])
    assert len(hard) == 25 and {i.qtype for i in hard} == {"hard"}
    assert all(i.answerable and i.requires == "all" and len(i.relevant_pages) >= 2 for i in hard)
    assert all(i.set == "hard" and i.reference.strip() for i in hard)
    ambiguous = load_items("hf_docs", types=["ambiguous"])
    assert len(ambiguous) == 10 and {i.qtype for i in ambiguous} == {"ambiguous"}
    assert all(not i.answerable and not i.relevant_pages for i in ambiguous)
    assert all(i.reference.lower().startswith("clarification expected") for i in ambiguous)
    both = load_items("hf_docs", types=["hard", "ambiguous"])
    assert [i.qtype for i in both] == ["hard"] * 25 + ["ambiguous"] * 10
    pilot = load_items("hf_docs", take={"hard": 3, "ambiguous": 2})
    assert [i.qtype for i in pilot] == ["hard"] * 3 + ["ambiguous"] * 2


def test_default_selection_is_unchanged() -> None:
    assert DEFAULT_TYPES == ("single_hop", "two_hop", "colloquial", "no_answer")
    assert HARD_TYPES == ("hard", "ambiguous")
    assert (*DEFAULT_TYPES, *HARD_TYPES) == QTYPES
    items = load_items("hf_docs")
    assert len(items) == 150 and {i.qtype for i in items} == set(DEFAULT_TYPES)
    assert "ambiguous" not in ANSWERABLE_TYPES and "hard" in ANSWERABLE_TYPES


# ---- AC7: the rule ------------------------------------------------------------------------------


def rec(system: str, qid: str, qtype: str = "ambiguous", **kw: Any) -> SystemRecord:
    answerable = qtype in {"hard", "single_hop"}
    fields: dict[str, Any] = {
        "system": system,
        "qid": qid,
        "qtype": qtype,
        "question": f"Question {qid}?",
        "reference": "ref" if answerable else "Clarification expected: which one?",
        "relevant_pages": [f"g{qid}", f"h{qid}"] if answerable else [],
        "requires": "all" if answerable else "any",
        "answerable": answerable,
        "status": "answered",
        "found": True,
        "answered": True,
        "text": "It is 10 seconds.",
    }
    fields.update(kw)
    return SystemRecord(**fields)


def test_asked_back_rule() -> None:
    graph_asked = rec(
        "agentic",
        "a",
        status="need_clarification",
        found=False,
        answered=False,
        clarification=True,
        text="",
    )
    assert asked_back(graph_asked)
    assert asked_back(
        rec("agentic", "b", status="need_clarification", clarification=False, text="")
    )
    assert not asked_back(rec("agentic", "c"))  # answered one reading
    assert not asked_back(
        rec("standard", "d", status="not_found", found=False, answered=False, text="")
    )
    assert not asked_back(
        rec("standard", "e", status="error", found=False, answered=False, text="", error="Timeout")
    )
    # The standard system has no clarification path; a reply that ends by asking counts.
    assert asked_back(rec("standard", "f", text="Which library do you mean: datasets or the Hub?"))
    assert asked_back(rec("standard", "g", text="Do you mean Transformers or Diffusers？"))  # noqa: RUF001
    # A long answer that merely ends with a question does not count.
    long_answer = "The default is 10 seconds. " * 12 + "Does that help?"
    assert not asked_back(rec("standard", "h", text=long_answer))
    assert not asked_back(rec("standard", "i", text="It is 10 seconds. Nothing else applies."))
    # A reply that gives one reading's value and then asks does not count; a framing sentence does.
    assert not asked_back(rec("standard", "k", text="The default is 10 seconds. Does that help?"))
    assert not asked_back(rec("standard", "k2", text="Use `HF_HOME`. Is that what you need?"))
    assert asked_back(
        rec("standard", "l", text="That depends on the library. Which one do you mean?")
    )
    # An empty reply is not a question.
    assert not asked_back(rec("standard", "j", text="", answered=False))


def test_clarification_summary_with_wilson_intervals() -> None:
    records = [
        rec(
            "agentic", "1", status="need_clarification", clarification=True, text="", answered=False
        ),
        rec(
            "agentic", "2", status="need_clarification", clarification=True, text="", answered=False
        ),
        rec("agentic", "3"),
        rec("standard", "1", text="Which one do you mean?"),
        rec("standard", "2"),
        rec("standard", "3"),
        rec("standard", "x", qtype="hard"),  # other types are not counted
    ]
    summary = clarification_summary(records)
    assert (summary["agentic"].count, summary["agentic"].n) == (2, 3)
    assert (summary["standard"].count, summary["standard"].n) == (1, 3)
    low, high = summary["agentic"].low, summary["agentic"].high
    assert 0.2 < low < 0.67 < high < 0.95
    assert clarification_summary([]) == {}


# ---- AC8: the report and the projection ---------------------------------------------------------


def test_report_with_hard_and_ambiguous_records() -> None:
    records = [
        rec("standard", "h1", "hard"),
        rec("agentic", "h1", "hard"),
        rec("standard", "a1", "ambiguous"),
        rec(
            "agentic",
            "a1",
            "ambiguous",
            status="need_clarification",
            clarification=True,
            text="",
            answered=False,
        ),
        rec("standard", "s1", "single_hop", relevant_pages=["g"], requires="any"),
        rec("agentic", "s1", "single_hop", relevant_pages=["g"], requires="any"),
    ]
    text = render_report(records, [])
    assert "| hard | 1 | 1 |" in text and "| ambiguous | 1 | 1 |" in text
    assert "Ambiguous questions" in text and "asked back" in text.lower()
    assert "1/1" in text  # the graph asked back on its one ambiguous question
    summary = per_type_summary(records, [])
    assert "ambiguous" in summary and summary["ambiguous"].correct is None
    loop = loop_report(records)
    assert "ambiguous" not in loop.false_clarification
    assert loop.clarified_questions == []  # an ambiguous question is not an answerable one
    projection = project_cost(records, {"hard": 2, "ambiguous": 3}, judge_cny_per_call=0.01)
    assert projection.judge == pytest.approx(2 * 2 * 2 * 0.01)  # only hard questions are judged


def test_render_report_without_hard_records_has_no_ambiguous_section() -> None:
    records = [rec("standard", "s1", "single_hop"), rec("agentic", "s1", "single_hop")]
    assert "Ambiguous questions" not in render_report(records, [])


# ---- AC9: the check command ---------------------------------------------------------------------


def write_set(path: Path, questions: list[LabeledQuestion]) -> None:
    lines = [json.dumps(q.model_dump(mode="json"), ensure_ascii=False) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_check_command_passes_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(hard_cli, "load_corpus", lambda data_dir=None: CORPUS)
    good = tmp_path / "good.jsonl"
    write_set(good, [hard_question()])
    assert hard_cli.main(["check", "--file", str(good)]) == 0
    out = capsys.readouterr().out
    assert "hf_docs_hard:99" in out and "ok" in out

    broken = hard_question(qid="hf_docs_hard:98")
    broken.meta["facts"][0]["evidence"] = "text that is not on the page"
    bad = tmp_path / "bad.jsonl"
    write_set(bad, [hard_question(), broken])
    assert hard_cli.main(["check", "--file", str(bad)]) == 1
    assert "evidence" in capsys.readouterr().out

    monkeypatch.setattr(hard_cli, "load_corpus", lambda data_dir=None: None)
    assert hard_cli.main(["check", "--file", str(good)]) == 2
    assert "corpus" in capsys.readouterr().err


# ---- AC10: the owner's blind sheet ---------------------------------------------------------------


def test_self_check_sheet_hides_the_reference(tmp_path: Path) -> None:
    questions = load_hard()
    sheet, key = export_self_check(questions, n=10, seed=0, out_dir=tmp_path / "a")
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig", newline="")))
    keys = list(csv.DictReader(key.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == len(keys) == 10
    assert list(rows[0]) == ["sample_id", "question", "your_answer", "your_pages", "comment"]
    assert all(r["your_answer"] == "" and r["your_pages"] == "" for r in rows)
    sheet_text = sheet.read_text(encoding="utf-8-sig")
    by_qid = {q.qid: q for q in questions}
    for k in keys:
        q = by_qid[k["qid"]]
        assert q.meta["type"] == "hard"
        assert k["reference_answer"] == q.answer
        assert all(p in k["gold_pages"] for p in q.relevant_pages)
        assert k["owner_matches_reference"] == ""
        assert q.answer not in sheet_text
        assert all(p not in sheet_text for p in q.relevant_pages)
    assert [r["question"] for r in rows] == [by_qid[k["qid"]].question for k in keys]
    assert len({k["qid"] for k in keys}) == 10
    # The same seed gives the same questions; another seed gives another draw.
    _, key2 = export_self_check(questions, n=10, seed=0, out_dir=tmp_path / "b")
    again = [r["qid"] for r in csv.DictReader(key2.open(encoding="utf-8-sig", newline=""))]
    assert again == [k["qid"] for k in keys]
    _, key3 = export_self_check(questions, n=10, seed=1, out_dir=tmp_path / "c")
    other = [r["qid"] for r in csv.DictReader(key3.open(encoding="utf-8-sig", newline=""))]
    assert other != again
    assert (tmp_path / "a" / "self_check_rubric.txt").is_file()
    with pytest.raises(ValueError, match="at most 25"):
        export_self_check(questions, n=26, seed=0, out_dir=tmp_path / "d")


def test_self_check_command_writes_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert hard_cli.main(["self-check", "--n", "10", "--seed", "0", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "self_check_sheet.csv").is_file() and (
        tmp_path / "self_check_key.csv"
    ).is_file()
    assert "self_check_sheet.csv" in capsys.readouterr().out
    assert hard_cli.main(["self-check", "--n", "0", "--out", str(tmp_path)]) == 2
    assert "error" in capsys.readouterr().err


def test_clarify_report_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from rag.eval.agentic.records import append_line, system_file

    for r in [
        rec(
            "agentic",
            "a1",
            status="need_clarification",
            clarification=True,
            text="",
            answered=False,
        ),
        rec("agentic", "a2"),
        rec("standard", "a1"),
        rec("standard", "a2", text="Which one do you mean?"),
    ]:
        append_line(system_file(tmp_path, r.system), r)
    assert hard_cli.main(["clarify-report", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "agentic" in out and "1/2" in out
    assert hard_cli.main(["clarify-report", "--out", str(tmp_path / "missing")]) == 2


# ---- review round 1: co-mention search, projection, run comparison ----------------


def test_pages_that_mention_every_term_must_be_read_and_cleared() -> None:
    d = "hf_docs:org/repo/blob/main/d.md"
    corpus = {**CORPUS, d: "Alpha " + "filler " * 40 + "Beta, two stages, pre of trained."}
    [finding] = corpus_report([hard_question()], corpus)
    assert finding.co_mentions == [d]
    assert any("read_and_cleared" in p and d in p for p in finding.problems)
    cleared = hard_question(meta={"read_and_cleared": {d: "only lists the words, states no fact"}})
    assert corpus_report([cleared], corpus)[0].problems == []
    stale = hard_question(meta={"read_and_cleared": {C: "not a co-mention"}})
    assert any("not co-mentions" in p for p in corpus_report([stale], CORPUS)[0].problems)


def test_subjects_narrow_the_co_mention_search() -> None:
    d = "hf_docs:org/repo/blob/main/d.md"
    corpus = {**CORPUS, d: "Alpha " + "filler " * 40 + "Beta, two stages, pre of trained."}
    q = hard_question()
    q.meta["facts"][0]["subjects"] = ["Omega"]
    assert corpus_report([q], corpus)[0].co_mentions == []


def test_projection_prices_hard_types_only_when_the_records_have_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag.eval.agentic import cli as agentic_cli
    from rag.eval.agentic.records import append_line, system_file

    def write(records: list[SystemRecord]) -> None:
        for r in records:
            append_line(system_file(tmp_path, r.system), r)

    write([rec("standard", "s1", "single_hop"), rec("agentic", "s1", "single_hop")])
    assert agentic_cli.main(["project", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "single_hop" in out and "hard" not in out and "ambiguous" not in out
    write([rec("standard", "h1", "hard"), rec("agentic", "h1", "hard")])
    assert agentic_cli.main(["project", "--out", str(tmp_path)]) == 0
    assert "hard" in capsys.readouterr().out


def test_run_comparison_counts_only_no_answer_questions_as_refusals() -> None:
    from rag.eval.agentic.runs import compare_runs

    def run() -> list[SystemRecord]:
        return [
            rec("agentic", "n1", "no_answer", answered=False, text="", found=False),
            rec("agentic", "a1", "ambiguous"),  # answered instead of asking back
        ]

    report = compare_runs(run(), run(), [], [], system="agentic")
    assert report.refusal is not None and report.refusal.n == 1
