# ruff: noqa: RUF001 (the pre-score comments start with a full-width colon)
"""Spec EVAL-1 AC3 to AC6: top-up of the two-answer sheet, agreement with the judge, pre-score
helpers and the `human-agreement` command."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rag.eval.agentic import cli
from rag.eval.agentic.human import (
    agentic_agreement,
    export_human_sheet,
    fill_prescores,
    prescore_items,
)
from rag.eval.agentic.records import JudgeLine, SystemRecord, append_line, judge_file, system_file

TAG = {"standard": "Alpha", "agentic": "Beta"}


def _pair(qtype: str, n: int) -> list[SystemRecord]:
    return [
        SystemRecord(
            system=system,  # type: ignore[arg-type]
            qid=f"{qtype}:{i:02d}",
            qtype=qtype,
            question=f"Question {qtype} {i}?",
            reference=f"Reference {i}.",
            answered=True,
            found=True,
            text=f"{TAG[system]} says {i}.",
        )
        for i in range(n)
        for system in ("standard", "agentic")
    ]


def _records() -> list[SystemRecord]:
    return [*_pair("single_hop", 40), *_pair("two_hop", 20), *_pair("colloquial", 15)]


def _lines(
    records: list[SystemRecord], low: set[tuple[str, str]], *, run: int = 1
) -> list[JudgeLine]:
    """Score 1 for the (system, qid) pairs in `low`, 3 for every other answerable answer."""
    return [
        JudgeLine(system=r.system, qid=r.qid, run=run, score=1 if (r.system, r.qid) in low else 3)
        for r in records
        if r.answerable
    ]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _low_answers(records: list[SystemRecord], n: int) -> set[tuple[str, str]]:
    """Both systems of `n` single-hop questions scored low (so they are not 'differ' rows)."""
    return {(s, f"single_hop:{i:02d}") for i in range(n) for s in ("standard", "agentic")}


def test_human_sheet_tops_up_non_full_answers(tmp_path: Path) -> None:
    records = _records()
    lines = _lines(records, _low_answers(records, 12))  # 24 low answers, none 'differ'
    base, base_key = export_human_sheet(records, n=30, seed=0, out_dir=tmp_path / "a", lines=lines)
    base_qids = {k["qid"] for k in _rows(base_key)}
    in_sheet = sum(1 for q in base_qids if q in {f"single_hop:{i:02d}" for i in range(12)})
    assert in_sheet < 5  # a random draw of 12 of 40 single-hop questions holds only a few
    sheet, key = export_human_sheet(
        records, n=30, seed=0, out_dir=tmp_path / "b", lines=lines, min_non_full=10
    )
    keys = _rows(key)
    non_full = sum(
        2 for k in keys if k["qid"].startswith("single_hop:") and int(k["qid"][-2:]) < 12
    )
    assert non_full >= 10 and len(keys) > 30
    assert len(_rows(sheet)) == len(keys)
    assert {k["origin"] for k in keys} <= {"differ", "random", "top_up"}
    # Rows already in the plain sheet stay; the added rows are marked top_up and each holds a
    # low answer.
    assert base_qids <= {k["qid"] for k in keys}
    added = [k for k in keys if k["origin"] == "top_up"]
    assert added and all(
        k["qid"].startswith("single_hop:") and int(k["qid"][-2:]) < 12 for k in added
    )
    # Still blind, and deterministic for a seed.
    text = sheet.read_text(encoding="utf-8-sig")
    assert "standard" not in text and "agentic" not in text and "top_up" not in text
    again, _ = export_human_sheet(
        records, n=30, seed=0, out_dir=tmp_path / "c", lines=lines, min_non_full=10
    )
    assert again.read_text(encoding="utf-8-sig") == text
    # Nothing to add when the sheet already holds enough, or when min_non_full is 0.
    plain, plain_key = export_human_sheet(
        records, n=30, seed=0, out_dir=tmp_path / "d", lines=lines, min_non_full=0
    )
    # Plain export: 30 rows, the 3 types in the 12/10/8 split, no top_up row, same sheet as default.
    plain_keys = _rows(plain_key)
    assert len(plain_keys) == 30 and "top_up" not in {k["origin"] for k in plain_keys}
    assert plain.read_text(encoding="utf-8-sig") == base.read_text(encoding="utf-8-sig")
    # All the judge's scores are 3: the number cannot be reached and the sheet stays at 30.
    _s, k3 = export_human_sheet(
        records, n=30, seed=0, out_dir=tmp_path / "e", lines=_lines(records, set()), min_non_full=10
    )
    assert len(_rows(k3)) == 30


def _fill(sheet: Path, key: Path, human: dict[tuple[str, str], int]) -> None:
    """Write the human scores (by system and qid) into the sheet."""
    keys = {k["sample_id"]: k for k in _rows(key)}
    rows = _rows(sheet)
    for row in rows:
        k = keys[row["sample_id"]]
        for slot in (1, 2):
            system = k[f"answer_{slot}_system"]
            row[f"score_1_3_answer_{slot}"] = str(human.get((system, k["qid"]), 3))
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_agentic_agreement_from_filled_sheet(tmp_path: Path) -> None:
    records = _records()
    low = _low_answers(records, 12)
    lines = _lines(records, low)
    sheet, key = export_human_sheet(
        records, n=30, seed=0, out_dir=tmp_path, lines=lines, min_non_full=10
    )
    keys = _rows(key)
    low_here = [
        (s, k["qid"]) for k in keys for s in ("standard", "agentic") if (s, k["qid"]) in low
    ]
    assert len(low_here) >= 10
    human = {pair: 1 for pair in low_here}
    # The human disagrees on two of the judge's low answers (gives 3) and on one full answer (2).
    human[low_here[0]] = 3
    human[low_here[1]] = 3
    other = next(
        (s, k["qid"]) for k in keys for s in ("standard", "agentic") if (s, k["qid"]) not in low
    )
    human[other] = 2
    _fill(sheet, key, human)
    # Leave one row blank: skipped.
    rows = _rows(sheet)
    rows[0]["score_1_3_answer_1"] = rows[0]["score_1_3_answer_2"] = ""
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    first = {(k["answer_1_system"], k["qid"]) for k in keys[:1]} | {
        (k["answer_2_system"], k["qid"]) for k in keys[:1]
    }
    result = agentic_agreement(sheet, key, lines)
    assert result.blank_answers == 2
    report = result.binary
    assert report.n == 2 * len(keys) - 2
    # Hand count from the same inputs.
    judge = {(r.system, r.qid): (1 if (r.system, r.qid) in low else 3) for r in records}
    pairs = [
        (human.get((s, k["qid"]), 3), judge[s, k["qid"]])
        for k in keys
        for s in (k["answer_1_system"], k["answer_2_system"])
        if (s, k["qid"]) not in first
    ]
    assert report.human_full == sum(h == 3 for h, _ in pairs)
    assert report.tnr.count == sum(1 for h, j in pairs if h != 3 and j != 3)
    assert report.tpr.count == sum(1 for h, j in pairs if h == 3 and j == 3)
    assert len(report.disagreements) == sum(h != j for h, j in pairs)
    assert set(result.by_origin) <= {"differ", "random", "top_up"}
    assert sum(r.n for r in result.by_origin.values()) == report.n
    assert {d.label.split("/")[0] for d in report.disagreements} <= {k["sample_id"] for k in keys}
    # A judge line with an error (no score) is left out and counted.
    broken = [*lines]
    victim = next(i for i, x in enumerate(broken) if (x.system, x.qid) == human_pair(keys[1]))
    broken[victim] = broken[victim].model_copy(update={"score": None, "error": "boom"})
    assert agentic_agreement(sheet, key, broken).unjudged_answers == 1
    # A score outside 1..3 is refused.
    rows[1]["score_1_3_answer_1"] = "4"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        agentic_agreement(sheet, key, lines)


def human_pair(key_row: dict[str, str]) -> tuple[str, str]:
    return key_row["answer_1_system"], key_row["qid"]


def test_prescore_items_are_blind_and_fill_round_trips(tmp_path: Path) -> None:
    records = _records()
    sheet, _key = export_human_sheet(records, n=30, seed=0, out_dir=tmp_path)
    rows = _rows(sheet)
    items, mapping = prescore_items(sheet, seed=3)
    assert len(items) == 2 * len(rows)
    assert all(set(item) == {"id", "question", "reference", "answer"} for item in items)
    text = " ".join(" ".join(item.values()) for item in items)
    assert "standard" not in text and "agentic" not in text
    assert [i["id"] for i in items] != sorted(i["id"] for i in items)  # shuffled
    by_id = {i["id"]: i for i in items}
    for item_id, (sample_id, slot) in mapping.items():
        row = next(r for r in rows if r["sample_id"] == sample_id)
        assert by_id[item_id]["answer"] == row[f"answer_{slot}"]
        assert by_id[item_id]["question"] == row["question"]
    scores = {i["id"]: 1 + int(i["id"][1:]) % 3 for i in items}
    out = tmp_path / "prescored.csv"
    fill_prescores(sheet, mapping, scores, out, comments={items[0]["id"]: "wrong number"})
    filled = _rows(out)
    assert list(filled[0]) == [*rows[0], "confirmed"] and all(r["confirmed"] == "" for r in filled)
    for item_id, (sample_id, slot) in mapping.items():
        row = next(r for r in filled if r["sample_id"] == sample_id)
        assert row[f"score_1_3_answer_{slot}"] == str(scores[item_id])
    assert all(r["comment"].startswith("AI：") for r in filled)
    first_row = next(r for r in filled if r["sample_id"] == mapping[items[0]["id"]][0])
    assert "wrong number" in first_row["comment"]
    # A score missing for an item is an error: a pre-score must cover every answer.
    scores.pop(items[0]["id"])
    with pytest.raises(ValueError):
        fill_prescores(sheet, mapping, scores, tmp_path / "bad.csv")


def test_cli_human_agreement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = _records()
    low = _low_answers(records, 12)
    lines = _lines(records, low)
    out = tmp_path / "run"
    for system in ("standard", "agentic"):
        for r in records:
            if r.system == system:
                append_line(system_file(out, system), r)
    for line in lines:
        append_line(judge_file(out), line)
    assert cli.main(["human-sheet", "--out", str(out), "--min-non-full", "10"]) == 0
    elsewhere = tmp_path / "elsewhere"
    assert cli.main(["human-sheet", "--out", str(out), "--out-dir", str(elsewhere)]) == 0
    assert (elsewhere / "human_sheet.csv").is_file() and (elsewhere / "human_key.csv").is_file()
    sheet, key = out / "human" / "human_sheet.csv", out / "human" / "human_key.csv"
    printed = capsys.readouterr()
    assert "non-full answers in the sheet" in printed.out
    assert "warning" not in printed.err  # 24 low answers exist: the 10 wanted are reached
    keys = _rows(key)
    human = {
        (s, k["qid"]): 1 for k in keys for s in ("standard", "agentic") if (s, k["qid"]) in low
    }
    _fill(sheet, key, human)  # the human agrees with the judge everywhere
    md = tmp_path / "agreement.md"
    code = cli.main(
        ["human-agreement", "--out", str(out), "--sheet", str(sheet), "--source", "ai-prescore",
         "--md", str(md)]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert (
        "TPR" in printed and "TNR" in printed and "kappa" in printed and "gate" in printed.lower()
    )
    text = md.read_text(encoding="utf-8")
    assert "AI pre-score" in text and "TPR" in text and "1.000" in text
    # Missing key: refused with a usage error.
    assert cli.main(["human-agreement", "--out", str(out), "--sheet", str(tmp_path / "x.csv")]) == 2


def test_human_sheet_warns_when_the_wanted_number_cannot_be_reached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _records()
    out = tmp_path / "run"
    for r in records:
        append_line(system_file(out, r.system), r)
    for line in _lines(records, _low_answers(records, 2)):  # only 4 low answers exist
        append_line(judge_file(out), line)
    assert cli.main(["human-sheet", "--out", str(out), "--min-non-full", "10"]) == 0
    printed = capsys.readouterr()
    assert "non-full answers in the sheet (the judge's first pass scored 1 or 2): 4" in printed.out
    assert "warning" in printed.err


def test_agreement_tolerates_a_short_csv_row(tmp_path: Path) -> None:
    records = _records()
    lines = _lines(records, set())
    sheet, key = export_human_sheet(records, n=30, seed=0, out_dir=tmp_path, lines=lines)
    text = sheet.read_text(encoding="utf-8-sig").splitlines()
    header = text[0] + ",confirmed"
    # Every data row stays one cell short of the header: no confirmed cell at all.
    sheet.write_text("\n".join([header, *text[1:]]) + "\n", encoding="utf-8-sig")
    assert agentic_agreement(sheet, key, lines).confirmed_rows == 0
