# ruff: noqa: RUF001 (the comments in the sheet start with a full-width colon)
"""Spec EVAL-1 AC8: blind re-score sheet of the P1-04 answer-quality sheet."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rag.eval.answer.agreement import KEY_COLUMNS, SHEET_COLUMNS
from rag.eval.answer.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeConfig,
    JudgeRun,
    JudgeScore,
    summarize_scores,
)
from rag.eval.answer.rejudge import export_rejudge_sheet, rejudge_agreement


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _fixture(tmp_path: Path) -> tuple[Path, Path, JudgeRun]:
    """A filled 50-row sheet (with AI comments and a confirmed column) and a judge run."""
    sheet, key = tmp_path / "filled.csv", tmp_path / "key.csv"
    rows, keys = [], []
    for i in range(50):
        sid = f"S{i + 1:02d}"
        rows.append(
            {
                "sample_id": sid,
                "question": f"question {i}?",
                "reference_answer": f"reference {i}",
                "answer": f"answer {i}",
                "score_1_5": "5",
                "score_1_3": "3",
                "comment": "AI：ok" if i % 2 else "",
                "confirmed": "yes",
            }
        )
        keys.append({"sample_id": sid, "qid": f"q{i:03d}", "answer_run_id": "answers-1"})
    rows[7]["score_1_5"] = "4"  # the human gave 4 where the judge gave 5 (S08)
    rows[20]["score_1_3"] = "2"  # and 2 where the judge gave 3 (S21)
    _write(sheet, rows)
    _write(key, keys)
    scores = [JudgeScore(qid=f"q{i:03d}", scale="1-5", score=5) for i in range(50)]
    scores += [JudgeScore(qid=f"q{i:03d}", scale="1-3", score=3) for i in range(50)]
    judge = JudgeRun(
        run_id="j",
        created_at="t",
        answer_run_id="answers-1",
        kb="hf_docs",
        judge_model="m",
        prompt_version=JUDGE_PROMPT_VERSION,
        config=JudgeConfig(model="m"),
        scales=["1-5", "1-3"],
        scores=scores,
        summary=summarize_scores(scores, ["1-5", "1-3"]),
    )
    return sheet, key, judge


def test_rejudge_export_is_blind_and_keeps_the_disagreements(tmp_path: Path) -> None:
    sheet, key, judge = _fixture(tmp_path)
    out_sheet, out_key = export_rejudge_sheet(
        sheet, key, judge, n=15, seed=0, out_dir=tmp_path / "o"
    )
    rows, keys = _read(out_sheet), _read(out_key)
    assert len(rows) == len(keys) == 15
    assert list(rows[0]) == SHEET_COLUMNS and list(keys[0]) == [*KEY_COLUMNS, "earlier_sample_id"]
    # No earlier score, comment or confirmation leaks into the blind sheet.
    assert all(r["score_1_5"] == "" and r["score_1_3"] == "" and r["comment"] == "" for r in rows)
    assert "AI" not in out_sheet.read_text(encoding="utf-8-sig") and "confirmed" not in " ".join(
        rows[0]
    )
    earlier = {k["earlier_sample_id"] for k in keys}
    assert {"S08", "S21"} <= earlier  # the rows where the human and the judge differed
    assert len(earlier) == 15
    # New ids differ from the old ones and the order is not the old order.
    assert [k["sample_id"] for k in keys] == [f"R{i:02d}" for i in range(1, 16)]
    by_new = {r["sample_id"]: r for r in rows}
    for k in keys:
        assert by_new[k["sample_id"]]["answer"] == f"answer {int(k['qid'][1:])}"
    again, _ = export_rejudge_sheet(sheet, key, judge, n=15, seed=0, out_dir=tmp_path / "again")
    assert _read(again) == rows
    with pytest.raises(ValueError):
        export_rejudge_sheet(sheet, key, judge, n=0, seed=0, out_dir=tmp_path / "zero")


def test_rejudge_agreement_counts_changed_scores(tmp_path: Path) -> None:
    sheet, key, judge = _fixture(tmp_path)
    out_sheet, out_key = export_rejudge_sheet(
        sheet, key, judge, n=15, seed=0, out_dir=tmp_path / "o"
    )
    rows, keys = _read(out_sheet), _read(out_key)
    earlier = {k["sample_id"]: k["earlier_sample_id"] for k in keys}
    # Re-score: the same as before everywhere except S08 (1-5: 5 now, 4 before) and one more row
    # (1-5: 3 now, 5 before).
    other = next(r for r in rows if earlier[r["sample_id"]] not in ("S08", "S21"))
    for r in rows:
        old = earlier[r["sample_id"]]
        r["score_1_5"] = "5"
        r["score_1_3"] = "2" if old == "S21" else "3"
    other["score_1_5"] = "3"
    _write(out_sheet, rows)
    result = rejudge_agreement(out_sheet, out_key, sheet)
    assert result.rows == 15 and result.blank == 0
    assert result.scales["1-5"].same == 13 and result.scales["1-3"].same == 15
    changed = {c.earlier_sample_id: (c.earlier, c.now) for c in result.scales["1-5"].changed}
    assert changed == {"S08": (4, 5), earlier[other["sample_id"]]: (5, 3)}
    assert result.scales["1-3"].changed == []
    # Blank rows are skipped.
    rows[0]["score_1_5"] = rows[0]["score_1_3"] = ""
    _write(out_sheet, rows)
    assert rejudge_agreement(out_sheet, out_key, sheet).blank == 1


def test_cli_rejudge_and_binary_agreement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag.eval.answer import cli

    sheet, key, judge = _fixture(tmp_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "results.json").write_text(judge.model_dump_json(), encoding="utf-8")
    out = tmp_path / "o"
    argv = ["rejudge-sheet", "--sheet", str(sheet), "--key", str(key), "--judge", str(judge_dir)]
    assert cli.main([*argv, "--out-dir", str(out)]) == 0
    assert len(_read(out / "rejudge_sheet.csv")) == 15
    rows = _read(out / "rejudge_sheet.csv")
    for r in rows:
        r["score_1_5"], r["score_1_3"] = "5", "3"
    _write(out / "rejudge_sheet.csv", rows)
    capsys.readouterr()
    code = cli.main(
        ["rejudge-agreement", "--sheet", str(out / "rejudge_sheet.csv"),
         "--key", str(out / "rejudge_key.csv"), "--earlier", str(sheet)]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "1-5" in printed and "same" in printed
    # The P1-04 command prints the full / not full rates and the gate too.
    assert cli.main(["human-agreement", "--sheet", str(sheet), "--key", str(key),
                     "--judge", str(judge_dir)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "TPR" in printed and "TNR" in printed and "gate" in printed.lower()
