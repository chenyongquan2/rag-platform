"""Blind re-score of some rows of the P1-04 answer-quality sheet (spec EVAL-1).

The filled sheet was scored by an AI first and then checked by the owner, who saw the AI's score
while checking. To see how far that anchored the owner, `export_rejudge_sheet` writes a short
sheet (15 rows) with no earlier score, no comment and no confirmation, under new ids; the owner
scores it from the question, the reference answer and the answer alone. `rejudge_agreement` then
counts how many scores differ from the earlier ones. Rows where the earlier human score differed
from the judge are always in: they are where a changed mind matters most.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from pydantic import BaseModel

from rag.eval.answer.agreement import RUBRIC_TEXT, SHEET_COLUMNS
from rag.eval.answer.judge import SCALES, JudgeRun

SHEET_NAME = "rejudge_sheet.csv"
KEY_NAME = "rejudge_key.csv"
RUBRIC_NAME = "rejudge_rubric.txt"
KEY_COLUMNS = ["sample_id", "qid", "answer_run_id", "earlier_sample_id"]
_SCALE_COLUMN = {"1-5": "score_1_5", "1-3": "score_1_3"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _int(text: str | None) -> int | None:
    text = (text or "").strip()
    return int(text) if text else None


def export_rejudge_sheet(
    sheet: str | Path,
    key: str | Path,
    judge: JudgeRun,
    *,
    n: int = 15,
    seed: int = 0,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the blind re-score sheet and its key; returns (sheet, key). `sheet` is the filled
    sheet with its `key`; `judge` supplies the scores to find the rows where the earlier human
    score differs on either scale."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rows = _read(Path(sheet))
    keys = {k["sample_id"]: k for k in _read(Path(key))}
    machine = {(s.qid, s.scale): s.score for s in judge.scores}

    def differs(row: dict[str, str]) -> bool:
        qid = keys[row["sample_id"]]["qid"]
        for scale, column in _SCALE_COLUMN.items():
            human, mine = _int(row.get(column)), machine.get((qid, scale))
            if human is not None and mine is not None and human != mine:
                return True
        return False

    scored = [r for r in rows if any(_int(r.get(c)) is not None for c in _SCALE_COLUMN.values())]
    rng = random.Random(seed)
    must = [r for r in scored if differs(r)]
    chosen = rng.sample(must, n) if len(must) > n else list(must)
    rest = [r for r in scored if r not in chosen]
    chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    rng.shuffle(chosen)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_sheet, out_key = out / SHEET_NAME, out / KEY_NAME
    width = max(2, len(str(len(chosen))))
    with (
        out_sheet.open("w", encoding="utf-8-sig", newline="") as fs,
        out_key.open("w", encoding="utf-8-sig", newline="") as fk,
    ):
        ws = csv.DictWriter(fs, fieldnames=SHEET_COLUMNS)
        wk = csv.DictWriter(fk, fieldnames=KEY_COLUMNS)
        ws.writeheader()
        wk.writeheader()
        for i, row in enumerate(chosen, start=1):
            new_id = f"R{i:0{width}d}"
            ws.writerow(
                {
                    "sample_id": new_id,
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "answer": row["answer"],
                    "score_1_5": "",
                    "score_1_3": "",
                    "comment": "",
                }
            )
            k = keys[row["sample_id"]]
            wk.writerow(
                {
                    "sample_id": new_id,
                    "qid": k["qid"],
                    "answer_run_id": k["answer_run_id"],
                    "earlier_sample_id": row["sample_id"],
                }
            )
    (out / RUBRIC_NAME).write_text(RUBRIC_TEXT, encoding="utf-8", newline="\n")
    return out_sheet, out_key


class ScoreChange(BaseModel):
    earlier_sample_id: str
    earlier: int
    now: int


class ScaleRejudge(BaseModel):
    scale: str
    n: int
    same: int
    changed: list[ScoreChange]


class RejudgeReport(BaseModel):
    rows: int
    blank: int
    scales: dict[str, ScaleRejudge]


def rejudge_agreement(
    rejudge_sheet: str | Path, rejudge_key: str | Path, confirmed_sheet: str | Path
) -> RejudgeReport:
    """How many of the re-scored answers got the same score as in the confirmed sheet, and which
    ones changed. A row with both cells empty is skipped (`blank`)."""
    rows = _read(Path(rejudge_sheet))
    keys = {k["sample_id"]: k for k in _read(Path(rejudge_key))}
    earlier = {r["sample_id"]: r for r in _read(Path(confirmed_sheet))}
    blank = 0
    same = dict.fromkeys(_SCALE_COLUMN, 0)
    changes: dict[str, list[ScoreChange]] = {scale: [] for scale in _SCALE_COLUMN}
    compared = dict.fromkeys(_SCALE_COLUMN, 0)
    for row in rows:
        old_id = keys[row["sample_id"]]["earlier_sample_id"]
        filled = False
        for scale, column in _SCALE_COLUMN.items():
            now = _int(row.get(column))
            if now is None:
                continue
            low, high = SCALES[scale]
            if not low <= now <= high:
                raise ValueError(f"{row['sample_id']}: {scale} score {now} outside {low}..{high}")
            filled = True
            before = _int(earlier[old_id].get(column))
            if before is None:
                continue
            compared[scale] += 1
            if before == now:
                same[scale] += 1
            else:
                changes[scale].append(
                    ScoreChange(earlier_sample_id=old_id, earlier=before, now=now)
                )
        blank += not filled
    return RejudgeReport(
        rows=len(rows),
        blank=blank,
        scales={
            scale: ScaleRejudge(
                scale=scale, n=compared[scale], same=same[scale], changed=changes[scale]
            )
            for scale in _SCALE_COLUMN
        },
    )
