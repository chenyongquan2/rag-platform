"""Judge against human scores (spec P1-04, design s14-3).

    sheet, key = export_human_sheet(answer_run, n=50, seed=0, out_dir=...)
    # the owner fills score_1_5 and score_1_3 in the sheet, blind
    report = human_agreement(sheet, key, judge_run)

The sheet shows only the question, the reference answer and the answer, in a seeded random
order under neutral sample ids: no system, model, qid or judge score, so the person scoring
does not know what produced the answer or what the judge said. The key file (sample id -> qid
and answer run) is kept apart and only read by `human_agreement`. Agreement is reported as
quadratic weighted kappa and Spearman, plus exact and within-one agreement with Wilson
intervals (50 items give about +-11 points on a share).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from pydantic import BaseModel

from rag.eval.answer.judge import SCALES, JudgeRun
from rag.eval.answer.record import AnswerRunRecord, Proportion
from rag.eval.stats import spearman, weighted_kappa

SHEET_COLUMNS = [
    "sample_id",
    "question",
    "reference_answer",
    "answer",
    "score_1_5",
    "score_1_3",
    "comment",
]
KEY_COLUMNS = ["sample_id", "qid", "answer_run_id"]
SHEET_NAME = "human_sheet.csv"
KEY_NAME = "human_key.csv"
RUBRIC_NAME = "human_rubric.txt"
_SCALE_COLUMN = {"1-5": "score_1_5", "1-3": "score_1_3"}
NO_ANSWER = "(no answer: the system returned none)"

_RUBRIC = """How to score (the same rubric the judge gets; the reference answer is the top score)

score_1_5: Is the answer correct, accurate and factual compared with the reference answer?
  1 completely incorrect, inaccurate and/or not factual
  2 mostly incorrect, inaccurate and/or not factual
  3 somewhat correct, accurate and/or factual
  4 mostly correct, accurate and factual
  5 completely correct, accurate and factual
score_1_3:
  1 completely incorrect, inaccurate and/or not factual
  2 somewhat correct, accurate and/or factual
  3 completely correct, accurate and/or factual

Score every row on both scales, or leave both cells of a row empty to skip it.
Do not open human_key.csv before you are done.
"""


def export_human_sheet(
    run: AnswerRunRecord, *, n: int = 50, seed: int = 0, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write the blind sheet, the key and the rubric; returns (sheet, key)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    records = list(run.questions)
    chosen = random.Random(seed).sample(records, min(n, len(records)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = out / SHEET_NAME, out / KEY_NAME
    width = len(str(len(chosen)))
    with (
        sheet.open("w", encoding="utf-8-sig", newline="") as fs,
        key.open("w", encoding="utf-8-sig", newline="") as fk,
    ):
        ws = csv.DictWriter(fs, fieldnames=SHEET_COLUMNS)
        wk = csv.DictWriter(fk, fieldnames=KEY_COLUMNS)
        ws.writeheader()
        wk.writeheader()
        for i, record in enumerate(chosen, start=1):
            sample_id = f"S{i:0{width}d}"
            ws.writerow(
                {
                    "sample_id": sample_id,
                    "question": record.question,
                    "reference_answer": record.reference,
                    "answer": record.text.strip() or NO_ANSWER,
                    "score_1_5": "",
                    "score_1_3": "",
                    "comment": "",
                }
            )
            wk.writerow({"sample_id": sample_id, "qid": record.qid, "answer_run_id": run.run_id})
    (out / RUBRIC_NAME).write_text(_RUBRIC, encoding="utf-8", newline="\n")
    return sheet, key


class ScaleAgreement(BaseModel):
    scale: str
    n: int
    weighted_kappa: float | None
    """Quadratic weights."""
    spearman: float | None
    exact: Proportion
    within_one: Proportion


class AgreementReport(BaseModel):
    answer_run_id: str
    judge_run_id: str
    judge_model: str
    prompt_version: str
    rows: int
    blank: int
    scales: dict[str, ScaleAgreement]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _score(text: str, scale: str, sample_id: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    low, high = SCALES[scale]
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{sample_id}: {scale} score {text!r} is not an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{sample_id}: {scale} score {value} outside {low}..{high}")
    return value


def human_agreement(sheet: str | Path, key: str | Path, judge: JudgeRun) -> AgreementReport:
    rows = _read(Path(sheet))
    keys = {k["sample_id"]: k for k in _read(Path(key))}
    runs = {k["answer_run_id"] for k in keys.values()}
    if runs != {judge.answer_run_id}:
        raise ValueError(
            f"the key is for answer run(s) {sorted(runs)}, the judge run scored "
            f"{judge.answer_run_id!r}"
        )
    pairs: dict[str, list[tuple[int, int]]] = {s: [] for s in judge.scales}
    blank = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in keys:
            raise ValueError(f"{sample_id}: not in the key file")
        qid = keys[sample_id]["qid"]
        filled = False
        for scale in judge.scales:
            human = _score(row.get(_SCALE_COLUMN[scale], ""), scale, sample_id)
            if human is None:
                continue
            filled = True
            machine = next(
                (s.score for s in judge.scores if s.qid == qid and s.scale == scale), None
            )
            if machine is not None:
                pairs[scale].append((machine, human))
        blank += not filled
    scales: dict[str, ScaleAgreement] = {}
    for scale, values in pairs.items():
        low, high = SCALES[scale]
        a = [m for m, _ in values]
        b = [h for _, h in values]
        scales[scale] = ScaleAgreement(
            scale=scale,
            n=len(values),
            weighted_kappa=weighted_kappa(a, b, low=low, high=high) if values else None,
            spearman=spearman(a, b),
            exact=Proportion.of(sum(x == y for x, y in values), len(values)),
            within_one=Proportion.of(sum(abs(x - y) <= 1 for x, y in values), len(values)),
        )
    return AgreementReport(
        answer_run_id=judge.answer_run_id,
        judge_run_id=judge.run_id,
        judge_model=judge.judge_model,
        prompt_version=judge.prompt_version,
        rows=len(rows),
        blank=blank,
        scales=scales,
    )
