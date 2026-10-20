"""Blind sheet for the owner's human scores of the two systems' answers (spec P3-05, design s14-3).

One row per question: the question, the reference answer and the two answers in a seeded random
order under neutral sample ids; nothing names the system or shows the judge's score. The key
file (sample id -> which system wrote answer 1 and answer 2) is kept apart. Same layout idea as
P1-04's sheet (`rag.eval.answer.agreement`), with two answers per row and the 1-3 scale only.
The scores are the owner's to fill: this module never writes any.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path

from rag.eval.agentic.records import SystemRecord

SHEET_COLUMNS = [
    "sample_id",
    "question",
    "reference_answer",
    "answer_1",
    "answer_2",
    "score_1_3_answer_1",
    "score_1_3_answer_2",
    "comment",
]
KEY_COLUMNS = ["sample_id", "qid", "answer_1_system", "answer_2_system"]
SHEET_NAME = "human_sheet.csv"
KEY_NAME = "human_key.csv"
RUBRIC_NAME = "human_rubric.txt"
NO_ANSWER = "(no answer: the system returned none)"
QUOTA_WEIGHTS = {"single_hop": 12, "two_hop": 10, "colloquial": 8}
"""Rows per type for 30 rows (other sizes keep the proportions)."""

_RUBRIC = """How to score (the same rubric the judge gets; the reference answer is the top score)

Score each of the two answers of a row on its own, from 1 to 3:
  1 completely incorrect, inaccurate and/or not factual
  2 somewhat correct, accurate and/or factual
  3 completely correct, accurate and/or factual

Leave both cells of a row empty to skip it. Do not open human_key.csv before you are done.
"""


def _quotas(n: int) -> dict[str, int]:
    total = sum(QUOTA_WEIGHTS.values())
    quotas = {t: n * w // total for t, w in QUOTA_WEIGHTS.items()}
    quotas["single_hop"] += n - sum(quotas.values())
    return quotas


def export_human_sheet(
    records: Sequence[SystemRecord], *, n: int = 30, seed: int = 0, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write the sheet, the key and the rubric; returns (sheet, key)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    pairs: dict[str, dict[str, SystemRecord]] = {}
    for r in records:
        if r.answerable:
            pairs.setdefault(r.qid, {})[r.system] = r
    rng = random.Random(seed)
    chosen: list[str] = []
    for qtype, quota in _quotas(n).items():
        qids = sorted(q for q, p in pairs.items() if len(p) == 2 and p["standard"].qtype == qtype)
        chosen.extend(rng.sample(qids, min(quota, len(qids))))
    rng.shuffle(chosen)
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
        for i, qid in enumerate(chosen, start=1):
            pair = pairs[qid]
            order = ["standard", "agentic"]
            if rng.random() < 0.5:
                order.reverse()
            sample_id = f"S{i:0{width}d}"
            ws.writerow(
                {
                    "sample_id": sample_id,
                    "question": pair[order[0]].question,
                    "reference_answer": pair[order[0]].reference,
                    "answer_1": pair[order[0]].text.strip() or NO_ANSWER,
                    "answer_2": pair[order[1]].text.strip() or NO_ANSWER,
                    "score_1_3_answer_1": "",
                    "score_1_3_answer_2": "",
                    "comment": "",
                }
            )
            wk.writerow(
                {
                    "sample_id": sample_id,
                    "qid": qid,
                    "answer_1_system": order[0],
                    "answer_2_system": order[1],
                }
            )
    (out / RUBRIC_NAME).write_text(_RUBRIC, encoding="utf-8", newline="\n")
    return sheet, key
