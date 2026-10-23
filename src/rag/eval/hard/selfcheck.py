"""Blind sheet for the owner to answer ten hard questions (spec EVAL-2, AC10).

The sheet has the question and empty answer cells; the reference answer, the gold pages and the
evidence are in a key file kept apart. The owner answers first, then opens the key and marks each
row yes / partly / no. A row marked no, because the reference looks wrong or ambiguous, is
dropped or rewritten before any system runs. The draw is seeded, so the same command gives the
same ten questions.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path

from rag.eval.questions import LabeledQuestion

SHEET_COLUMNS = ["sample_id", "question", "your_answer", "your_pages", "comment"]
KEY_COLUMNS = [
    "sample_id",
    "qid",
    "shape",
    "reference_answer",
    "gold_pages",
    "evidence",
    "owner_matches_reference",
]
SHEET_NAME = "self_check_sheet.csv"
KEY_NAME = "self_check_key.csv"
RUBRIC_NAME = "self_check_rubric.txt"

_RUBRIC = """How to use this sheet (about 30 minutes for 10 questions)

1. Answer each question in your own words, with the Hugging Face documentation open. Write the
   pages you used in `your_pages`. These questions need two or three pages each.
2. Do not open self_check_key.csv before you have answered all ten.
3. Then open the key and write in `owner_matches_reference`: yes (same facts), partly (one fact
   differs or is missing), no (different answer).
4. If you answer no because the reference looks wrong, or the question has two fair answers, say
   so in `comment` in the sheet; that question is dropped or rewritten before any system runs, and
   the drop is recorded in hard_manifest.json.

The agreement (yes / partly / no counts) is written into the dataset description.
"""


def export_self_check(
    questions: Sequence[LabeledQuestion], *, n: int = 10, seed: int = 0, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write the sheet and the key for `n` randomly drawn hard questions; returns (sheet, key)."""
    hard = sorted((q for q in questions if q.meta.get("type") == "hard"), key=lambda q: q.qid)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > len(hard):
        raise ValueError(f"there are {len(hard)} hard questions, so n can be at most {len(hard)}")
    chosen = random.Random(seed).sample(hard, n)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = out / SHEET_NAME, out / KEY_NAME
    width = len(str(n))
    with (
        sheet.open("w", encoding="utf-8-sig", newline="") as fs,
        key.open("w", encoding="utf-8-sig", newline="") as fk,
    ):
        ws = csv.DictWriter(fs, fieldnames=SHEET_COLUMNS)
        wk = csv.DictWriter(fk, fieldnames=KEY_COLUMNS)
        ws.writeheader()
        wk.writeheader()
        for i, q in enumerate(chosen, start=1):
            sample_id = f"H{i:0{width}d}"
            ws.writerow(
                {
                    "sample_id": sample_id,
                    "question": q.question,
                    "your_answer": "",
                    "your_pages": "",
                    "comment": "",
                }
            )
            wk.writerow(
                {
                    "sample_id": sample_id,
                    "qid": q.qid,
                    "shape": q.meta.get("shape", ""),
                    "reference_answer": q.answer,
                    "gold_pages": "; ".join(q.relevant_pages),
                    "evidence": " | ".join(f["evidence"] for f in q.meta.get("facts", [])),
                    "owner_matches_reference": "",
                }
            )
    (out / RUBRIC_NAME).write_text(_RUBRIC, encoding="utf-8", newline="\n")
    return sheet, key
