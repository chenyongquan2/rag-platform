"""Blind sheet for the owner's check of the support judge (spec P1-06, design s14-3).

One row per (sentence, cited fragment) pair: the question, the sentence and the fragment's text,
and an empty cell for the owner's label. Nothing names the system or shows the judge's verdict;
the key file (sample id -> source, system, question id, sentence, chunk id, the judge's labels)
is kept apart. Rows: every pair the judge did not call supported (at most half of the sheet,
drawn at random when there are more), the rest drawn at random from the supported pairs, seeded,
then shuffled. This module never writes a label.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path

from rag.answer.context import TITLE_SEPARATOR
from rag.eval.support.records import SupportLine

SHEET_COLUMNS = [
    "sample_id",
    "question",
    "sentence",
    "fragment_title",
    "fragment_text",
    "human_label",
    "comment",
]
KEY_COLUMNS = [
    "sample_id",
    "source",
    "system",
    "qid",
    "sentence_index",
    "chunk_id",
    "judge_label_run1",
    "judge_label_run2",
]
SHEET_NAME = "support_sheet.csv"
KEY_NAME = "support_key.csv"
RUBRIC_NAME = "support_rubric.txt"

_RUBRIC = """How to label (the same three labels the judge uses)

For each row, read the sentence and the fragment. Decide from this fragment alone, not from what
you know and not from other fragments, and write one of:
  supported      the fragment states everything the sentence claims (a paraphrase is fine)
  partly         the fragment supports some claims of the sentence, not all, or only with a guess
  not_supported  the fragment does not state the claim, or contradicts it (a different number,
                 name, version or condition counts as not_supported)

Leave human_label empty to skip a row. Do not open support_key.csv before you are done.

How the rows were chosen: pairs the judge did not call supported are over-represented on purpose
(up to half of the sheet), the rest are drawn at random. The sheet does not say which rows are
which.
"""


def export_support_sheet(
    lines: Sequence[SupportLine], *, n: int = 40, seed: int = 0, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write the sheet, the key and the rubric from the first pass; returns (sheet, key)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    first = [x for x in lines if x.run == 1 and x.result is not None and x.item.planted is None]
    second = {(x.source, x.system, x.qid): x for x in lines if x.run == 2 and x.result is not None}
    pairs = []
    for line in first:
        assert line.result is not None
        chunks = {c.chunk_id: c for c in line.item.chunks}
        for s in line.result.sentences:
            for p in s.pairs:
                if p.label != "unknown" and p.chunk_id in chunks:
                    pairs.append((line, s, p, chunks[p.chunk_id]))
    rng = random.Random(seed)
    other = [x for x in pairs if x[2].label != "supported"]
    supported = [x for x in pairs if x[2].label == "supported"]
    chosen = rng.sample(other, min(len(other), n // 2))
    chosen += rng.sample(supported, min(len(supported), n - len(chosen)))
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
        for i, (line, sentence, pair, chunk) in enumerate(chosen, start=1):
            sample_id = f"C{i:0{width}d}"
            ws.writerow(
                {
                    "sample_id": sample_id,
                    "question": line.item.question,
                    "sentence": sentence.text,
                    "fragment_title": TITLE_SEPARATOR.join(chunk.title_path),
                    "fragment_text": chunk.text,
                    "human_label": "",
                    "comment": "",
                }
            )
            run2 = second.get((line.source, line.system, line.qid))
            label2 = ""
            if run2 is not None and run2.result is not None:
                label2 = next(
                    (
                        q.label
                        for t in run2.result.sentences
                        for q in t.pairs
                        if q.sentence == pair.sentence and q.chunk_id == pair.chunk_id
                    ),
                    "",
                )
            wk.writerow(
                {
                    "sample_id": sample_id,
                    "source": line.source,
                    "system": line.system,
                    "qid": line.qid,
                    "sentence_index": pair.sentence,
                    "chunk_id": pair.chunk_id,
                    "judge_label_run1": pair.label,
                    "judge_label_run2": label2,
                }
            )
    (out / RUBRIC_NAME).write_text(_RUBRIC, encoding="utf-8", newline="\n")
    return sheet, key
