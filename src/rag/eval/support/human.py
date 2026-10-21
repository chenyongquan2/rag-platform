"""Blind sheet for the owner's check of the support judge (spec P1-06, design s14-3).

One row per (sentence, cited fragment) pair: the question, the sentence and the fragment's text,
and empty cells for the owner's label and comment. Nothing names the system, the stratum or the
judge's verdict; the key file (sample id -> stratum, source, system, question id, sentence,
chunk id, the judge's labels, the known answer) is kept apart, so agreement can be computed per
stratum. The rows, in four strata:

- `judge_not_supported`: every real pair the judge did not call supported in run 1 or run 2;
- `planted`: the planted pair of every planted copy (known answer: not supported);
- `reviewer_hard`: the pairs of the changed sentence of every hand-made near miss passed as
  `extra` (known answer: not supported);
- `random_supported`: the rest of the `n` rows, drawn at random (seeded) from the real pairs both
  runs called supported.

The first three strata are always complete, even beyond `n`. The rows are shuffled. This module
never writes a label.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path

from rag.answer.context import TITLE_SEPARATOR
from rag.answer.support import PairVerdict, SentenceVerdict
from rag.eval.support.records import SupportLine
from rag.types import Chunk

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
    "stratum",
    "source",
    "system",
    "qid",
    "sentence_index",
    "chunk_id",
    "judge_label_run1",
    "judge_label_run2",
    "known_answer",
]
SHEET_NAME = "support_sheet.csv"
KEY_NAME = "support_key.csv"
RUBRIC_NAME = "support_rubric.txt"
KNOWN_NOT_SUPPORTED = "not_supported_or_partly"
"""Known answer of a planted or hand-made pair: the fragment does not fully support it."""

_RUBRIC = """How to label (the same three labels the judge uses)

For each row, read the sentence and the fragment. Decide from this fragment alone, not from what
you know and not from other fragments, and write one of:
  supported      the fragment states everything the sentence claims (a paraphrase is fine)
  partly         the fragment supports some claims of the sentence, not all, or only with a guess
  not_supported  the fragment does not state the claim, or contradicts it (a different number,
                 name, version or condition counts as not_supported)

Leave human_label empty to skip a row. Do not open support_key.csv before you are done.

How the rows were chosen: pairs the judge doubted, sentences changed on purpose and pairs drawn
at random are mixed. The sheet does not say which rows are which.
"""

_Row = tuple[str, SupportLine, SentenceVerdict, PairVerdict, Chunk]


def _pairs(line: SupportLine) -> list[tuple[SentenceVerdict, PairVerdict, Chunk]]:
    if line.result is None:
        return []
    chunks = {c.chunk_id: c for c in line.item.chunks}
    return [
        (s, p, chunks[p.chunk_id])
        for s in line.result.sentences
        for p in s.pairs
        if p.chunk_id in chunks
    ]


def _label(line: SupportLine | None, sentence: int, chunk_id: str) -> str:
    if line is None or line.result is None:
        return ""
    for s in line.result.sentences:
        for p in s.pairs:
            if p.sentence == sentence and p.chunk_id == chunk_id:
                return p.label
    return ""


def export_support_sheet(
    lines: Sequence[SupportLine],
    *,
    n: int = 40,
    seed: int = 0,
    out_dir: str | Path,
    extra: Sequence[SupportLine] = (),
) -> tuple[Path, Path]:
    """Write the sheet, the key and the rubric; returns (sheet, key). `lines` are the eval's
    lines (both runs, planted copies included); `extra` are hand-made near misses."""
    if n < 1:
        raise ValueError("n must be at least 1")
    run2 = {(x.source, x.system, x.qid): x for x in lines if x.run == 2 and x.result is not None}
    must: list[_Row] = []
    supported: list[_Row] = []
    for line in lines:
        if line.run != 1:
            continue
        planted = line.item.planted
        for s, p, chunk in _pairs(line):
            if planted is not None:
                if s.index == planted.sentence and p.chunk_id == planted.chunk_id:
                    must.append(("planted", line, s, p, chunk))
                continue
            if p.label == "unknown":
                continue
            other = _label(run2.get((line.source, line.system, line.qid)), p.sentence, p.chunk_id)
            if p.label != "supported" or other in ("partly", "not_supported"):
                must.append(("judge_not_supported", line, s, p, chunk))
            else:
                supported.append(("random_supported", line, s, p, chunk))
    for line in extra:
        planted = line.item.planted
        for s, p, chunk in _pairs(line):
            if planted is None or s.index == planted.sentence:
                must.append(("reviewer_hard", line, s, p, chunk))

    rng = random.Random(seed)
    chosen = must + rng.sample(supported, max(0, min(len(supported), n - len(must))))
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
        for i, (stratum, line, sentence, pair, chunk) in enumerate(chosen, start=1):
            sample_id = f"C{i:0{width}d}"
            text = line.item.sentences[sentence.index - 1].text
            ws.writerow(
                {
                    "sample_id": sample_id,
                    "question": line.item.question,
                    "sentence": text,
                    "fragment_title": TITLE_SEPARATOR.join(chunk.title_path),
                    "fragment_text": chunk.text,
                    "human_label": "",
                    "comment": "",
                }
            )
            wk.writerow(
                {
                    "sample_id": sample_id,
                    "stratum": stratum,
                    "source": line.source,
                    "system": line.system,
                    "qid": line.qid,
                    "sentence_index": pair.sentence,
                    "chunk_id": pair.chunk_id,
                    "judge_label_run1": pair.label,
                    "judge_label_run2": _label(
                        run2.get((line.source, line.system, line.qid)), pair.sentence, pair.chunk_id
                    ),
                    "known_answer": KNOWN_NOT_SUPPORTED
                    if stratum in ("planted", "reviewer_hard")
                    else "",
                }
            )
    (out / RUBRIC_NAME).write_text(_RUBRIC, encoding="utf-8", newline="\n")
    return sheet, key
