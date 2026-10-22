# ruff: noqa: RUF001, RUF002 (the pre-score comments use full-width Chinese punctuation)
"""Blind sheet for the owner's human scores of the two systems' answers (spec P3-05, design s14-3),
and how the judge is compared with the scores in it (spec EVAL-1).

One row per question: the question, the reference answer and the two answers in a seeded random
order under neutral sample ids; nothing names the system or shows the judge's score. The key
file (sample id -> which system wrote answer 1 and answer 2, and why the row is in the sheet) is
kept apart. Same layout idea as P1-04's sheet (`rag.eval.answer.agreement`), with two answers per
row and the 1-3 scale only. The export never writes scores. `fill_prescores` copies the sheet with
a pre-scorer's scores in it (an AI's, never the owner's); the owner still confirms them.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from rag.eval.agentic.records import JudgeLine, SystemRecord
from rag.eval.calibration import (
    BinaryAgreement,
    CalibrationItem,
    GateCheck,
    Source,
    calibrate,
    check_gate,
    render_calibration,
)

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
KEY_COLUMNS = ["sample_id", "qid", "answer_1_system", "answer_2_system", "origin"]
"""`origin`: `differ` (the judge scored the two answers differently), `random`, or `top_up`
(added because the judge scored an answer 1 or 2)."""
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

How the rows were chosen: every question where the judge scored the two answers differently is
in the sheet; the other rows are drawn at random (seeded) per question type. The sheet does not
say which rows are which.
"""

_RUBRIC_TOP_UP = """A few more rows were added because the judge gave one of their answers a
low score.
"""


def _quotas(n: int) -> dict[str, int]:
    total = sum(QUOTA_WEIGHTS.values())
    quotas = {t: n * w // total for t, w in QUOTA_WEIGHTS.items()}
    quotas["single_hop"] += n - sum(quotas.values())
    return quotas


def _differing(pairs: dict[str, dict[str, SystemRecord]], lines: Sequence[JudgeLine]) -> list[str]:
    """Questions whose first-pass judge scores differ between the two systems."""
    scores = {(x.system, x.qid): x.score for x in lines if x.run == 1 and x.score is not None}
    return sorted(
        q
        for q, p in pairs.items()
        if len(p) == 2
        and ("standard", q) in scores
        and ("agentic", q) in scores
        and scores["standard", q] != scores["agentic", q]
    )


def _low_scores(lines: Sequence[JudgeLine], run: int) -> set[tuple[str, str]]:
    """(system, qid) of the answers the judge scored 1 or 2 in pass `run`."""
    return {(x.system, x.qid) for x in lines if x.run == run and x.score in (1, 2)}


def _top_up(
    pairs: dict[str, dict[str, SystemRecord]],
    chosen: Sequence[str],
    lines: Sequence[JudgeLine],
    wanted: int,
    run: int,
    seed: int,
) -> list[str]:
    """Questions to add so the sheet holds `wanted` answers the judge scored 1 or 2: the
    questions with the most such answers first, ties in a seeded random order."""
    if wanted <= 0:
        return []
    low = _low_scores(lines, run)

    def count(qid: str) -> int:
        return sum((system, qid) in low for system in pairs[qid])

    have = sum(count(q) for q in chosen)
    rest = sorted(q for q, p in pairs.items() if len(p) == 2 and q not in chosen and count(q) > 0)
    random.Random(f"{seed}-top-up").shuffle(rest)
    rest.sort(key=lambda q: -count(q))
    added: list[str] = []
    for qid in rest:
        if have >= wanted:
            break
        added.append(qid)
        have += count(qid)
    return added


def export_human_sheet(
    records: Sequence[SystemRecord],
    *,
    n: int = 30,
    seed: int = 0,
    out_dir: str | Path,
    lines: Sequence[JudgeLine] = (),
    min_non_full: int = 0,
    judge_run: int = 1,
) -> tuple[Path, Path]:
    """Write the sheet, the key and the rubric; returns (sheet, key). With the judge's `lines`,
    every question where the two systems' scores differ is included (at most `n` of them), the
    rest of the `n` rows is drawn at random by type. With `min_non_full`, further rows (marked
    `top_up` in the key) are added until the sheet holds that many answers the judge scored 1 or
    2 in pass `judge_run`, or no such question is left."""
    if n < 1:
        raise ValueError("n must be at least 1")
    pairs: dict[str, dict[str, SystemRecord]] = {}
    for r in records:
        if r.answerable:
            pairs.setdefault(r.qid, {})[r.system] = r
    rng = random.Random(seed)
    must = _differing(pairs, lines)
    chosen: list[str] = rng.sample(must, n) if len(must) > n else list(must)
    origin = dict.fromkeys(chosen, "differ")
    for qtype, quota in _quotas(n - len(chosen)).items():
        qids = sorted(
            q
            for q, p in pairs.items()
            if len(p) == 2 and p["standard"].qtype == qtype and q not in chosen
        )
        picked = rng.sample(qids, min(quota, len(qids)))
        chosen.extend(picked)
        origin.update(dict.fromkeys(picked, "random"))
    added = _top_up(pairs, chosen, lines, min_non_full, judge_run, seed)
    chosen.extend(added)
    origin.update(dict.fromkeys(added, "top_up"))
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
                    "origin": origin[qid],
                }
            )
    rubric = _RUBRIC + (_RUBRIC_TOP_UP if added else "")
    (out / RUBRIC_NAME).write_text(rubric, encoding="utf-8", newline="\n")
    return sheet, key


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def count_non_full(key: str | Path, lines: Sequence[JudgeLine], *, run: int = 1) -> int:
    """Answers in the sheet of this key that the judge scored 1 or 2 in pass `run`."""
    low = _low_scores(lines, run)
    return sum(
        (row[f"answer_{slot}_system"], row["qid"]) in low
        for row in _read_csv(Path(key))
        for slot in (1, 2)
    )


def _score(text: str, label: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{label}: score {text!r} is not an integer") from None
    if value not in (1, 2, 3):
        raise ValueError(f"{label}: score {value} outside 1..3")
    return value


class SheetAgreement(BaseModel):
    """The judge against the scores in a filled two-answer sheet."""

    judge_run: int
    rows: int
    blank_answers: int
    """Answers left without a score."""
    unjudged_answers: int
    """Scored answers the judge has no score for (an error, or not judged)."""
    confirmed_rows: int | None
    """Rows whose `confirmed` cell says yes; None when the sheet has no such column."""
    binary: BinaryAgreement
    by_origin: dict[str, BinaryAgreement]
    """The same numbers per how the row was chosen (`differ`, `random`, `top_up`); the judge's
    own scores picked the first and the last, so only `random` is free of that selection."""


def agentic_agreement(
    sheet: str | Path, key: str | Path, lines: Sequence[JudgeLine], *, run: int = 1
) -> SheetAgreement:
    """Compare the filled sheet's scores with the judge's pass `run` (default: the first, the
    primary one). Blank answers are skipped; a score outside 1..3 raises `ValueError`."""
    rows = _read_csv(Path(sheet))
    keys = {k["sample_id"]: k for k in _read_csv(Path(key))}
    judge = {(x.system, x.qid): x.score for x in lines if x.run == run and x.score is not None}
    items: list[CalibrationItem] = []
    by_origin: dict[str, list[CalibrationItem]] = {}
    blank = unjudged = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in keys:
            raise ValueError(f"{sample_id}: not in the key file")
        k = keys[sample_id]
        for slot in (1, 2):
            system = k[f"answer_{slot}_system"]
            label = f"{sample_id}/answer_{slot} ({system})"
            human = _score(row.get(f"score_1_3_answer_{slot}", ""), label)
            if human is None:
                blank += 1
                continue
            machine = judge.get((system, k["qid"]))
            if machine is None:
                unjudged += 1
                continue
            item = CalibrationItem(label=label, human=human, judge=machine)
            items.append(item)
            by_origin.setdefault(k.get("origin", ""), []).append(item)
    confirmed = (
        sum(r["confirmed"].strip().lower() in ("yes", "y") for r in rows)
        if rows and "confirmed" in rows[0]
        else None
    )
    return SheetAgreement(
        judge_run=run,
        rows=len(rows),
        blank_answers=blank,
        unjudged_answers=unjudged,
        confirmed_rows=confirmed,
        binary=calibrate(items),
        by_origin={o: calibrate(v) for o, v in sorted(by_origin.items()) if o},
    )


def render_sheet_agreement(
    result: SheetAgreement, gate: GateCheck, *, source: Source = "human", title: str = ""
) -> str:
    """Markdown: the calibration table and disagreements, then the per-origin breakdown."""
    text = render_calibration(result.binary, gate, source=source, title=title)
    notes = [
        f"Judge pass {result.judge_run}; sheet rows {result.rows}; answers without a score "
        f"{result.blank_answers}; scored answers the judge has no score for "
        f"{result.unjudged_answers}."
    ]
    if result.confirmed_rows is not None:
        notes.append(f"Rows marked confirmed: {result.confirmed_rows} of {result.rows}.")
    if result.by_origin:
        notes += [
            "",
            "By how the row was chosen (only `random` rows were not picked by the judge's own "
            "score):",
            "",
        ]
        for name, r in result.by_origin.items():
            tpr = "n/a" if r.tpr.rate is None else r.tpr.text()
            tnr = "n/a" if r.tnr.rate is None else r.tnr.text()
            notes.append(f"- {name}: {r.n} answers, TPR {tpr}, TNR {tnr}")
    return text + "\n" + "\n".join(notes) + "\n"


def prescore_items(
    sheet: str | Path, *, seed: int = 0
) -> tuple[list[dict[str, str]], dict[str, tuple[str, int]]]:
    """One item per answer of the sheet for a pre-scorer: only the question, the reference
    answer and the one answer, under opaque ids, in a shuffled order. The second value maps an
    id back to (sample_id, answer slot 1 or 2)."""
    rows = _read_csv(Path(sheet))
    width = len(str(2 * len(rows)))
    items: list[dict[str, str]] = []
    mapping: dict[str, tuple[str, int]] = {}
    for row in rows:
        for slot in (1, 2):
            item_id = f"P{len(items) + 1:0{width}d}"
            items.append(
                {
                    "id": item_id,
                    "question": row["question"],
                    "reference": row["reference_answer"],
                    "answer": row[f"answer_{slot}"],
                }
            )
            mapping[item_id] = (row["sample_id"], slot)
    random.Random(seed).shuffle(items)
    return items, mapping


def fill_prescores(
    sheet: str | Path,
    mapping: Mapping[str, tuple[str, int]],
    scores: Mapping[str, int],
    out: str | Path,
    *,
    comments: Mapping[str, str] | None = None,
) -> Path:
    """Copy the sheet with a pre-scorer's scores in it, comments starting with `AI：`, and an
    empty `confirmed` column for the owner. Every answer needs a score of 1 to 3."""
    comments = comments or {}
    for item_id in mapping:
        if _score(str(scores.get(item_id, "")), item_id) is None:
            raise ValueError(f"{item_id}: no score")
    rows = _read_csv(Path(sheet))
    by_sample = {r["sample_id"]: r for r in rows}
    notes: dict[str, list[str]] = {}
    for item_id, (sample_id, slot) in sorted(mapping.items()):
        by_sample[sample_id][f"score_1_3_answer_{slot}"] = str(scores[item_id])
        if comments.get(item_id):
            notes.setdefault(sample_id, []).append(f"答案{slot}：{comments[item_id]}")
    columns = [*SHEET_COLUMNS, "confirmed"]
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row["comment"] = "AI：" + "；".join(notes.get(row["sample_id"], []))
            row["confirmed"] = ""
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


__all__ = [
    "KEY_COLUMNS",
    "SHEET_COLUMNS",
    "SheetAgreement",
    "agentic_agreement",
    "check_gate",
    "count_non_full",
    "export_human_sheet",
    "fill_prescores",
    "prescore_items",
    "render_sheet_agreement",
]
