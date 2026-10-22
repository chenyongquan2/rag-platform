"""Spec P1-04: weighted kappa, Spearman, blind human sheet and agreement."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from rag.eval.answer import AnswerQuestionRecord, AnswerRunRecord, format_compliance
from rag.eval.answer.agreement import (
    KEY_COLUMNS,
    SHEET_COLUMNS,
    export_human_sheet,
    human_agreement,
    spearman,
    weighted_kappa,
)
from rag.eval.answer.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeConfig,
    JudgeRun,
    JudgeScore,
    summarize_scores,
)
from rag.types import Answer, AnswerSentence, AnswerStatus, Citation


def _qwk_reference(a: list[int], b: list[int]) -> float:
    """Quadratic weighted kappa by its moment identity: 2 cov / (var a + var b + (mean diff)^2)."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def test_weighted_kappa_known_values() -> None:
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], low=1, high=5) == pytest.approx(1.0)
    a, b = [1, 2, 3, 4, 5, 3, 2], [1, 3, 3, 5, 4, 2, 2]
    assert weighted_kappa(a, b, low=1, high=5) == pytest.approx(_qwk_reference(a, b))
    a, b = [1, 1, 2, 3, 3, 2], [1, 2, 2, 3, 1, 3]
    assert weighted_kappa(a, b, low=1, high=3) == pytest.approx(_qwk_reference(a, b))
    # Linear weights by hand: a=[1,2,3], b=[1,3,3] on 1..3. One pair (2,3) differs by one step
    # of two: observed disagreement 0.5/3; expected from the marginals a=(1,1,1)/3, b=(1,0,2)/3.
    pa, pb = np.array([1, 1, 1]) / 3, np.array([1, 0, 2]) / 3
    w = np.abs(np.subtract.outer(np.arange(3), np.arange(3))) / 2
    expected = float(pa @ w @ pb)
    observed = 0.5 / 3
    assert weighted_kappa([1, 2, 3], [1, 3, 3], low=1, high=3, weights="linear") == (
        pytest.approx(1 - observed / expected)
    )
    # Both raters constant: kappa undefined.
    assert weighted_kappa([2, 2], [2, 2], low=1, high=3) is None
    with pytest.raises(ValueError):
        weighted_kappa([1, 4], [1, 2], low=1, high=3)


def test_spearman_known_values() -> None:
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(0.8207826816681233)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def _records(n: int) -> list[AnswerQuestionRecord]:
    out = []
    for i in range(n):
        answer = Answer(
            request_id=str(i),
            kb="hf_docs",
            status=AnswerStatus.ANSWERED,
            found=True,
            sentences=[
                AnswerSentence(text=f"answer {i}", citations=[Citation(marker=1, chunk_id="c#1")])
            ],
        )
        out.append(
            AnswerQuestionRecord.from_answer(
                qid=f"q{i:03d}",
                question=f"question {i}?",
                reference=f"reference {i}",
                relevant_pages=[],
                answer=answer,
                error=None,
            )
        )
    return out


def _run(n: int) -> AnswerRunRecord:
    records = _records(n)
    return AnswerRunRecord(
        run_id="answers-1",
        created_at="t",
        kb="hf_docs",
        label="",
        kind="hf",
        chat_model="deepseek-v4-flash",
        questions=records,
        compliance=format_compliance(records),
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_human_sheet_is_blind(tmp_path: Path) -> None:
    run = _run(65)
    sheet, key = export_human_sheet(run, n=50, seed=0, out_dir=tmp_path)
    rows, keys = _read(sheet), _read(key)
    assert len(rows) == 50 and len(keys) == 50
    assert list(rows[0]) == SHEET_COLUMNS and list(keys[0]) == KEY_COLUMNS
    header = " ".join(SHEET_COLUMNS).lower()
    for word in ("judge", "system", "model", "qid", "deepseek"):
        assert word not in header
    assert all(r["score_1_5"] == "" and r["score_1_3"] == "" for r in rows)
    assert "deepseek" not in sheet.read_text(encoding="utf-8-sig")
    qids = [k["qid"] for k in keys]
    assert len(set(qids)) == 50 and qids != sorted(qids)  # shuffled
    assert {k["answer_run_id"] for k in keys} == {"answers-1"}
    again, _ = export_human_sheet(run, n=50, seed=0, out_dir=tmp_path / "again")
    assert _read(again) == rows  # seeded
    other, _ = export_human_sheet(run, n=50, seed=1, out_dir=tmp_path / "other")
    assert _read(other) != rows
    by_id = {r["sample_id"]: r for r in rows}
    k0 = keys[0]
    i = int(k0["qid"][1:])
    assert by_id[k0["sample_id"]]["answer"] == f"answer {i}"
    assert by_id[k0["sample_id"]]["reference_answer"] == f"reference {i}"


def test_human_agreement_from_filled_sheet(tmp_path: Path) -> None:
    run = _run(10)
    sheet, key = export_human_sheet(run, n=6, seed=0, out_dir=tmp_path)
    rows, keys = _read(sheet), _read(key)
    qid_of = {k["sample_id"]: k["qid"] for k in keys}
    judge5 = {f"q{i:03d}": 1 + (i % 5) for i in range(10)}
    judge3 = {f"q{i:03d}": 1 + (i % 3) for i in range(10)}
    human5, human3 = [], []
    for j, row in enumerate(rows):
        if j == 5:  # left blank: skipped
            continue
        qid = qid_of[row["sample_id"]]
        h5 = judge5[qid] if j % 2 == 0 else max(1, judge5[qid] - 1)
        row["score_1_5"], row["score_1_3"] = str(h5), str(judge3[qid])
        human5.append((judge5[qid], h5))
        human3.append((judge3[qid], judge3[qid]))
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    scores = [JudgeScore(qid=q, scale="1-5", score=s) for q, s in judge5.items()]
    scores += [JudgeScore(qid=q, scale="1-3", score=s) for q, s in judge3.items()]
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
    report = human_agreement(sheet, key, judge)
    r5, r3 = report.scales["1-5"], report.scales["1-3"]
    assert r5.n == 5 and r3.n == 5 and report.blank == 1
    j5, h5 = zip(*human5, strict=True)
    assert r5.weighted_kappa == pytest.approx(weighted_kappa(list(j5), list(h5), low=1, high=5))
    assert r5.exact.count == sum(a == b for a, b in human5)
    assert r5.within_one.count == 5
    assert r3.exact.count == 5 and r3.weighted_kappa == pytest.approx(1.0)
    assert r3.spearman is None or math.isclose(r3.spearman, 1.0)
    other = judge.model_copy(update={"answer_run_id": "different"})
    with pytest.raises(ValueError):
        human_agreement(sheet, key, other)


def test_human_agreement_reports_binary_rates(tmp_path: Path) -> None:
    run = _run(12)
    sheet, key = export_human_sheet(run, n=12, seed=0, out_dir=tmp_path)
    rows, keys = _read(sheet), _read(key)
    qid_of = {k["sample_id"]: k["qid"] for k in keys}
    judge3 = {f"q{i:03d}": (1, 2, 3)[i % 3] for i in range(12)}
    human3 = dict(judge3)
    # One answer the judge scored 3 is 2 for the human; one the judge scored 1 is 3.
    q_lenient = next(q for q, s in judge3.items() if s == 3)
    q_strict = next(q for q, s in judge3.items() if s == 1)
    human3[q_lenient], human3[q_strict] = 2, 3
    for row in rows:
        qid = qid_of[row["sample_id"]]
        row["score_1_5"], row["score_1_3"] = "3", str(human3[qid])
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    scores = [JudgeScore(qid=q, scale="1-5", score=3) for q in judge3]
    scores += [JudgeScore(qid=q, scale="1-3", score=s) for q, s in judge3.items()]
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
    binary = human_agreement(sheet, key, judge).binary
    assert binary is not None and binary.n == 12
    # Human 3: four answers (3 of them judge 3, one judge 1); human 1 or 2: eight answers.
    human_full = [q for q, s in human3.items() if s == 3]
    assert binary.human_full == len(human_full) == 4
    assert binary.tpr.count == sum(judge3[q] == 3 for q in human_full) == 3
    bad = [q for q, s in human3.items() if s != 3]
    assert binary.tnr.count == sum(judge3[q] != 3 for q in bad) == 7 and binary.tnr.n == 8
    assert len(binary.disagreements) == 2
