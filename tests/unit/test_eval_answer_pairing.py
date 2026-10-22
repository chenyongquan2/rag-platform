"""Spec P7-01 AC3, AC4: two chat models paired question by question on one question set."""

from __future__ import annotations

import pytest

from rag.eval.answer.pairing import PairedOutcome, paired_answer_table, render_paired_answers
from rag.eval.stats import mcnemar_exact


def _o(
    qid: str,
    score: int | None,
    *,
    cited: bool = True,
    recall: float | None = 1.0,
    ms: float = 1000.0,
    cny: float = 0.002,
    pairs: tuple[int, int] = (1, 1),
) -> PairedOutcome:
    return PairedOutcome(
        qid=qid,
        score=score,
        found=score is not None and score > 1,
        fully_cited=cited,
        citation_recall=recall,
        supported_pairs=pairs[0],
        judged_pairs=pairs[1],
        generation_ms=ms,
        cny=cny,
    )


def _systems() -> tuple[dict[str, PairedOutcome], dict[str, PairedOutcome]]:
    base = {
        "q1": _o("q1", 3),
        "q2": _o("q2", 3),
        "q3": _o("q3", 2, cited=False, recall=0.5),
        "q4": _o("q4", 1, recall=None),
        "only_base": _o("only_base", 3),
    }
    cand = {
        "q1": _o("q1", 3, ms=500.0, cny=0.001),
        "q2": _o("q2", 2, ms=500.0, cny=0.001, pairs=(0, 1)),
        "q3": _o("q3", 3, recall=1.0, ms=500.0, cny=0.001),
        "q4": _o("q4", 3, ms=500.0, cny=0.001),
        "only_cand": _o("only_cand", 1),
    }
    return base, cand


def test_paired_table_counts_and_tests() -> None:
    base, cand = _systems()
    table = paired_answer_table(base, cand, iterations=2000, seed=0)
    assert table.n == 4  # only the common question ids
    rows = {r.metric: r for r in table.rows}

    s3 = rows["score 3"]
    assert (s3.base, s3.candidate) == (2, 3)  # counts of questions with score 3
    assert (s3.better, s3.worse, s3.unchanged) == (2, 1, 1)
    assert s3.p_value == pytest.approx(mcnemar_exact(2, 1))

    cited = rows["every sentence cited"]
    assert (cited.better, cited.worse) == (1, 0)

    score = rows["judge score (1-3)"]
    assert score.test == "permutation"
    assert score.mean_diff == pytest.approx((0 - 1 + 1 + 2) / 4)
    assert 0.0 < score.p_value <= 1.0

    recall = rows["citation recall per answer"]
    assert recall.n == 3  # q4 has no recall on the base side
    assert recall.mean_diff == pytest.approx((0 + 0 + 0.5) / 3)

    assert table.base_precision.count == 4 and table.base_precision.n == 4
    assert table.candidate_precision.count == 3 and table.candidate_precision.n == 4
    assert table.base_latency_p50 == pytest.approx(1000.0)
    assert table.candidate_latency_p50 == pytest.approx(500.0)
    assert table.base_cny_mean == pytest.approx(0.002)
    assert table.candidate_cny_mean == pytest.approx(0.001)


def test_render_has_metrics_p_latency_cost() -> None:
    base, cand = _systems()
    table = paired_answer_table(base, cand, iterations=2000, seed=0)
    text = render_paired_answers(table, base_name="deepseek-v4-flash", candidate_name="gemini")
    assert "deepseek-v4-flash" in text and "gemini" in text
    assert "n = 4" in text
    for metric in ("score 3", "every sentence cited", "judge score (1-3)", "citation recall"):
        assert metric in text
    assert "p =" in text or "| p |" in text
    assert "P50" in text and "P95" in text
    assert "CNY per question" in text


def test_no_common_question_is_an_error() -> None:
    with pytest.raises(ValueError, match="no common"):
        paired_answer_table({"a": _o("a", 3)}, {"b": _o("b", 3)})
