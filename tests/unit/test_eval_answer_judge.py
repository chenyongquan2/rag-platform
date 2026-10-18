"""Spec P1-04: judge prompt, output parsing, judge run, stability."""

from __future__ import annotations

import asyncio

import pytest

from rag.eval.answer import AnswerQuestionRecord, AnswerRunRecord, format_compliance
from rag.eval.answer.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeConfig,
    JudgeRun,
    JudgeScore,
    build_judge_prompt,
    judge_answers,
    judge_stability,
    parse_judge_output,
    summarize_scores,
)
from rag.testing import FakeChatModel
from rag.types import Answer, AnswerSentence, AnswerStatus, Citation


def _record(qid: str, text: str | None, *, found: bool = True) -> AnswerQuestionRecord:
    answer = None
    if text is not None:
        answer = Answer(
            request_id=qid,
            kb="hf_docs",
            status=AnswerStatus.ANSWERED if found else AnswerStatus.NOT_FOUND,
            found=found,
            sentences=[AnswerSentence(text=text, citations=[Citation(marker=1, chunk_id="c#1")])]
            if found
            else [AnswerSentence(text=text)],
        )
    return AnswerQuestionRecord.from_answer(
        qid=qid,
        question=f"question {qid}?",
        reference=f"reference {qid}",
        relevant_pages=[],
        answer=answer,
        error=None if answer else "RuntimeError: boom",
    )


def _run(records: list[AnswerQuestionRecord]) -> AnswerRunRecord:
    return AnswerRunRecord(
        run_id="20261018T000000Z_hf_docs_test",
        created_at="2026-10-18T00:00:00Z",
        kb="hf_docs",
        label="test",
        kind="hf",
        chat_model="fake-chat",
        questions=records,
        compliance=format_compliance(records),
    )


@pytest.mark.parametrize("scale, top", [("1-5", 5), ("1-3", 3)])
def test_judge_prompt_contents(scale: str, top: int) -> None:
    prompt = build_judge_prompt(
        scale, question="What is X?", response="X is Y.", reference="X is Y indeed."
    )
    assert "What is X?" in prompt and "X is Y." in prompt and "X is Y indeed." in prompt
    assert f"Reference Answer (Score {top})" in prompt
    for score in range(1, top + 1):
        assert f"Score {score}:" in prompt
    assert f"Score {top + 1}:" not in prompt
    assert prompt.index("Feedback:") < prompt.index("[RESULT]")
    assert f"an integer between 1 and {top}" in prompt
    assert JUDGE_PROMPT_VERSION


def test_parse_judge_output() -> None:
    p = parse_judge_output("Feedback: good, matches. [RESULT] 4", "1-5")
    assert (p.score, p.strict, p.feedback) == (4, True, "good, matches.")
    p = parse_judge_output("Feedback: ok [RESULT]: 3\n", "1-3")
    assert (p.score, p.strict) == (3, True)
    p = parse_judge_output("Feedback: fine. [RESULT] **5**", "1-5")
    assert (p.score, p.strict) == (5, True)
    # tolerated, flagged: no [RESULT] but a final score
    p = parse_judge_output("Feedback: mostly right. Score: 2", "1-3")
    assert (p.score, p.strict) == (2, False)
    p = parse_judge_output("Feedback: mostly right. [2]", "1-3")
    assert (p.score, p.strict) == (2, False)
    # out of range, missing
    assert parse_judge_output("Feedback: x [RESULT] 5", "1-3").score is None
    assert parse_judge_output("Feedback: no score at all", "1-5").score is None
    assert parse_judge_output("", "1-5").score is None


def test_judge_answers_scores_and_summary() -> None:
    records = [
        _record("q1", "Right answer."),
        _record("q2", "Half right."),
        _record("q3", "The documentation does not contain the answer.", found=False),
        _record("q4", None),
    ]
    model = FakeChatModel(
        responses=[
            "Feedback: correct. [RESULT] 5",
            "Feedback: partly. [RESULT] 3",
            "Feedback: correct. [RESULT] 3",
            "Feedback: garbled",
        ]
    )
    config = JudgeConfig(model="fake-judge", concurrency=1)
    run = asyncio.run(
        judge_answers(_run(records), model=model, config=config, scales=["1-5", "1-3"], label="t")
    )
    assert isinstance(run, JudgeRun)
    assert run.judge_model == "fake-judge" and run.prompt_version == JUDGE_PROMPT_VERSION
    assert run.answer_run_id == "20261018T000000Z_hf_docs_test"
    assert len(model.calls) == 4  # q3 and q4 scored by rule, twice each
    prompts = [str(call[-1].content) for call in model.calls]
    assert "Right answer." in prompts[0] and "Score 5:" in prompts[0]
    assert "Score 4:" not in prompts[2]  # 1-3 prompt
    by = {(s.qid, s.scale): s for s in run.scores}
    assert by[("q1", "1-5")].score == 5 and by[("q1", "1-5")].by_rule is None
    assert by[("q3", "1-5")].score == 1 and by[("q3", "1-5")].by_rule == "no_answer"
    assert by[("q4", "1-3")].score == 1 and by[("q4", "1-3")].by_rule == "no_answer"
    assert by[("q2", "1-3")].score is None and by[("q2", "1-3")].error == "judge_parse_failed"
    s5 = run.summary["1-5"]
    assert (s5.n_scored, s5.n_rule, s5.n_failed) == (4, 2, 0)
    assert s5.mean == pytest.approx((5 + 3 + 1 + 1) / 4)
    assert s5.percent == pytest.approx(100 * ((4 + 2 + 0 + 0) / 4) / 4)
    s3 = run.summary["1-3"]
    assert (s3.n_scored, s3.n_failed) == (3, 1)
    assert s3.percent == pytest.approx(100 * ((2 / 2) + 0 + 0) / 3)
    assert s3.distribution == {"1": 2, "2": 0, "3": 1}


def _score(qid: str, scale: str, score: int | None, rule: str | None = None) -> JudgeScore:
    return JudgeScore(qid=qid, scale=scale, score=score, by_rule=rule)


def _judge_run(scores: list[JudgeScore]) -> JudgeRun:
    return JudgeRun(
        run_id="j",
        created_at="t",
        answer_run_id="a",
        kb="hf_docs",
        judge_model="m",
        prompt_version=JUDGE_PROMPT_VERSION,
        config=JudgeConfig(model="m"),
        scales=["1-5"],
        scores=scores,
        summary=summarize_scores(scores, ["1-5"]),
    )


def test_judge_stability() -> None:
    a = _judge_run(
        [
            _score("q1", "1-5", 5),
            _score("q2", "1-5", 4),
            _score("q3", "1-5", 2),
            _score("q4", "1-5", 1, "no_answer"),
            _score("q5", "1-5", None),
        ]
    )
    b = _judge_run(
        [
            _score("q1", "1-5", 5),
            _score("q2", "1-5", 3),
            _score("q3", "1-5", 2),
            _score("q4", "1-5", 1, "no_answer"),
            _score("q5", "1-5", 4),
        ]
    )
    s = judge_stability(a, b, "1-5")
    assert s.n == 3
    assert s.n_rule_excluded == 1 and s.n_missing == 1
    assert (s.flips.count, s.flips.n) == (1, 3)
    assert 0 < s.flips.low < 1 / 3 < s.flips.high
    assert s.mean_abs_diff == pytest.approx(1 / 3)
    assert s.max_abs_diff == 1
    assert s.changed == [{"qid": "q2", "a": 4, "b": 3}]
    assert s.weighted_kappa is not None and 0 < s.weighted_kappa < 1
