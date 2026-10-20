"""Spec P3-05 AC7: the judge step (P1-04 judge, 1-3 scale) over the records of both systems."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rag.eval.agentic.judging import judge_records
from rag.eval.agentic.records import (
    JudgeLine,
    SystemRecord,
    append_line,
    read_judge_lines,
)
from rag.eval.answer.judge import JUDGE_PROMPT_VERSION, JudgeConfig
from rag.testing.ledger_chat import LedgerChat


def _record(system: str, qid: str, *, answered: bool, answerable: bool = True) -> SystemRecord:
    return SystemRecord(
        system=system,  # type: ignore[arg-type]
        qid=qid,
        qtype="single_hop" if answerable else "no_answer",
        question=f"Question {qid}?",
        reference=f"Reference {qid}." if answerable else "",
        answerable=answerable,
        status="answered" if answered else "not_found",
        found=answered,
        answered=answered,
        text=f"Answer of {system} to {qid}." if answered else "",
    )


RECORDS = [
    _record("standard", "q1", answered=True),
    _record("standard", "q2", answered=False),
    _record("agentic", "q1", answered=True),
    _record("agentic", "q2", answered=True),
    _record("agentic", "q3", answered=False, answerable=False),  # no-answer: refusal, not judged
]


def _model(tmp_path: Path, *replies: str) -> LedgerChat:
    return LedgerChat(
        responses=list(replies),
        default_response="Feedback: fine. [RESULT] 3",
        ledger_path=str(tmp_path / "ledger.jsonl"),
        profile="offline",
    )


def _judge(tmp_path: Path, model: Any, *, run: int = 1) -> list[JudgeLine]:
    return asyncio.run(
        judge_records(
            RECORDS,
            run=run,
            model=model,
            config=JudgeConfig(),
            out_path=tmp_path / "judge.jsonl",
        )
    )


def test_judge_lines_and_rule_scores(tmp_path: Path) -> None:
    model = _model(tmp_path, "Feedback: wrong. [RESULT] 1")
    new = _judge(tmp_path, model)
    lines = {(x.system, x.qid): x for x in read_judge_lines(tmp_path / "judge.jsonl")}
    assert len(new) == 4 and set(lines) == {
        ("standard", "q1"),
        ("standard", "q2"),
        ("agentic", "q1"),
        ("agentic", "q2"),
    }  # the no-answer question is not judged
    # No answer: the lowest score by rule, without a call (as in P1-04).
    assert (lines["standard", "q2"].score, lines["standard", "q2"].by_rule) == (1, "no_answer")
    assert lines["standard", "q2"].usage.calls == 0
    assert len(model.calls) == 3  # three answers were read by the judge
    scores = sorted(x.score for k, x in lines.items() if x.by_rule is None and x.score is not None)
    assert scores == [1, 3, 3]
    called = lines["agentic", "q2"]
    assert called.usage.calls == 1 and called.usage.cny > 0
    assert (called.judge_model, called.prompt_version) == (
        "qwen/qwen3.6-flash",
        JUDGE_PROMPT_VERSION,
    )
    assert called.run == 1 and called.feedback
    # The prompt is the 1-3 rubric with the reference marked as the top score.
    prompt = "\n".join(str(m.content) for m in model.calls[0])
    assert "Reference q1." in prompt and "Answer of standard to q1." in prompt
    assert "score that is an integer between 1 and 3" in prompt


def test_judge_resume(tmp_path: Path) -> None:
    _judge(tmp_path, _model(tmp_path))
    again = _model(tmp_path)
    assert _judge(tmp_path, again) == []  # everything of run 1 is there
    assert again.calls == []
    second = _model(tmp_path)
    new = _judge(tmp_path, second, run=2)  # a second pass is new work
    assert len(new) == 4 and len(second.calls) == 3
    lines = read_judge_lines(tmp_path / "judge.jsonl")
    assert sorted({x.run for x in lines}) == [1, 2] and len(lines) == 8


def test_judge_unparseable_output_is_a_failed_score(tmp_path: Path) -> None:
    out = tmp_path / "judge.jsonl"
    append_line(out, JudgeLine(system="standard", qid="q1", run=1, score=3))  # already done
    model = _model(tmp_path, "no verdict here", "still none")
    new = _judge(tmp_path, model)
    failed = [x for x in new if x.error]
    assert len(failed) == 2 and all(
        x.score is None and x.error == "judge_parse_failed" for x in failed
    )
