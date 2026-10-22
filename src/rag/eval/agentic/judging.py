"""Score the answers of both systems with the P1-04 judge (spec P3-05, design s14-3).

Same judge as P1-04, unchanged: `qwen/qwen3.6-flash`, prompt version `p1-04-hf-v1`, temperature 0,
thinking off, the 1-3 scale (HF agent_rag). Only answerable questions are judged; a question with
no answer (not found, parse failure, error, a clarification instead of an answer) gets 1 by rule
without a call. Every score is appended to the judge file as soon as it exists, and a line that
is already there (same system, question and pass) is not asked again, so a crash loses nothing
(except a by-rule score of a question that has been answered since: that one is judged again).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.eval.agentic.records import (
    JudgeLine,
    SystemRecord,
    append_line,
    read_judge_lines,
    repair_tail,
)
from rag.eval.answer.judge import JudgeConfig, _judge_one  # the P1-04 call, reused as it is
from rag.eval.answer.record import AnswerQuestionRecord

SCALE = "1-3"


async def _score(
    record: SystemRecord, run: int, bound: Any, semaphore: asyncio.Semaphore, config: JudgeConfig
) -> JudgeLine:
    if not record.answered:
        return JudgeLine(
            system=record.system,
            qid=record.qid,
            run=run,
            score=1,
            by_rule="no_answer",
            judge_model=config.model,
            prompt_version=config.prompt_version,
        )
    as_answer = AnswerQuestionRecord(
        qid=record.qid,
        question=record.question,
        reference=record.reference,
        text=record.text,
        found=True,
    )
    scored = await _judge_one(as_answer, SCALE, bound, semaphore)
    return JudgeLine(
        system=record.system,
        qid=record.qid,
        run=run,
        score=scored.score,
        error=scored.error,
        feedback=scored.feedback,
        strict_format=scored.strict_format,
        usage=scored.usage,
        judge_model=config.model,
        prompt_version=config.prompt_version,
    )


async def judge_records(
    records: Sequence[SystemRecord],
    *,
    run: int,
    model: Any,
    config: JudgeConfig,
    out_path: Path,
) -> list[JudgeLine]:
    """Judge pass `run` over the answerable `records` that have no line yet; returns the new
    lines (appended to `out_path` one by one)."""
    repair_tail(out_path)
    answered = {(r.system, r.qid) for r in records if r.answered}
    # A rule score ("no answer") of a question that has been answered since (re-run by
    # `--retry-errors`) is stale: judge the new answer; the later line wins when scores are read.
    have = {
        (x.system, x.qid)
        for x in read_judge_lines(out_path)
        if x.run == run and not (x.by_rule == "no_answer" and (x.system, x.qid) in answered)
    }
    todo = [r for r in records if r.answerable and (r.system, r.qid) not in have]
    bound = model.bind(extra_body=config.extra_body) if config.extra_body else model
    semaphore = asyncio.Semaphore(config.concurrency)
    new: list[JudgeLine] = []

    async def one(record: SystemRecord) -> None:
        line = await _score(record, run, bound, semaphore, config)
        append_line(out_path, line)
        new.append(line)

    await asyncio.gather(*(one(r) for r in todo))
    return new
