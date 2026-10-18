"""Spec P1-04: the real judge model scores a right answer above a wrong one, on both scales.

Run once when a spec says so: `uv run pytest -m llm tests/llm/test_judge_gateway.py -s`.
Four short judge calls (about 0.01 CNY at the configured high price estimate).
"""

from __future__ import annotations

import asyncio

import pytest

from rag.config import ServiceSettings
from rag.eval.answer import AnswerQuestionRecord, AnswerRunRecord, format_compliance
from rag.eval.answer.cli import build_judge_model
from rag.eval.answer.judge import JudgeConfig, judge_answers
from rag.types import Answer, AnswerSentence, AnswerStatus, Citation

pytestmark = pytest.mark.llm

QUESTION = "What is the default checkpoint of the summarization pipeline in Transformers?"
REFERENCE = "The summarization pipeline uses sshleifer/distilbart-cnn-12-6 by default."


def _record(qid: str, text: str) -> AnswerQuestionRecord:
    answer = Answer(
        request_id=qid,
        kb="hf_docs",
        status=AnswerStatus.ANSWERED,
        found=True,
        sentences=[AnswerSentence(text=text, citations=[Citation(marker=1, chunk_id="c#1")])],
    )
    return AnswerQuestionRecord.from_answer(
        qid=qid,
        question=QUESTION,
        reference=REFERENCE,
        relevant_pages=[],
        answer=answer,
        error=None,
    )


def test_real_judge_orders_right_above_wrong() -> None:
    settings = ServiceSettings()
    if settings.chat_backend != "gateway" or not settings.llm_base_url:
        pytest.skip("the chat gateway is not configured (LLM_BASE_URL, CHAT_BACKEND)")
    records = [
        _record("right", "By default it uses the sshleifer/distilbart-cnn-12-6 checkpoint."),
        _record("wrong", "By default it uses google/pegasus-xsum."),
    ]
    run = AnswerRunRecord(
        run_id="llm-test",
        created_at="t",
        kb="hf_docs",
        kind="hf",
        chat_model="none",
        questions=records,
        compliance=format_compliance(records),
    )
    config = JudgeConfig(max_tokens=400)
    judged = asyncio.run(
        judge_answers(run, model=build_judge_model(settings, config), config=config)
    )
    by = {(s.qid, s.scale): s for s in judged.scores}
    for scale in ("1-5", "1-3"):
        right, wrong = by[("right", scale)], by[("wrong", scale)]
        print(scale, right.raw, "|", wrong.raw)
        assert right.score is not None and wrong.score is not None
        assert right.score > wrong.score
