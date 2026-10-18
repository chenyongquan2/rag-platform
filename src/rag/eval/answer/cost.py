"""Project the cost of the full answer eval from a pilot run (spec P1-04, CLAUDE.md budget rule)."""

from __future__ import annotations

from pydantic import BaseModel

from rag.eval.answer.record import AnswerRunRecord


class CostProjection(BaseModel):
    pilot_questions: int
    mean_prompt_tokens: float
    """Per question, all chat calls (a parse retry included)."""
    mean_completion_tokens: float
    mean_reasoning_tokens: float
    per_answer_cny: float
    judge_cny_per_call: float
    questions: int
    judge_calls_per_answer: int
    security_questions: int
    answers_cny: float
    judge_cny: float
    security_cny: float
    total_cny: float


def project_cost(
    pilot: AnswerRunRecord,
    *,
    questions: int,
    judge_cny_per_call: float,
    judge_calls_per_answer: int = 4,
    security_questions: int = 30,
) -> CostProjection:
    """Scale the pilot's real chat cost per question to `questions` answers, judge calls
    (`judge_calls_per_answer`: scales x repeats) and the security answers (same cost per answer
    assumed; their context is one planted page, usually shorter)."""
    n = len(pilot.questions)
    if n == 0:
        raise ValueError("the pilot run has no questions")
    chat = [q.usage_by_kind.get("chat") for q in pilot.questions]
    prompt = sum(u.prompt_tokens for u in chat if u) / n
    completion = sum(u.completion_tokens for u in chat if u) / n
    reasoning = sum(u.reasoning_tokens for u in chat if u) / n
    per_answer = sum(u.cny for u in chat if u) / n
    answers = questions * per_answer
    judge = questions * judge_calls_per_answer * judge_cny_per_call
    security = security_questions * per_answer
    return CostProjection(
        pilot_questions=n,
        mean_prompt_tokens=prompt,
        mean_completion_tokens=completion,
        mean_reasoning_tokens=reasoning,
        per_answer_cny=per_answer,
        judge_cny_per_call=judge_cny_per_call,
        questions=questions,
        judge_calls_per_answer=judge_calls_per_answer,
        security_questions=security_questions,
        answers_cny=answers,
        judge_cny=judge,
        security_cny=security,
        total_cny=answers + judge + security,
    )
