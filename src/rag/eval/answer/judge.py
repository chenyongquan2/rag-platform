# ruff: noqa: RUF001 (the judge may write a full-width colon after [RESULT])
"""LLM judge of answer correctness against the reference answer (spec P1-04, design s14-3, s13-5).

The prompt follows the Hugging Face cookbook judges (rag_evaluation for 1-5, agent_rag for 1-3,
Apache-2.0): the question, the response, the reference answer marked as the top score, one line
per score, and the output "Feedback: ... [RESULT] n", so the judge writes its reasons before the
score. Temperature 0. A record without an answer (not found, parse failure, gateway or retrieval
error) gets the lowest score by rule, without a call: every question of the set has an answer in
the documentation, so not answering is wrong.

Summary per scale: the mean score and the percent HF reports, the mean of
`(score - low) / (high - low)`; for 1-3 that is HF agent_rag's `(score - 1) / 2`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag.eval.answer.record import AnswerQuestionRecord, AnswerRunRecord, Proportion, Usage
from rag.eval.answer.usage import collect_ledger_rows
from rag.eval.retrieval.runner import make_run_id
from rag.eval.stats import weighted_kappa
from rag.llm import chat_error_category

JudgeScale = Literal["1-5", "1-3"]
SCALES: dict[str, tuple[int, int]] = {"1-5": (1, 5), "1-3": (1, 3)}
JUDGE_PROMPT_VERSION = "p1-04-hf-v1"
"""Bump when the prompt text or the parse rules change: scores of two versions do not compare."""

SYSTEM_MESSAGE = "You are a fair evaluator language model."

_RUBRICS: dict[str, list[str]] = {
    "1-5": [
        "The response is completely incorrect, inaccurate, and/or not factual.",
        "The response is mostly incorrect, inaccurate, and/or not factual.",
        "The response is somewhat correct, accurate, and/or factual.",
        "The response is mostly correct, accurate, and factual.",
        "The response is completely correct, accurate, and factual.",
    ],
    "1-3": [
        "The response is completely incorrect, inaccurate, and/or not factual.",
        "The response is somewhat correct, accurate, and/or factual.",
        "The response is completely correct, accurate, and/or factual.",
    ],
}

_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of {top}, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between {low} and {top}. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between {low} and {top}}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score {top}):
{reference}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
{rubric}

###Feedback:"""  # noqa: E501 (prompt text, kept on one line per instruction)

_RESULT = re.compile(r"\[RESULT\]\s*[:：]?\s*\**\s*(\d+)", re.IGNORECASE)
_LOOSE = (
    re.compile(r"\bscore\s*(?:is|of)?\s*[:：]?\s*\**\s*(\d+)\s*\**\s*(?:/\s*\d+)?\s*\.?\s*$", re.I),
    re.compile(r"\[\s*(\d+)\s*\]\s*\.?\s*$"),
    re.compile(r"(?:^|\n)\s*\**(\d+)\**\s*\.?\s*$"),  # a number alone on the last line
)
_FEEDBACK = re.compile(r"^\s*(?:#+\s*)?feedback\s*[:：]\s*", re.IGNORECASE)


def build_judge_prompt(scale: str, *, question: str, response: str, reference: str) -> str:
    low, top = SCALES[scale]
    rubric = "\n".join(f"Score {i}: {line}" for i, line in enumerate(_RUBRICS[scale], start=low))
    return _TEMPLATE.format(
        low=low,
        top=top,
        question=question.strip(),
        response=response.strip(),
        reference=reference.strip(),
        rubric=rubric,
    )


class JudgeParse(BaseModel):
    feedback: str
    score: int | None
    strict: bool
    """The score came after `[RESULT]`; False for a tolerated variant (`Score: 2`, `[2]`)."""


def parse_judge_output(text: str, scale: str) -> JudgeParse:
    """Score after the last `[RESULT]`; else a final `Score: n`, `[n]` or bare `n` (flagged);
    a number outside the scale counts as no score."""
    low, top = SCALES[scale]
    text = text or ""
    strict = True
    matches = list(_RESULT.finditer(text))
    if matches:
        match: re.Match[str] | None = matches[-1]
        feedback = text[: matches[-1].start()]
    else:
        strict = False
        match = None
        stripped = text.rstrip()
        for pattern in _LOOSE:
            match = pattern.search(stripped)
            if match:
                break
        feedback = stripped[: match.start()] if match else text
    feedback = _FEEDBACK.sub("", feedback).strip()
    score = int(match.group(1)) if match else None
    if score is not None and not low <= score <= top:
        score = None
    return JudgeParse(feedback=feedback, score=score, strict=strict)


class JudgeConfig(BaseModel):
    """Recorded with every judge run (design s14-3: judge model and prompt version)."""

    model: str = "qwen/qwen3.6-flash"
    """A different family from the generator (`deepseek-v4-flash`); see docs/decisions.md."""
    profile: Literal["offline"] = "offline"
    temperature: float = 0.0
    max_tokens: int = 1024
    extra_body: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})
    """Sent with every call; Qwen3 models think by default (600 reasoning tokens for "2+2")."""
    concurrency: int = Field(default=4, ge=1)
    prompt_version: str = JUDGE_PROMPT_VERSION


class JudgeScore(BaseModel):
    qid: str
    scale: str
    score: int | None = None
    feedback: str = ""
    raw: str = ""
    by_rule: str | None = None
    """`no_answer`: lowest score given without a call."""
    strict_format: bool = True
    error: str | None = None
    """`judge_parse_failed` or a chat error category (`llm_timeout`, ...)."""
    usage: Usage = Field(default_factory=Usage)


class ScaleSummary(BaseModel):
    scale: str
    n_scored: int
    n_rule: int
    """Of the scored, given by rule (no answer)."""
    n_failed: int
    """No score: the judge's output did not parse or the call failed."""
    n_loose_format: int = 0
    mean: float | None
    percent: float | None
    """Mean of (score - low) / (high - low), times 100 (HF agent_rag's percentage on 1-3)."""
    distribution: dict[str, int]


class JudgeRun(BaseModel):
    run_id: str
    created_at: str
    answer_run_id: str
    kb: str
    judge_model: str
    prompt_version: str
    config: JudgeConfig
    scales: list[str]
    scores: list[JudgeScore]
    summary: dict[str, ScaleSummary]
    usage: Usage = Field(default_factory=Usage)


def summarize_scores(
    scores: Sequence[JudgeScore], scales: Sequence[str]
) -> dict[str, ScaleSummary]:
    out: dict[str, ScaleSummary] = {}
    for scale in scales:
        low, top = SCALES[scale]
        rows = [s for s in scores if s.scale == scale]
        values = [s.score for s in rows if s.score is not None]
        out[scale] = ScaleSummary(
            scale=scale,
            n_scored=len(values),
            n_rule=sum(s.by_rule is not None for s in rows),
            n_failed=sum(s.score is None for s in rows),
            n_loose_format=sum(not s.strict_format and s.score is not None for s in rows),
            mean=sum(values) / len(values) if values else None,
            percent=(
                100.0 * sum((v - low) / (top - low) for v in values) / len(values)
                if values
                else None
            ),
            distribution={str(v): values.count(v) for v in range(low, top + 1)},
        )
    return out


def _has_answer(record: AnswerQuestionRecord) -> bool:
    return record.answer is not None and record.found and bool(record.text.strip())


async def _judge_one(
    record: AnswerQuestionRecord,
    scale: str,
    model: Any,
    semaphore: asyncio.Semaphore,
) -> JudgeScore:
    prompt = build_judge_prompt(
        scale, question=record.question, response=record.text, reference=record.reference
    )
    messages = [SystemMessage(SYSTEM_MESSAGE), HumanMessage(prompt)]
    async with semaphore:
        with collect_ledger_rows() as rows:
            try:
                message = await model.ainvoke(messages)
            except Exception as exc:
                category = chat_error_category(exc)
                if category is None:
                    raise
                return JudgeScore(
                    qid=record.qid, scale=scale, error=str(category), usage=Usage.from_rows(rows)
                )
    raw = message.content if isinstance(message.content, str) else str(message.content)
    parsed = parse_judge_output(raw, scale)
    return JudgeScore(
        qid=record.qid,
        scale=scale,
        score=parsed.score,
        feedback=parsed.feedback,
        raw=raw,
        strict_format=parsed.strict,
        error=None if parsed.score is not None else "judge_parse_failed",
        usage=Usage.from_rows(rows),
    )


async def judge_answers(
    run: AnswerRunRecord,
    *,
    model: BaseChatModel,
    config: JudgeConfig,
    scales: Sequence[str] = ("1-5", "1-3"),
    label: str = "",
    now: datetime | None = None,
) -> JudgeRun:
    """Score every answer of `run` on each scale (one call per answer and scale)."""
    for scale in scales:
        if scale not in SCALES:
            raise ValueError(f"unknown scale {scale!r}; choose from {list(SCALES)}")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    bound: Any = model.bind(extra_body=config.extra_body) if config.extra_body else model
    semaphore = asyncio.Semaphore(config.concurrency)
    jobs: list[Any] = []
    for scale in scales:
        low, _ = SCALES[scale]
        for record in run.questions:
            if _has_answer(record):
                jobs.append(_judge_one(record, scale, bound, semaphore))
            else:
                jobs.append(_rule_score(record.qid, scale, low))
    scores = list(await asyncio.gather(*jobs))
    total = Usage()
    for s in scores:
        total = total + s.usage
    return JudgeRun(
        run_id=make_run_id(run.kb, f"judge-{label}" if label else "judge", now),
        created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        answer_run_id=run.run_id,
        kb=run.kb,
        judge_model=config.model,
        prompt_version=config.prompt_version,
        config=config,
        scales=list(scales),
        scores=scores,
        summary=summarize_scores(scores, scales),
        usage=total,
    )


async def _rule_score(qid: str, scale: str, low: int) -> JudgeScore:
    return JudgeScore(qid=qid, scale=scale, score=low, by_rule="no_answer")


class Stability(BaseModel):
    """Two judge runs on the same answers (design s14-3: share of questions whose score moves)."""

    scale: str
    n: int
    """Questions scored by the judge model in both runs."""
    n_rule_excluded: int
    n_missing: int
    """No score in one of the runs."""
    flips: Proportion
    mean_abs_diff: float | None
    max_abs_diff: int | None
    weighted_kappa: float | None
    changed: list[dict[str, Any]]


def judge_stability(a: JudgeRun, b: JudgeRun, scale: str) -> Stability:
    low, top = SCALES[scale]
    first = {s.qid: s for s in a.scores if s.scale == scale}
    second = {s.qid: s for s in b.scores if s.scale == scale}
    pairs: list[tuple[str, int, int]] = []
    rule = missing = 0
    for qid, x in first.items():
        y = second.get(qid)
        if y is None:
            missing += 1
        elif x.by_rule or y.by_rule:
            rule += 1
        elif x.score is None or y.score is None:
            missing += 1
        else:
            pairs.append((qid, x.score, y.score))
    diffs = [abs(x - y) for _, x, y in pairs]
    changed = [{"qid": q, "a": x, "b": y} for q, x, y in pairs if x != y]
    return Stability(
        scale=scale,
        n=len(pairs),
        n_rule_excluded=rule,
        n_missing=missing,
        flips=Proportion.of(len(changed), len(pairs)),
        mean_abs_diff=sum(diffs) / len(diffs) if diffs else None,
        max_abs_diff=max(diffs) if diffs else None,
        weighted_kappa=weighted_kappa(
            [x for _, x, _ in pairs], [y for _, _, y in pairs], low=low, high=top
        )
        if pairs
        else None,
        changed=changed,
    )


def render_judge_summary(run: JudgeRun) -> str:
    lines = [
        f"# Judge run {run.run_id}",
        "",
        f"- answers: {run.answer_run_id}",
        f"- judge: {run.judge_model}, prompt {run.prompt_version}, temperature "
        f"{run.config.temperature}, extra_body {run.config.extra_body}",
        f"- usage: {run.usage.calls} calls, prompt {run.usage.prompt_tokens}, completion "
        f"{run.usage.completion_tokens}, {run.usage.cny:.4f} CNY",
        "",
        "| scale | scored | by rule | failed | loose format | mean | percent | distribution |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in run.summary.values():
        mean = "n/a" if s.mean is None else f"{s.mean:.3f}"
        pct = "n/a" if s.percent is None else f"{s.percent:.1f}%"
        lines.append(
            f"| {s.scale} | {s.n_scored} | {s.n_rule} | {s.n_failed} | {s.n_loose_format} | "
            f"{mean} | {pct} | {s.distribution} |"
        )
    return "\n".join(lines) + "\n"


def write_judge_run(run: JudgeRun, out_root: str | Path) -> Path:
    out = Path(out_root) / "judge" / run.run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    (out / "results.json").write_text(
        run.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "summary.md").write_text(render_judge_summary(run), encoding="utf-8", newline="\n")
    return out


def load_judge_run(path: str | Path) -> JudgeRun:
    file = Path(path)
    if file.is_dir():
        file = file / "results.json"
    return JudgeRun.model_validate_json(file.read_text(encoding="utf-8"))
