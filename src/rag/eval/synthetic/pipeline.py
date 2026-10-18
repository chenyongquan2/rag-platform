"""Generate questions from spans and critique them with real or fake chat models (spec P1-05).

The generator (`deepseek-v4-flash`) and the critic (`qwen/qwen3.6-flash`, the P1-04 judge with
thinking off) must be different models: a model rating its own questions favours them (design
s14-3). Every call's ledger lines are collected into its record, so the run's cost per stage is
known. A gateway error is recorded with its category and the run goes on; `budget_exceeded`
stops the run (the ledger refused the call before sending).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from rag.eval.answer.record import Usage
from rag.eval.answer.usage import collect_ledger_rows
from rag.eval.synthetic.prompts import (
    CRITERIA,
    CRITIQUE_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    build_critique_prompt,
    build_generation_prompt,
    parse_critique,
    parse_generation,
)
from rag.eval.synthetic.sampler import Span
from rag.llm import chat_error_category
from rag.types import ErrorCategory


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "deepseek-v4-flash"
    profile: Literal["offline"] = "offline"
    temperature: float = 0.0
    max_tokens: int = 4096
    """Reasoning and answer together (deepseek-v4-flash is served as a reasoning model)."""
    extra_body: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = Field(default=4, ge=1)
    prompt_version: str = GENERATION_PROMPT_VERSION


class CriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "qwen/qwen3.6-flash"
    profile: Literal["offline"] = "offline"
    temperature: float = 0.0
    max_tokens: int = 1024
    extra_body: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})
    """Qwen3 thinks by default; the rationale written before the rating is the reasoning."""
    concurrency: int = Field(default=4, ge=1)
    prompt_version: str = CRITIQUE_PROMPT_VERSION


def check_models_differ(generator: GeneratorConfig, critic: CriticConfig) -> None:
    if generator.model == critic.model:
        raise ValueError(
            f"generator and critic must differ (design s14-3); both are {generator.model!r}"
        )


class GenerationRecord(BaseModel):
    span_id: str
    doc_id: str
    stratum: str
    model: str
    prompt_version: str
    raw: str = ""
    question: str | None = None
    answer: str | None = None
    error: str | None = None
    """`generation_parse_failed` or a chat error category (`llm_timeout`, ...)."""
    usage: Usage = Field(default_factory=Usage)


class CritiqueRecord(BaseModel):
    span_id: str
    criterion: str
    model: str
    prompt_version: str
    raw: str = ""
    score: float | None = None
    rationale: str = ""
    error: str | None = None
    """`critique_parse_failed` or a chat error category."""
    usage: Usage = Field(default_factory=Usage)


TRANSIENT_ERRORS = frozenset({str(ErrorCategory.LLM_TIMEOUT), str(ErrorCategory.LLM_ERROR)})
"""Records with these errors are tried again by the next `generate` / `critique` run."""


def _bind(model: Any, extra_body: dict[str, Any]) -> Any:
    return model.bind(extra_body=extra_body) if extra_body else model


async def _call(model: Any, prompt: str) -> tuple[str | None, str | None, Usage]:
    """(text, error category, usage) of one call; budget_exceeded is raised."""
    with collect_ledger_rows() as rows:
        try:
            message = await model.ainvoke([HumanMessage(prompt)])
        except Exception as exc:
            category = chat_error_category(exc)
            if category is None or category == ErrorCategory.BUDGET_EXCEEDED:
                raise
            return None, str(category), Usage.from_rows(rows)
    content = message.content if isinstance(message.content, str) else str(message.content)
    return content, None, Usage.from_rows(rows)


async def generate_questions(
    spans: Sequence[Span], model: Any, *, config: GeneratorConfig
) -> list[GenerationRecord]:
    """One generation call per span, in span order."""
    bound = _bind(model, config.extra_body)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(span: Span) -> GenerationRecord:
        async with semaphore:
            raw, error, usage = await _call(bound, build_generation_prompt(span.text))
        parsed = parse_generation(raw) if raw is not None else None
        if raw is not None and parsed is None:
            error = "generation_parse_failed"
        return GenerationRecord(
            span_id=span.span_id,
            doc_id=span.doc_id,
            stratum=span.stratum,
            model=config.model,
            prompt_version=config.prompt_version,
            raw=raw or "",
            question=parsed.question if parsed else None,
            answer=parsed.answer if parsed else None,
            error=error,
            usage=usage,
        )

    return list(await asyncio.gather(*(one(s) for s in spans)))


async def critique_questions(
    spans: Sequence[Span],
    generations: Sequence[GenerationRecord],
    model: Any,
    *,
    config: CriticConfig,
    criteria: Sequence[tuple[str, str]] | None = None,
) -> list[CritiqueRecord]:
    """One call per generated question and criterion; `criteria` limits the work to these
    (span_id, criterion) pairs (resuming a run). Generations without a question are skipped."""
    by_span = {s.span_id: s for s in spans}
    wanted = set(criteria) if criteria is not None else None
    bound = _bind(model, config.extra_body)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(gen: GenerationRecord, criterion: str) -> CritiqueRecord:
        prompt = build_critique_prompt(criterion, gen.question or "", by_span[gen.span_id].text)
        async with semaphore:
            raw, error, usage = await _call(bound, prompt)
        parsed = parse_critique(raw) if raw is not None else None
        if parsed is not None and parsed.score is None:
            error = "critique_parse_failed"
        return CritiqueRecord(
            span_id=gen.span_id,
            criterion=criterion,
            model=config.model,
            prompt_version=config.prompt_version,
            raw=raw or "",
            score=parsed.score if parsed else None,
            rationale=parsed.rationale if parsed else "",
            error=error,
            usage=usage,
        )

    jobs = [
        one(gen, criterion)
        for gen in generations
        if gen.question
        for criterion in CRITERIA
        if wanted is None or (gen.span_id, criterion) in wanted
    ]
    return list(await asyncio.gather(*jobs))
