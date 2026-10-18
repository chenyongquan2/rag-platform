"""Spec P1-05 AC9: generation and critique with fake chat models."""

from __future__ import annotations

import asyncio

import pytest

from rag.eval.synthetic.pipeline import (
    CriticConfig,
    GeneratorConfig,
    check_models_differ,
    critique_questions,
    generate_questions,
)
from rag.eval.synthetic.prompts import CRITIQUE_PROMPT_VERSION, GENERATION_PROMPT_VERSION
from rag.eval.synthetic.sampler import Span
from rag.testing import FakeChatModel


def _span(i: int) -> Span:
    return Span(
        span_id=f"s{i}",
        doc_id=f"hf_docs:repo/p{i}.md",
        source_rel=f"repo/p{i}.md",
        doc_type="markdown",
        stratum="repo:markdown",
        start=0,
        end=20,
        text=f"Span text number {i}.",
    )


def test_pipeline_generate_and_critique() -> None:
    spans = [_span(1), _span(2)]
    generator = FakeChatModel(
        responses=["Factoid question: What is one?\nAnswer: One.", "I cannot write a question."]
    )
    gens = asyncio.run(
        generate_questions(spans, generator, config=GeneratorConfig(model="gen-x", concurrency=1))
    )
    assert [g.span_id for g in gens] == ["s1", "s2"]
    assert gens[0].question == "What is one?" and gens[0].answer == "One."
    assert gens[0].model == "gen-x" and gens[0].prompt_version == GENERATION_PROMPT_VERSION
    assert gens[1].question is None and gens[1].error == "generation_parse_failed"
    assert "Span text number 1." in generator.calls[0][-1].content
    assert gens[0].usage.calls == 0  # fakes write no ledger line

    critic = FakeChatModel(default_response="Evaluation: fine.\nTotal rating: 4")
    config = CriticConfig(model="critic-y", concurrency=1)
    scores = asyncio.run(critique_questions(spans, gens, critic, config=config))
    # only the parsed question is scored, once per criterion
    assert sorted((s.span_id, s.criterion) for s in scores) == [
        ("s1", "groundedness"),
        ("s1", "relevance"),
        ("s1", "standalone"),
    ]
    assert all(s.score == 4 and s.rationale == "fine." for s in scores)
    assert all(s.model == "critic-y" for s in scores)
    assert all(s.prompt_version == CRITIQUE_PROMPT_VERSION for s in scores)
    prompts = {s.criterion: m[-1].content for s, m in zip(scores, critic.calls, strict=True)}
    assert "Span text number 1." in prompts["groundedness"]
    assert "Span text number 1." not in prompts["relevance"]

    # a failed parse is recorded, not raised
    bad = FakeChatModel(default_response="no rating")
    scores = asyncio.run(critique_questions(spans, gens, bad, config=config))
    assert all(s.score is None and s.error == "critique_parse_failed" for s in scores)

    with pytest.raises(ValueError, match="differ"):
        check_models_differ(GeneratorConfig(model="m"), CriticConfig(model="m"))
    check_models_differ(GeneratorConfig(), CriticConfig())
    assert GeneratorConfig().model == "deepseek-v4-flash"
    assert CriticConfig().model == "qwen/qwen3.6-flash"
    assert CriticConfig().extra_body == {"enable_thinking": False}
