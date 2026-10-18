"""Spec P1-05 AC4-AC5: generation and critique prompts (HF rag_evaluation style) and parsing."""

from __future__ import annotations

from rag.eval.synthetic.prompts import (
    CRITERIA,
    build_critique_prompt,
    build_generation_prompt,
    parse_critique,
    parse_generation,
)

SPAN = "Set `push_to_hub=True` in `TrainingArguments` to upload the model after training."


def test_generation_prompt_and_parse() -> None:
    prompt = build_generation_prompt(SPAN)
    assert SPAN in prompt
    assert "factoid question" in prompt
    assert "search engine" in prompt
    assert "MUST NOT mention" in prompt
    assert "do not copy" in prompt.lower()
    assert prompt.rstrip().endswith("Output:::")

    out = parse_generation(
        "Output:::\nFactoid question: How do I upload a trained model automatically?\n"
        "Answer: Set push_to_hub=True in TrainingArguments.\n"
    )
    assert out is not None
    assert out.question == "How do I upload a trained model automatically?"
    assert out.answer == "Set push_to_hub=True in TrainingArguments."
    # markdown bold and a multi-line answer
    out = parse_generation("**Factoid question:** What flag uploads?\n**Answer:** `push_to_hub`")
    assert (
        out is not None and out.question == "What flag uploads?" and out.answer == "`push_to_hub`"
    )
    assert parse_generation("Factoid question: only a question?") is None
    assert parse_generation("Answer: only an answer") is None
    assert parse_generation("Factoid question:   \nAnswer: x") is None
    assert parse_generation("") is None


def test_critique_prompts_and_parse() -> None:
    assert CRITERIA == ("groundedness", "relevance", "standalone")
    q = "How do I upload a trained model automatically?"
    grounded = build_critique_prompt("groundedness", q, SPAN)
    assert q in grounded and SPAN in grounded
    for criterion in ("relevance", "standalone"):
        prompt = build_critique_prompt(criterion, q, SPAN)
        assert q in prompt and SPAN not in prompt
    assert "Hugging Face ecosystem" in build_critique_prompt("relevance", q, SPAN)
    assert "context-independent" in build_critique_prompt("standalone", q, SPAN)
    for criterion in CRITERIA:
        prompt = build_critique_prompt(criterion, q, SPAN)
        # rationale first, then the rating
        assert prompt.index("Evaluation:") < prompt.index("Total rating:")

    parsed = parse_critique("Answer:::\nEvaluation: clear and answerable.\nTotal rating: 5")
    assert parsed.score == 5 and parsed.rationale == "clear and answerable."
    assert parse_critique("Evaluation: ok\nTotal rating: **4**").score == 4
    assert parse_critique("Evaluation: ok\nTotal rating: 4.5").score == 4.5
    assert parse_critique("Evaluation: x\nTotal rating: 3\nTotal rating: 2").score == 2
    assert parse_critique("Evaluation: no rating here").score is None
    assert parse_critique("Evaluation: x\nTotal rating: 7").score is None
    assert parse_critique("Evaluation: x\nTotal rating: 0").score is None
