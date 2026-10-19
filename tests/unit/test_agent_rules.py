# ruff: noqa: RUF001 (the normaliser and the Chinese rules are tested with full-width text)
"""Spec P3-02 AC1, AC2: the rules of the route and rewrite steps (no model, no state)."""

from __future__ import annotations

import pytest

from rag.agent.rules import (
    clarity,
    default_clarification,
    is_question_form,
    looks_like_follow_up,
    missing_tokens,
    multi_hop_hint,
    normalise,
    protected_tokens,
)


def test_normalise() -> None:
    assert normalise("  How   do I\npush\t a model？ ") == "How do I push a model?"
    assert normalise("ｔｒａｉｎｅｒ（ＧＰＵ）") == "trainer(GPU)"  # full-width letters, brackets
    assert normalise("") == ""


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How do I use push_to_hub with the Trainer?", ["push_to_hub"]),
        ("What does TrainingArguments(output_dir) do?", ["TrainingArguments", "output_dir"]),
        ("Is `device_map=auto` supported in 4.36.0?", ["device_map=auto", "4.36.0"]),
        (
            "Call IMTManagerAPI::TimeServer and handle MT_RET_ERR_PARTIAL",
            [
                "IMTManagerAPI::TimeServer",
                "MT_RET_ERR_PARTIAL",
            ],
        ),
        ("Why does torch.nn.Module fail on python 3.10?", ["torch.nn.Module", "3.10"]),
        ("Run it with --max_new_tokens 20 and v2.1", ["--max_new_tokens", "v2.1"]),
        (
            "Load google/flan-t5-base and edit config.json",
            [
                "google/flan-t5-base",
                "config.json",
            ],
        ),
        ('I get "CUDA out of memory" on a T5 model', ["CUDA out of memory", "T5"]),
        ("transformers里怎么用fromPretrained加载模型", ["fromPretrained"]),
        ("AutoModel.from_pretrained 怎么用", ["AutoModel.from_pretrained"]),
        ("error 0x80070005 after update", ["0x80070005"]),
    ],
)
def test_protected_tokens(question: str, expected: list[str]) -> None:
    assert protected_tokens(question) == expected


def test_plain_words_are_not_protected() -> None:
    for text in (
        "How do I push a model to the hub?",
        "yes/no or and/or, e.g. this one",
        "What is the best way to share a trained model?",
        "怎么把训练好的模型发布出去",
        "It was released in 2024 and works well",
    ):
        assert protected_tokens(text) == [], text


def test_missing_tokens_is_exact_and_case_sensitive() -> None:
    tokens = ["push_to_hub", "4.36.0", "TrainingArguments"]
    assert (
        missing_tokens(tokens, "Push models with push_to_hub in 4.36.0 and TrainingArguments") == []
    )
    assert missing_tokens(tokens, "push models to hub in 4.36 with trainingarguments") == tokens
    # The tokens may be spread over several queries (a split question).
    assert missing_tokens(tokens, "use push_to_hub", "in 4.36.0", "TrainingArguments") == []
    assert missing_tokens([], "anything") == []


def test_question_and_follow_up_forms() -> None:
    for text in (
        "How do I push a model?",
        "does it support 8-bit",
        "Is this thread safe",
        "push a model?",
        "怎么上传模型",
        "这个参数有什么用",
        "能否使用 LoRA",
    ):
        assert is_question_form(text), text
    for text in ("push_to_hub parameters", "Trainer installation guide", "模型上传教程", "hello"):
        assert not is_question_form(text), text

    for text in (
        "what about its tokenizer?",
        "and for diffusers?",
        "How do I do that?",
        "why?",
        "那 diffusers 呢",
        "它支持量化吗",
    ):
        assert looks_like_follow_up(text), text
    for text in (
        "How do I push a fine-tuned model to the Hugging Face Hub?",
        "Which scheduler does the Trainer use by default for fine-tuning?",
        "如何把微调好的模型上传到 Hugging Face Hub",
    ):
        assert not looks_like_follow_up(text), text


def test_clarity_and_multi_hop_signals() -> None:
    # vague: nothing to search for; clear: names something or has topic words; the session's
    # history makes anything clear (the follow-up is rewritten, not asked back).
    for text in ("help", "it doesn't work", "How do I fix this error?", "怎么办", "I get an error"):
        assert clarity(text, has_history=False) == "vague", text
    for text in (
        "How do I push a model to the hub?",
        "What does push_to_hub do",
        "fix error 0x80070005",
        "如何把微调好的模型上传到仓库",
    ):
        assert clarity(text, has_history=False) == "clear", text
    assert clarity("How do I save this model to the hub?", has_history=False) == "undecided"
    assert clarity("这个模型怎么上传到仓库", has_history=False) == "undecided"
    for text in ("help", "How do I fix this error?", "what about it?"):
        assert clarity(text, has_history=True) == "clear"

    assert multi_hop_hint("What is the difference between Trainer and SFTTrainer?") == "yes"
    assert multi_hop_hint("How do I load a dataset and then how do I tokenize it?") == "yes"
    assert multi_hop_hint("Trainer 和 SFTTrainer 有什么区别") == "yes"
    assert (
        multi_hop_hint("How do I fine-tune a model and push the checkpoint to the hub?") == "maybe"
    )
    assert multi_hop_hint("How do I push a model?") == "no"
    assert multi_hop_hint("save and load") == "no"  # too short to be two lookups


def test_default_clarification_is_in_the_question_language() -> None:
    assert "specific" in default_clarification("en")
    assert "具体" in default_clarification("zh")


def test_rules_leave_the_shipped_hf_questions_alone() -> None:
    """The questions of the HF evaluation, no-answer and synthetic sets (about 300 real ones) are
    searched as they stand: the rules never call one vague, undecided or a comparison. Guards
    against rules that would make the small model ask back or split ordinary questions."""
    import json
    from pathlib import Path

    from rag.config import REPO_ROOT

    folder = Path(REPO_ROOT) / "evals" / "datasets" / "hf_docs"
    questions: list[str] = []
    for name in (
        "huggingface_doc_qa_eval.jsonl",
        "no_answer.jsonl",
        "no_answer_extra.jsonl",
        "synthetic_dev.jsonl",
        "synthetic_holdout.jsonl",
    ):
        for line in (folder / name).read_text("utf-8").splitlines():
            if line.strip():
                questions.append(json.loads(line)["question"])
    assert len(questions) > 250
    flagged = [
        (clarity(q, has_history=False), multi_hop_hint(q), q)
        for q in questions
        if clarity(q, has_history=False) != "clear" or multi_hop_hint(q) == "yes"
    ]
    assert flagged == []
    maybe = [q for q in questions if multi_hop_hint(q) == "maybe"]
    assert len(maybe) <= len(questions) // 50  # at most 2 in 100 ask the small model about a split

    # What the guard stands for: names make a question clear even when it says "its".
    for text in (
        "How much money did Hugging Face raise in its Series D round?",
        "What does Decision Transformer condition its model on?",
        "What error do other users get when they visit the URL of a private repository?",
        "What parameter is used to ensure that elements in a row have the same height?",
    ):
        assert clarity(text, has_history=False) == "clear", text
    assert clarity("How do I fix this error?", has_history=False) == "vague"
