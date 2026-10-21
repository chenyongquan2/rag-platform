"""Spec P5-01 AC8: the entity guard of the semantic cache (design s12-3): a stored question may
stand in for a new one only when their identifiers, names, numbers and negations are equal."""

from __future__ import annotations

import pytest

from rag.cache.semantic import (
    content_words,
    entity_guard,
    entity_signature,
    lexical_guard,
    question_guard,
)


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        (
            "How do I push a model with push_to_hub?",
            "What is the way to push a model with push_to_hub",
        ),
        ("what does Trainer.train return", "What does Trainer.train return?"),
        ("which models support fp16", "Which models support fp16 ?"),
        ("How do I install transformers 4.36.0", "how to install transformers 4.36.0 ?"),
        ("What is the top 5 result", "what's the top 5 result"),
        ("怎么用 push_to_hub 上传模型", "如何通过 push_to_hub 上传模型"),
        ("how do I upload a model", "how can I upload a model"),
        ("What does LoRA do?", "what does lora do"),  # an identifier is compared without case
        ("what is longt5 local attention", "What is LongT5 Local Attention"),
    ],
)
def test_questions_with_the_same_entities_pass(new: str, cached: str) -> None:
    assert entity_guard(new, cached).ok


@pytest.mark.parametrize(
    ("new", "cached", "reason"),
    [
        ("What does push_to_hub do?", "What does push_to_hub_fastai do?", "identifiers"),
        ("transformers 4.36.0 install", "transformers 4.36.1 install", "identifiers"),
        ("How do I use --lr", "How do I use --lr-scheduler", "identifiers"),
        ("How do I use bert-base-uncased", "How do I use bert-large-uncased", "identifiers"),
        ("What is the top 5 result", "What is the top 10 result", "numbers"),
        ("Give me three examples", "Give me two examples", "numbers"),
        ("三种方法", "两种方法", "numbers"),
        ("Which models support fp16", "Which models do not support fp16", "negation"),
        ("which options can I use", "which options can't I use", "negation"),
        ("哪些模型支持半精度", "哪些模型不支持半精度", "negation"),
        ("How do I use Gradio Spaces", "How do I use Streamlit Spaces", "names"),
        ("Gradio install guide", "Streamlit install guide", "names"),
        ("What is Inference Endpoints", "What is Inference Providers", "names"),
    ],
)
def test_questions_that_differ_in_an_entity_are_rejected(
    new: str, cached: str, reason: str
) -> None:
    verdict = entity_guard(new, cached)
    assert not verdict.ok and verdict.reason == reason, verdict


def test_the_guard_is_symmetric() -> None:
    pairs = [
        ("What does push_to_hub do?", "What does push_to_hub_fastai do?"),
        ("top 5", "top 10"),
        ("support fp16", "do not support fp16"),
    ]
    for a, b in pairs:
        assert entity_guard(a, b).ok == entity_guard(b, a).ok


def test_a_question_without_entities_passes_but_is_marked_so() -> None:
    verdict = entity_guard("how do I upload a model", "how can I upload a model")
    assert verdict.ok and verdict.entities is False
    with_entities = entity_guard("how do I use push_to_hub", "how can I use push_to_hub")
    assert with_entities.ok and with_entities.entities is True


def test_the_signature_names_what_it_compares() -> None:
    sig = entity_signature(
        "Which models in Hugging Face do not support push_to_hub 4.36.0 on 5 GPUs?"
    )
    assert "push_to_hub" in sig.tokens and "4.36.0" in sig.tokens
    assert sig.numbers == ("5",) or "5" in sig.numbers
    assert "not" in sig.negations
    assert {"Hugging", "Face"} <= set(sig.names)


def test_a_sentence_initial_question_word_is_not_a_name() -> None:
    assert entity_signature("How do I upload a model").names == frozenset()
    assert entity_signature("What is Spaces").names == frozenset({"Spaces"})
    assert entity_signature("Gradio how to install").names == frozenset({"Gradio"})


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("How do I push a model to the hub", "How do I pull a model to the hub"),
        ("what is the purpose of push_to_hub", "what does push_to_hub return"),
        ("how many classes are in the dataset", "how many classes are in the datasets"),
        ("which models support fp16", "which models support fp16 training"),
    ],
)
def test_the_lexical_guard_rejects_a_change_of_one_or_two_words(new: str, cached: str) -> None:
    assert entity_guard(new, cached).ok  # the entities do not see it
    assert not lexical_guard(new, cached).ok
    assert not question_guard(new, cached).ok
    assert question_guard(new, cached, lexical=False).ok


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("how do I upload a trained model", "what do I call to send my trained model online"),
        ("怎么上传模型", "如何把模型上传到远程仓库中去"),  # many other characters
        ("how do I push a model", "how do I push a model"),
    ],
)
def test_the_lexical_guard_lets_other_wordings_and_the_same_words_through(
    new: str, cached: str
) -> None:
    assert lexical_guard(new, cached).ok


def test_content_words_skip_stop_words_but_keep_question_words_and_modals() -> None:
    assert content_words("Can I push a model?") == frozenset({"can", "push", "model"})
    assert content_words("How do I push a model?") == frozenset({"how", "push", "model"})


def test_a_chinese_question_is_measured_by_its_characters_and_latin_words() -> None:
    assert content_words("怎么上传 Hub 模型?") == frozenset(
        {"如", "何", "上", "传", "hub", "模", "型"}  # 怎么 and 如何 are one word
    )


@pytest.mark.parametrize(
    ("new", "cached", "reason"),
    [
        # the identifiers are the same but in the other order
        ("difference between BERT and GPT-2", "difference between GPT-2 and BERT", "order"),
        ("convert evaluate_loop into AutoModel", "convert AutoModel into evaluate_loop", "order"),
        # the same plain words in the other order
        (
            "is it safer to use safetensors or pickle",
            "is it safer to use pickle or safetensors",
            "order",
        ),
        # another question word or another modal is one word of difference
        ("why is the loss nan", "how is the loss nan", "lexical"),
        ("when can I use fp16 training", "where can I use fp16 training", "lexical"),
        ("which options can I set", "which options should I set", "lexical"),
        ("how do I use this", "what do I use this", "lexical"),
        # a Chinese word and its opposite
        ("如何启用混合精度训练", "如何禁用混合精度训练", "lexical"),
        ("怎么把模型上传到 Hub", "怎么把模型下载到 Hub", "lexical"),
        ("先加载模型再训练", "先训练再加载模型", "order"),
    ],
)
def test_review_round_one_blind_spots_are_rejected(new: str, cached: str, reason: str) -> None:
    verdict = question_guard(new, cached)
    assert not verdict.ok and verdict.reason == reason, verdict


def test_the_identifier_order_is_part_of_the_entity_guard() -> None:
    verdict = entity_guard("BERT versus GPT-2 speed", "GPT-2 versus BERT speed")
    assert not verdict.ok and verdict.reason == "order"
    assert entity_guard("BERT and GPT-2", "bert and gpt-2").ok  # case is still no difference
