# ruff: noqa: RUF001 (full-width characters are the test data)
"""Spec P5-01 AC1: the cache key schema (design s12-1, s12-3): one test per field proves that
changing it changes the key, and what must not change it does not."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from rag.cache.keys import (
    CacheKeyFields,
    acl_hash,
    answer_config_version,
    cache_key,
    canonical_filters,
    canonical_question,
    make_fields,
    prompt_version,
)
from rag.config import AgentConfig, KnowledgeBaseConfig
from rag.testing import observe_settings


def _fields(**over: Any) -> CacheKeyFields:
    base: dict[str, Any] = {
        "kind": "ask",
        "kb": "hf_docs",
        "index_prefix": "",
        "data_generation": 17,
        "config_version": "cfg1",
        "answer_version": "ans1",
        "question": "how do i push a model",
        "filters": "{}",
        "acl": ("public",),
        "top_k": None,
        "language": "en",
    }
    base.update(over)
    return CacheKeyFields(**base)


# One changed value per field of the schema; the test below checks the list is the whole schema.
CHANGED: dict[str, Any] = {
    "kind": "search",
    "kb": "mt5",
    "index_prefix": "fake_",
    "data_generation": 18,
    "config_version": "cfg2",
    "answer_version": "ans2",
    "question": "how do i push a dataset",
    "filters": '{"doc_type":"md"}',
    "acl": ("public", "team-a"),
    "top_k": 5,
    "language": "zh",
}


def test_changed_table_covers_every_field_of_the_schema() -> None:
    assert set(CHANGED) == {f.name for f in dataclasses.fields(CacheKeyFields)}


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_every_field_changes_the_key(name: str) -> None:
    assert cache_key(_fields()) != cache_key(_fields(**{name: CHANGED[name]}))


def test_the_same_fields_give_the_same_key_and_the_key_names_kb_and_kind() -> None:
    a, b = cache_key(_fields()), cache_key(_fields())
    assert a == b
    assert a.startswith("hf_docs:ask:") and len(a.split(":")[-1]) == 64
    assert cache_key(_fields(index_prefix="fake_")).startswith("fake_hf_docs:ask:")


def test_acl_is_a_set_of_groups() -> None:
    one = make_fields("search", kb="k", acl_groups=["b", "a", "b"], question="q")
    two = make_fields("search", kb="k", acl_groups=("a", "b"), question="q")
    other = make_fields("search", kb="k", acl_groups=["a", "c"], question="q")
    assert cache_key(one) == cache_key(two) != cache_key(other)
    assert one.acl == ("a", "b")
    assert acl_hash(["b", "a"]) == acl_hash(("a", "b", "a")) != acl_hash(["a"])


def test_a_user_with_other_groups_does_not_share_an_entry() -> None:
    mine = make_fields("search", kb="k", acl_groups=["public"], question="q")
    theirs = make_fields("search", kb="k", acl_groups=["public", "hr"], question="q")
    assert cache_key(mine) != cache_key(theirs)


def test_filters_are_canonical() -> None:
    assert canonical_filters(None) == canonical_filters({}) == "{}"
    a = canonical_filters({"language": "en", "doc_type": ["md", "mdx", "md"]})
    b = canonical_filters({"doc_type": ["mdx", "md"], "language": "en"})
    assert a == b
    assert canonical_filters({"language": "en"}) != canonical_filters({"language": "zh"})
    assert canonical_filters({"source_rel_prefix": "docs/a"}) != canonical_filters(
        {"source_rel_prefix": "docs/b"}
    )


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("How do I push a model?", "how do i push a model"),
        ("  how   do I\tpush a  model ？ ", "how do i push a model"),
        ("HOW DO I PUSH A MODEL!", "how do i push a model"),
        ("How do I push a model.", "how do i push a model"),
        ("Ｈｏｗ　ｄｏ　I push a model", "how do i push a model"),  # full-width letters, NFKC
        ("how do i​ push a model", "how do i push a model"),  # zero-width space
        ("怎么上传模型？", "怎么上传模型"),
    ],
)
def test_canonical_question_removes_only_case_space_and_closing_punctuation(
    raw: str, canonical: str
) -> None:
    assert canonical_question(raw) == canonical


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("What does push_to_hub do?", "What does push_to_hub_fastai do?"),
        ("What does LoRA do?", "What does lora do?"),  # an identifier keeps its case
        ("How to install transformers 4.36.0", "How to install transformers 4.36.1"),
        ("Which models support fp16?", "Which models do not support fp16?"),
        ("How do I push a model", "How do I pull a model"),
        ("What is Trainer.train", "What is trainer.train"),
        ("Python 3.9 or 3.10?", "Python 3.9 or 3.11?"),
        ("pushing models", "push models"),  # no stemming
    ],
)
def test_canonical_question_never_merges_different_questions(a: str, b: str) -> None:
    assert canonical_question(a) != canonical_question(b)


def test_make_fields_canonicalises_and_names_the_language() -> None:
    f = make_fields(
        "search",
        kb="hf_docs",
        question="  How do I push a model? ",
        filters={"language": "en"},
        acl_groups=["public"],
        top_k=8,
        data_generation=3,
        config_version="c",
        language_setting="auto",
    )
    assert f.question == "how do i push a model"
    assert f.language == "en"
    assert f.filters == '{"language":"en"}'
    zh = make_fields("search", kb="k", question="怎么上传模型", language_setting="auto")
    assert zh.language == "zh"
    forced = make_fields("search", kb="k", question="how do i push", language_setting="zh")
    assert forced.language == "zh"


def test_answer_version_follows_the_kb_config_the_models_and_the_agent_settings() -> None:
    cfg = KnowledgeBaseConfig(kb="hf_docs")
    s = observe_settings()
    base = answer_config_version(cfg, s)
    assert base == answer_config_version(cfg, s) and len(base) == 16
    assert base != answer_config_version(
        cfg.model_copy(update={"agent": AgentConfig(judge=True)}), s
    )
    assert base != answer_config_version(cfg, observe_settings(chat_model="other-model"))
    assert base != answer_config_version(cfg, observe_settings(agent_answer_max_tokens=900))
    assert base != answer_config_version(cfg, observe_settings(agent_rewrite_profile="online"))
    # What does not change an answer does not change the version.
    assert base == answer_config_version(cfg.model_copy(update={"description": "x"}), s)
    assert base == answer_config_version(cfg, observe_settings(agent_resume_ttl_s=5))


def test_prompt_version_is_stable_and_follows_the_prompt_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = prompt_version()
    assert before == prompt_version()
    from rag.answer import prompts

    monkeypatch.setitem(prompts.SYSTEM_PROMPTS, "en", prompts.SYSTEM_PROMPTS["en"] + " Be brief.")
    assert prompt_version() != before
