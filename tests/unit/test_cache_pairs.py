"""Spec P5-01 AC10 and AC8: the cache pair generator and the committed test set."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rag.cache import pairs as cp
from rag.cache.cli import build_pair_sources
from rag.cache.keys import canonical_question
from rag.cache.semantic import entity_guard
from rag.config import repo_path

DATASET = repo_path("evals/datasets/hf_docs/cache_pairs.jsonl")


# ---- the mutations ---------------------------------------------------------------------------


def test_identifier_swap_replaces_one_identifier_with_one_of_the_pool() -> None:
    q = "What does push_to_hub do with the repo?"
    out = cp.identifier_swap(q, ["create_repo", "upload_folder", "push_to_hub"], "s")
    assert out in {
        "What does create_repo do with the repo?",
        "What does upload_folder do with the repo?",
    }
    assert cp.identifier_swap("What is the purpose of the model?", ["x_y"], "s") is None


def test_identifier_swap_keeps_the_kind_of_the_identifier() -> None:
    out = cp.identifier_swap("What does AutoModel return?", ["push_to_hub", "AutoTokenizer"], "s")
    assert out == "What does AutoTokenizer return?"


def test_identifier_case_changes_one_capital_letter() -> None:
    assert cp.identifier_case("What does AutoModel return?", "s") == "What does autoModel return?"
    assert cp.identifier_case("What does push_to_hub do?", "s") == "What does Push_to_hub do?"
    assert cp.identifier_case("What is the best model?", "s") is None


def test_version_and_number_changes() -> None:
    assert cp.version_bump("Since transformers 4.36.0 what changed?", "s") == (
        "Since transformers 4.36.1 what changed?"
    )
    assert cp.number_swap("How many classes does the dataset have in 5 folds?", "s") == (
        "How many classes does the dataset have in 6 folds?"
    )
    assert cp.number_swap("What are the two techniques?", "s") == "What are the three techniques?"
    assert cp.number_swap("What is the purpose of this?", "s") is None


def test_number_swap_leaves_the_numbers_of_identifiers_alone() -> None:
    assert cp.number_swap("What does T5 do?", "s") is None
    assert cp.number_swap("What does gpt2 do?", "s") is None


def test_negation_flip_adds_or_removes_a_negation() -> None:
    chosen = next(s for s in map(str, range(40)) if cp._h(s, "negation") % cp._NEGATION_SHARE < 2)
    other = next(s for s in map(str, range(40)) if cp._h(s, "negation") % cp._NEGATION_SHARE >= 2)
    assert cp.negation_flip("What is the purpose of X?", chosen) == "What is not the purpose of X?"
    assert cp.negation_flip("Which models can be used?", chosen) == "Which models cannot be used?"
    assert cp.negation_flip("Which models should be set?", chosen) == (
        "Which models should not be set?"
    )
    # "does the dataset have": the subject follows the auxiliary, `not` goes before the verb
    assert cp.negation_flip("How many splits does the dataset have?", chosen) == (
        "How many splits does the dataset not have?"
    )
    assert cp.negation_flip("What can you use Gradio for?", chosen) == (
        "What can you not use Gradio for?"
    )
    assert (
        cp.negation_flip("What does Trainer.train do?", chosen) is None
    )  # no verb to put it in front of
    # removing a negation is always done (the flip of a question that has one)
    assert cp.negation_flip("Which models do not support fp16?", other) == (
        "Which models do support fp16?"
    )
    assert cp.negation_flip("What is the purpose of X?", other) is None  # not chosen for this one
    assert cp.negation_flip("Hugging Face Hub", chosen) is None


def test_antonym_swap_changes_the_first_opposite_pair_word() -> None:
    assert cp.antonym_swap("How do I push a model to the hub?", "s") == (
        "How do I pull a model to the hub?"
    )
    assert cp.antonym_swap("Upload a file", "s") == "Download a file"
    assert cp.antonym_swap("What is a model?", "s") is None


def test_aspect_change_keeps_the_identifier_and_changes_what_is_asked() -> None:
    out = cp.aspect_change("What is the purpose of push_to_hub?", "seed")
    assert out is not None and "push_to_hub" in out
    assert out.rstrip("?.") != "What is the purpose of push_to_hub"
    assert cp.aspect_change("What is the purpose of the model?", "seed") is None


def test_positive_variants_are_merged_by_the_key_normaliser() -> None:
    q = "What is the purpose of the model?"
    for name, make in cp.POSITIVE_VARIANTS.items():
        variant = make(q)
        assert variant is not None and variant != q, name
        assert canonical_question(variant) == canonical_question(q), name
    assert cp.case_variant("What does push_to_hub do?") is None


# ---- building --------------------------------------------------------------------------------


SMALL = {
    "dev": [
        ("q:1", "What does push_to_hub do when the repo has 3 files?"),
        ("q:2", "How do I upload a model with AutoModel?"),
        ("q:3", "What is the purpose of create_repo?"),
        ("q:4", "Which models are not supported by the Trainer?"),
    ]
    * 1,
    "holdout": [("h:1", "What does upload_folder do?"), ("h:2", "How can I enable fp16?")],
}


def test_build_is_deterministic_and_pair_ids_are_unique() -> None:
    first = cp.build_pairs(SMALL)
    second = cp.build_pairs(SMALL)
    assert first == second
    ids = [p.pair_id for p in first]
    assert len(ids) == len(set(ids))


def test_a_question_gets_at_most_three_text_counter_examples() -> None:
    pairs = cp.build_pairs(SMALL)
    per_question = Counter(p.source_qid for p in cp.text_pairs(pairs))
    assert per_question and max(per_question.values()) <= cp.PER_QUESTION


def test_holdout_mutations_never_use_dev_identifiers() -> None:
    """The holdout part must not depend on the dev part: identifiers come from its own split."""
    dev_ids = {t for _, q in SMALL["dev"] for t in cp._swappable(q)}
    assert dev_ids
    for p in cp.build_pairs(SMALL):
        if p.split == "holdout":
            assert not any(t in p.b for t in dev_ids), p.pair_id


def test_metadata_pairs_have_the_same_question_and_another_context() -> None:
    pairs = [p for p in cp.build_pairs(SMALL) if p.b_ctx]
    assert {p.category for p in pairs} == set(cp.CONTEXT_CATEGORIES)
    for p in pairs:
        assert p.a == p.b and p.kind == "counter" and not p.expect_share


# ---- the committed set -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed() -> list[cp.Pair]:
    assert DATASET.exists(), "run: uv run python -m rag.cache pairs build"
    return cp.read_pairs(DATASET)


def test_the_committed_set_has_enough_pairs(committed: list[cp.Pair]) -> None:
    counters = [p for p in committed if p.kind == "counter"]
    positives = [p for p in committed if p.kind == "positive"]
    assert len(counters) >= 300 and len(positives) >= 100
    by_split = Counter(p.split for p in counters)
    assert by_split["dev"] >= 150 and by_split["holdout"] >= 100
    categories = {p.category for p in counters}
    assert {
        "identifier_swap",
        "identifier_case",
        "version_bump",
        "number_swap",
        "negation_flip",
        "antonym_swap",
        "aspect_change",
        "other_kb",
        "other_acl",
        "other_filters",
        "other_top_k",
        "other_language",
    } <= categories


def test_the_committed_set_is_what_the_generator_makes(committed: list[cp.Pair]) -> None:
    """A change of a rule or of a question set must come with a new file."""
    assert committed == cp.build_pairs(build_pair_sources())
    raw = DATASET.read_bytes()
    assert b"\r\n" not in raw  # one line ending on every checkout


def test_every_counter_example_differs_from_its_source(committed: list[cp.Pair]) -> None:
    for p in committed:
        if p.kind == "counter":
            assert p.a != p.b or p.b_ctx, p.pair_id
        else:
            assert p.expect_share


def test_the_exact_cache_never_shares_a_counter_example_and_always_shares_a_positive(
    committed: list[cp.Pair],
) -> None:
    """The argument: a key is a hash of canonical fields; a counter-example differs in a field
    after canonicalisation (a word, an identifier with its case, a number, the context), so the
    hashes differ; the test checks it on every pair."""
    result = cp.evaluate_l1(committed)
    assert result["false_hits"] == []
    assert result["positives_not_shared"] == []
    assert result["counter_pairs"] >= 300


def test_the_entity_guard_rejects_the_identifier_number_and_negation_counter_examples(
    committed: list[cp.Pair],
) -> None:
    """Without any distance: whatever the embedding says, the guard alone must stop the classes
    it is made for. The classes it cannot see (aspect, antonym) are measured in the report, and
    a capital letter in an identifier is no difference to it (the pairs of that class are the
    L1 policy: identifiers are case sensitive there)."""
    by_category: dict[str, list[bool]] = {}
    for p in cp.text_pairs(committed):
        by_category.setdefault(p.category, []).append(entity_guard(p.b, p.a).ok)
    for category in ("identifier_swap", "version_bump", "number_swap"):
        assert not any(by_category[category]), category
    assert sum(by_category["negation_flip"]) / len(by_category["negation_flip"]) < 0.1
    assert all(by_category["identifier_case"])


def test_the_two_questions_of_a_translation_pair_are_in_different_languages(
    committed: list[cp.Pair],
) -> None:
    from rag.answer.prompts import answer_language

    translations = [p for p in committed if p.category == "other_language"]
    assert len(translations) >= 20
    for p in translations:
        assert answer_language(p.a, "auto") != answer_language(p.b, "auto"), p.pair_id


def test_the_file_reads_back_with_chinese_text_intact(
    committed: list[cp.Pair], tmp_path: Path
) -> None:
    out = tmp_path / "pairs.jsonl"
    cp.write_pairs(committed, out)
    assert out.read_bytes() == DATASET.read_bytes()
