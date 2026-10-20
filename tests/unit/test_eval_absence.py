"""Spec P3-03 AC36 (review round 1): the absence checks catch the spellings a plain keyword
search missed (the two wrong no-answer questions the reviewer found)."""

from __future__ import annotations

from rag.eval.absence import concept_pages, concept_regex, inflected, pages_matching, phrase_regex

CORPUS = {
    "p:security": "## Security\nWe are storing logs for 30 days. Inference Endpoints are private.",
    "p:hardware": (
        "| Hardware | vCPU | RAM | Hourly Price |\n| CPU Upgrade | 8 vCPU | 32 GB | $0.03 |"
    ),
    "p:other": "The tokenizer splits text into word pieces. Nothing about money here.",
}


def test_the_miss_the_reviewer_found_is_caught_by_inflection() -> None:
    # "stored for|kept for|retention" missed "storing logs for 30 days".
    assert pages_matching(r"stored for|kept for|retention", CORPUS) == []
    assert pages_matching(concept_regex("store logs", "days"), CORPUS) == ["p:security"]
    assert pages_matching(concept_regex("logs", "30 days"), CORPUS) == ["p:security"]


def test_hyphen_space_and_underscore_variants_and_either_order_match() -> None:
    # "cpu-upgrade" (hyphen) missed the table row "CPU Upgrade ... $0.03".
    assert pages_matching(r"cpu-upgrade.{0,120}\$ ?\d", CORPUS) == []
    for spelling in ("cpu-upgrade", "cpu upgrade", "cpu_upgrade"):
        assert pages_matching(concept_regex(spelling, "$"), CORPUS) == ["p:hardware"]
    assert pages_matching(concept_regex("price", "cpu upgrade"), CORPUS) == ["p:hardware"]


def test_unrelated_pages_and_far_apart_words_do_not_match() -> None:
    assert pages_matching(concept_regex("logs", "days"), {"p:other": CORPUS["p:other"]}) == []
    far = {"x": "logs " + "word " * 100 + "days"}
    assert pages_matching(concept_regex("logs", "days"), far) == []
    assert pages_matching(concept_regex("logs", "days", window=1000), far) == ["x"]


def test_inflected_stems() -> None:
    assert inflected("storing") == inflected("stored")
    assert phrase_regex("cpu upgrade") == phrase_regex("cpu-upgrade") == phrase_regex("cpu_upgrade")


def test_concept_pages_is_the_same_search_with_a_cheap_prefilter() -> None:
    assert concept_pages("store logs", "days", CORPUS) == ["p:security"]
    assert concept_pages("cpu-upgrade", "$", CORPUS) == ["p:hardware"]
    assert concept_pages("logs", "days", {"x": "logs " + "word " * 100 + "days"}) == []
