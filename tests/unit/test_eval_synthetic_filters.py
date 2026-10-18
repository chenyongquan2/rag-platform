"""Spec P1-05 AC6-AC8: lexical overlap, rejection reasons and the dev/holdout split."""

from __future__ import annotations

from collections import Counter

import pytest

from rag.eval.synthetic.filters import FilterConfig, lexical_overlap, rejection_reasons
from rag.eval.synthetic.split import split_dev_holdout

SPAN = (
    "To share a model, set push_to_hub=True in TrainingArguments; the Trainer then uploads "
    "the checkpoint to your namespace on the Hub at the end of training."
)
GOOD = {"groundedness": 5.0, "relevance": 4.0, "standalone": 5.0}


def test_lexical_overlap() -> None:
    copied = lexical_overlap(
        "Does the Trainer upload the checkpoint to your namespace on the Hub?", SPAN
    )
    # "the checkpoint to your namespace on the hub" = 8 words in a row
    assert copied.longest_run == 8
    assert copied.content_recall >= 0.8
    fresh = lexical_overlap("Which option publishes my fine-tuned weights online?", SPAN)
    assert fresh.longest_run <= 1
    assert fresh.content_recall < 0.3
    # case and punctuation do not matter; code names count as words
    assert lexical_overlap("PUSH_TO_HUB=True?", SPAN).longest_run == 2
    empty = lexical_overlap("", SPAN)
    assert empty.longest_run == 0 and empty.content_recall == 0.0


def test_rejection_reasons() -> None:
    config = FilterConfig()
    assert (config.min_groundedness, config.min_relevance, config.min_standalone) == (4, 4, 4)
    q, a = "Which option publishes my fine-tuned weights online?", "push_to_hub=True"
    assert rejection_reasons(question=q, answer=a, span=SPAN, scores=GOOD, config=config) == []

    low = {"groundedness": 3.0, "relevance": 2.0, "standalone": 5.0}
    assert rejection_reasons(question=q, answer=a, span=SPAN, scores=low, config=config) == [
        "groundedness_low",
        "relevance_low",
    ]
    missing = {"groundedness": 5.0, "relevance": None}
    assert rejection_reasons(question=q, answer=a, span=SPAN, scores=missing, config=config) == [
        "relevance_missing",
        "standalone_missing",
    ]
    copied = "Does the Trainer upload the checkpoint to your namespace on the Hub?"
    assert rejection_reasons(question=copied, answer=a, span=SPAN, scores=GOOD, config=config) == [
        "copied_wording"
    ]
    assert rejection_reasons(
        question=q, answer="x" * 301, span=SPAN, scores=GOOD, config=config
    ) == ["answer_too_long"]
    assert rejection_reasons(question=None, answer=None, span=SPAN, scores={}, config=config) == [
        "generation_failed"
    ]
    # the copy limit is a setting
    loose = FilterConfig(max_copy_run=10)
    assert rejection_reasons(question=copied, answer=a, span=SPAN, scores=GOOD, config=loose) == []


def _items(n_a: int, n_b: int) -> list[tuple[str, str]]:
    return [(f"a{i:03d}", "A") for i in range(n_a)] + [(f"b{i:03d}", "B") for i in range(n_b)]


def test_split_dev_holdout() -> None:
    items = _items(150, 90)
    dev, holdout = split_dev_holdout(items, seed=3, holdout_share=0.5, min_each=100)
    assert set(dev).isdisjoint(holdout)
    assert set(dev) | set(holdout) == {i for i, _ in items}
    assert len(holdout) == 120 and len(dev) == 120
    strata = dict(items)
    assert Counter(strata[i] for i in holdout) == {"A": 75, "B": 45}
    again = split_dev_holdout(list(reversed(items)), seed=3, holdout_share=0.5, min_each=100)
    assert again == (dev, holdout)  # input order does not matter, the seed does
    assert split_dev_holdout(items, seed=4, holdout_share=0.5, min_each=100) != (dev, holdout)
    with pytest.raises(ValueError, match="holdout"):
        split_dev_holdout(_items(100, 50), seed=0, holdout_share=0.5, min_each=100)
