"""Spec P1-05 AC6-AC8: lexical overlap, rejection reasons and the dev/holdout split."""

from __future__ import annotations

from collections import Counter

import pytest

from rag.eval.synthetic.filters import (
    REDACTED,
    FilterConfig,
    lexical_overlap,
    redact_secrets,
    rejection_reasons,
)
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


def test_redact_secrets() -> None:
    token = "hf_" + "Ab3" * 12
    text, n = redact_secrets(f"Run `HF_TOKEN={token} python app.py`, key sk-{'x1' * 12}.")
    assert n == 2 and token not in text and text.count(REDACTED) == 2
    assert redact_secrets("use hf_hub_download and sk-learn") == (
        "use hf_hub_download and sk-learn",
        0,
    )


def test_near_duplicates_stay_in_one_set() -> None:
    from rag.eval.synthetic.split import near_duplicate_groups

    items = [
        (
            "q1",
            "Which parameter in timm.create_model specifies the number of classes?",
            "num_classes",
        ),
        ("q2", "Which parameter in timm.create_model sets the number of classes?", "`num_classes`"),
        ("q3", "What PyTorch function is used to get the top 5 predictions?", "torch.topk"),
        (
            "q4",
            "What PyTorch function returns the top 5 probabilities and class IDs?",
            "torch.topk",
        ),
        ("q5", "What does the label O mean in token classification?", "no entity"),
        ("q6", "Which scheduler is recommended for Stable Diffusion 2?", "DPMSolver"),
    ]
    groups = near_duplicate_groups(items)
    assert groups["q2"] == "q1" and groups["q4"] == "q3"
    assert len({groups[q] for q in ("q1", "q3", "q5", "q6")}) == 4
    # q3/q4 share only 45% of their words: grouped because the answers are equal
    assert near_duplicate_groups([items[2], (*items[3][:2], "other")])["q4"] == "q4"

    pairs = [(f"a{i:03d}", "A") for i in range(150)] + [(f"b{i:03d}", "B") for i in range(90)]
    dupes = {f"a{i:03d}": "a000" for i in range(0, 20, 2)}  # one group of 10 questions
    for seed in range(5):
        dev, holdout = split_dev_holdout(pairs, seed=seed, min_each=100, groups=dupes)
        group = set(dupes)
        assert group <= set(dev) or group <= set(holdout)
        assert len(dev) + len(holdout) == 240 and abs(len(holdout) - 120) < 10
    # without groups the split is the same as before
    assert split_dev_holdout(pairs, seed=3, min_each=100, groups={}) == split_dev_holdout(
        pairs, seed=3, min_each=100
    )


def test_answering_pages_by_exact_text() -> None:
    from rag.eval.synthetic.labels import PageIndex
    from rag.eval.synthetic.sampler import Page

    card = (
        "Use timm.create_model('{m}', pretrained=True, num_classes=NUMBER_OF_CLASSES) to finetune."
    )
    pages = [
        Page(
            doc_id=f"hf_docs:{m}.md",
            source_rel=f"{m}.md",
            doc_type="markdown",
            language="en",
            text=f"# {m}\n" + card.format(m=m),
        )
        for m in ("pnasnet", "xception65", "dla102")
    ]
    pages.append(
        Page(
            doc_id="hf_docs:other.md",
            source_rel="other.md",
            doc_type="markdown",
            language="en",
            text="num_classes is also a field of other configs.",
        )
    )
    index = PageIndex(pages)
    span = pages[0].text
    # a generic question: every model card answers it, the unrelated page does not
    assert index.answering_pages(
        question="Which timm.create_model parameter sets the number of classes to finetune?",
        answer="`num_classes`.",
        span=span,
        source="hf_docs:pnasnet.md",
    ) == ["hf_docs:dla102.md", "hf_docs:xception65.md"]
    # a page-specific answer stays on its page
    assert (
        index.answering_pages(
            question="What model name loads the pretrained PNASNet?",
            answer="pnasnet",
            span=span,
            source="hf_docs:pnasnet.md",
        )
        == []
    )
    # an answer that is not in the span (a paraphrase) adds nothing
    assert (
        index.answering_pages(
            question="Which argument sets the class count?",
            answer="the class count argument",
            span=span,
            source="hf_docs:pnasnet.md",
        )
        == []
    )
