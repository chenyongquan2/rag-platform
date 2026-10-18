"""Spec P1-05 AC1-AC3: spans sampled from the raw cleaned page text, seeded and stratified."""

from __future__ import annotations

from collections import Counter

import pytest

from rag.eval.synthetic.sampler import (
    Page,
    SamplerConfig,
    allocate,
    sample_spans,
    span_problem,
    stratum_of,
)

PROSE = (
    "The tokenizer splits every input sentence into subword units before the model sees it, "
    "and the vocabulary decides how many pieces a rare word becomes in practice.\n"
)


def _page(i: int, repo: str = "huggingface/transformers", doc_type: str = "markdown", **kw) -> Page:
    text = kw.pop("text", None) or "".join(f"Line {i}.{k}: {PROSE}" for k in range(40))
    return Page(
        doc_id=f"hf_docs:{repo}/blob/main/docs/p{i}.md",
        source_rel=f"{repo}/blob/main/docs/p{i}.md",
        doc_type=doc_type,
        language=kw.pop("language", "en"),
        text=text,
    )


def test_sample_spans_raw_text_seeded() -> None:
    pages = [_page(i) for i in range(30)]
    config = SamplerConfig(n=10, seed=7, span_chars=600, min_chars=200)
    first = sample_spans(pages, config)
    again = sample_spans(pages, config)
    assert [s.model_dump() for s in first.spans] == [s.model_dump() for s in again.spans]
    other = sample_spans(pages, config.model_copy(update={"seed": 8}))
    assert [s.span_id for s in other.spans] != [s.span_id for s in first.spans]

    by_id = {p.doc_id: p for p in pages}
    assert len(first.spans) == 10
    for span in first.spans:
        page = by_id[span.doc_id]
        # the span is a slice of the page's cleaned text, starting at a line start
        assert page.text[span.start : span.end] == span.text
        assert span.start == 0 or page.text[span.start - 1] == "\n"
        assert 600 <= len(span.text) <= 600 + config.max_line_extension
        assert span.stratum == stratum_of(page)
    # one span per page
    assert len({s.doc_id for s in first.spans}) == 10

    # a page shorter than span_chars gives the whole page
    short = _page(99, text=PROSE * 3)
    result = sample_spans([short], SamplerConfig(n=1, span_chars=5000, min_chars=100))
    assert result.spans[0].text == short.text and result.spans[0].start == 0


def test_allocate_proportional() -> None:
    sizes = {"a": 50, "b": 30, "c": 20}
    assert allocate(sizes, 10) == {"a": 5, "b": 3, "c": 2}
    got = allocate({"a": 2, "b": 1, "c": 1}, 3)
    assert sum(got.values()) == 3 and got["a"] >= 1
    assert allocate({"x": 1, "y": 99}, 50)["x"] <= 1
    assert sum(allocate({"x": 7, "y": 13, "z": 5}, 11).values()) == 11
    with pytest.raises(ValueError):
        allocate({"a": 2}, 3)


def test_sample_spans_stratified_and_excluded() -> None:
    pages = (
        [_page(i, "huggingface/transformers") for i in range(60)]
        + [_page(100 + i, "gradio-app/gradio", "mdx") for i in range(30)]
        + [_page(200 + i, "huggingface/blog", "notebook") for i in range(10)]
        + [_page(300 + i, "gradio-app/gradio", language="zh") for i in range(10)]
    )
    excluded = pages[0].doc_id
    config = SamplerConfig(n=20, seed=1, span_chars=500, min_chars=200, exclude_pages=[excluded])
    result = sample_spans(pages, config)
    strata = Counter(s.stratum for s in result.spans)
    assert strata == {
        "huggingface/transformers:markdown": 12,
        "gradio-app/gradio:mdx": 6,
        "huggingface/blog:notebook": 2,
    }
    assert excluded not in {s.doc_id for s in result.spans}
    zh = {p.doc_id for p in pages if p.language == "zh"}
    assert zh.isdisjoint(s.doc_id for s in result.spans)
    assert result.pages_excluded == {"excluded": 1, "language": 10}
    assert result.allocation["huggingface/transformers:markdown"] == 12
    # the order is shuffled, so the first spans of a pilot cover several strata
    assert len({s.stratum for s in result.spans[:8]}) > 1


def test_span_problem_reasons() -> None:
    config = SamplerConfig(min_chars=200)
    assert span_problem(PROSE * 3, config) is None
    assert span_problem("Short page.", config) == "tiny"
    code = "```python\n" + "x = model(**inputs)\nprint(x.shape)\n" * 20 + "```\n"
    assert span_problem(code, config) == "code_only"
    table = "| a | b |\n|---|---|\n" + "| 1 | 2 |\n" * 40
    assert span_problem(table, config) == "code_only"
    licence = (
        "Licensed under the Apache License, Version 2.0 (the License); you may not use this file "
        "except in compliance with the License. " + PROSE * 2
    )
    assert span_problem(licence, config) == "boilerplate"

    # a page whose spans are all bad is skipped and counted; the stratum then falls short
    pages = [_page(0, text=code * 3), *[_page(i) for i in range(1, 6)]]
    result = sample_spans(pages, SamplerConfig(n=6, seed=0, span_chars=300, min_chars=200))
    assert len(result.spans) == 5
    assert pages[0].doc_id not in {s.doc_id for s in result.spans}
    assert result.pages_skipped == {"code_only": 1}
    assert result.shortfall == {"huggingface/transformers:markdown": 1}
