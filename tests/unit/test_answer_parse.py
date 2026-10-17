# ruff: noqa: RUF001 (Chinese sample text)
"""Spec P1-02: parsing `[n]` answers and citation check level 1 (design s11-1, s11-2)."""

from __future__ import annotations

import pytest

from rag.answer.context import build_context
from rag.answer.parse import check_citations, parse_answer
from rag.answer.prompts import NOT_FOUND
from rag.types import Chunk


def context(n: int = 3):  # type: ignore[no-untyped-def]
    chunks = [
        Chunk(chunk_id=f"doc#{i}", doc_id="doc", kb="hf_docs", text=f"Fragment {i}.")
        for i in range(n)
    ]
    return build_context([chunks], budget_tokens=10_000)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("LongT5 uses local attention [1].", [("LongT5 uses local attention.", [1])]),
        ("A is B [1]. C is D [2].", [("A is B.", [1]), ("C is D.", [2])]),
        ("A is B 【2】。C 是 D［3］。", [("A is B。", [2]), ("C 是 D。", [3])]),
        ("A [1,2]. B [1，3]. C [2、3].", [("A.", [1, 2]), ("B.", [1, 3]), ("C.", [2, 3])]),
        ("A [1][2]. B [3]", [("A.", [1, 2]), ("B", [3])]),
        ("A [1-3].", [("A.", [1, 2, 3])]),
        ("A. [2] B. [3]", [("A.", [2]), ("B.", [3])]),
        ("A [1].   \n\n", [("A.", [1])]),
        ("A [ 1 ] .", [("A.", [1])]),
        ("A [1, 2, 2].", [("A.", [1, 2])]),
        (
            "模型支持中文[1]。它很快[2]！对吗[3]？",
            [("模型支持中文。", [1]), ("它很快！", [2]), ("对吗？", [3])],
        ),
        ("- A [1].\n- B [2].", [("A.", [1]), ("B.", [2])]),
        ("1. A [1].\n2. B [2].", [("A.", [1]), ("B.", [2])]),
        ("A [1]\nB [2]", [("A", [1]), ("B", [2])]),
        (
            "Use it, e.g. for batching [1]. Next [2].",
            [("Use it, e.g. for batching.", [1]), ("Next.", [2])],
        ),
        ("Call `x[0]` on version 1.5 [2].", [("Call `x[0]` on version 1.5.", [2])]),
        ('He said "yes." [1] Then left [2].', [('He said "yes."', [1]), ("Then left.", [2])]),
        ("A [source 1]. B [Fragment 2]. C [片段3]。", [("A.", [1]), ("B.", [2]), ("C。", [3])]),
        ("A [1] and B [2] differ.", [("A and B differ.", [1, 2])]),
        ("No citation here. Second one.", [("No citation here.", []), ("Second one.", [])]),
    ],
)
def test_parse_citation_variants(text: str, expected: list[tuple[str, list[int]]]) -> None:
    parsed = parse_answer(text)
    assert not parsed.not_found
    assert [(s.text, list(s.markers)) for s in parsed.sentences] == expected


def test_citation_check_unknown_and_unsupported() -> None:
    ctx = context(3)
    parsed = parse_answer("A [1]. B [7]. C. D [0][2]. E [2][2].")

    checked = check_citations(parsed, ctx)

    assert [s.text for s in checked.sentences] == ["A.", "B.", "C.", "D.", "E."]
    assert [[c.marker for c in s.citations] for s in checked.sentences] == [[1], [], [], [2], [2]]
    assert [[c.chunk_id for c in s.citations] for s in checked.sentences][3] == ["doc#1"]
    assert checked.removed_markers == [7, 0]
    assert checked.unsupported_sentences == 2
    assert checked.has_valid_citation

    nothing = check_citations(parse_answer("Nothing cites [4]. At all."), ctx)
    assert not nothing.has_valid_citation
    assert nothing.removed_markers == [4]
    assert nothing.unsupported_sentences == 2


@pytest.mark.parametrize(
    "text",
    [
        NOT_FOUND["en"],
        NOT_FOUND["zh"],
        f"  {NOT_FOUND['en']}  \n",
        NOT_FOUND["en"].rstrip("."),
        f'"{NOT_FOUND["zh"]}"',
        f"**{NOT_FOUND['en']}**",
        f"{NOT_FOUND['zh']}[1]",
        f"{NOT_FOUND['en']}\n{NOT_FOUND['en']}",
    ],
)
def test_parse_not_found_variants(text: str) -> None:
    parsed = parse_answer(text)
    assert parsed.not_found
    assert parsed.sentences == ()


def test_parse_drops_not_found_sentence_from_an_answer() -> None:
    parsed = parse_answer(f"A is B [1]. {NOT_FOUND['en']}")
    assert not parsed.not_found
    assert [s.text for s in parsed.sentences] == ["A is B."]
    assert parse_answer("").sentences == ()
    assert not parse_answer("   ").not_found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Brackets right after a word or a closing bracket are code, not markers; text intact.
        (
            "Take logits[1] for the positive class [2].",
            [("Take logits[1] for the positive class.", [2])],
        ),
        ("Use outputs[0][1] here [1].", [("Use outputs[0][1] here.", [1])]),
        ("Call f(x)[1] first [3].", [("Call f(x)[1] first.", [3])]),
        ("The shape is torch.Size([1, 2]) [2].", [("The shape is torch.Size([1, 2]).", [2])]),
        ("Set x=[1, 2] before [1].", [("Set x=[1, 2] before.", [1])]),
        ("Pass logits[1].", [("Pass logits[1].", [])]),
        # Indented code lines are kept whole.
        (
            "Write this [1].\n    a[1] = 3\nDone [2].",
            [("Write this.", [1]), ("a[1] = 3", []), ("Done.", [2])],
        ),
        ("Write this [1].\n\ta[1, 2] = 3", [("Write this.", [1]), ("a[1, 2] = 3", [])]),
        # Markers right after Chinese text, after another marker, in full-width brackets, or
        # followed by Chinese punctuation still count.
        ("模型支持中文[1][2]。", [("模型支持中文。", [1, 2])]),
        (
            "默认窗口是 r=127[2]。用法见 pipeline[1][3]，很简单。",
            [
                ("默认窗口是 r=127。", [2]),
                ("用法见 pipeline，很简单。", [1, 3]),
            ],
        ),
        ("C 是 D［3］。", [("C 是 D。", [3])]),
    ],
)
def test_parse_keeps_code_brackets(text: str, expected: list[tuple[str, list[int]]]) -> None:
    parsed = parse_answer(text)
    assert [(s.text, list(s.markers)) for s in parsed.sentences] == expected


@pytest.mark.parametrize(
    "text",
    [
        f"I'm sorry. {NOT_FOUND['en']}",
        "The documentation does not contain the answer.",
        "Sorry, I cannot help. The provided fragments do not mention LongT5.",
        "抱歉。手册里没有找到答案。",
        "文档中没有提到这个参数。",
    ],
)
def test_parse_not_found_with_uncited_extras(text: str) -> None:
    parsed = parse_answer(text)
    assert parsed.not_found
    assert parsed.sentences == ()


def test_parse_not_found_variant_beside_cited_sentences_is_dropped() -> None:
    parsed = parse_answer("A is B [1]. The fragments do not mention C.")
    assert not parsed.not_found
    assert [s.text for s in parsed.sentences] == ["A is B."]
