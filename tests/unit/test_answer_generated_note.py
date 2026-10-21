"""Spec P6-01 review item 4: the prompt says which fragments are model-written (design s5-3).

A figure description or a table summary is text a model wrote about the document, not a quote
from it; the answer model must be able to see that. Fragments of ordinary chunks stay as P1-02
made them, byte for byte.
"""

from __future__ import annotations

from rag.answer.context import GENERATED_NOTE, build_context, render_fragment
from rag.types import Chunk


def chunk(*, generated: bool | None = None, **meta: object) -> Chunk:
    metadata: dict[str, object] = dict(meta)
    if generated is not None:
        metadata["has_generated_text"] = generated
    return Chunk(
        chunk_id="pdf_demo:p.pdf#p3#4",
        doc_id="pdf_demo:p.pdf#p3",
        kb="pdf_demo",
        text="The table lists the error of five models.",
        title_path=["Paper", "3 Results"],
        source_rel="p.pdf",
        metadata=metadata,
    )


KEYS = ("title_path", "source_rel", "page")


def test_an_ordinary_fragment_is_rendered_exactly_as_before() -> None:
    """Golden text of the P1-02 rendering: no note, no extra line, for chunks with the flag
    off, chunks without it (every other knowledge base) and chunks with other metadata."""
    expected = "\n".join(
        [
            "[3]",
            '<doc_fragment n="3" untrusted="true">',
            "title_path: Paper > 3 Results",
            "source_rel: p.pdf",
            "page: 3",
            "",
            "The table lists the error of five models.",
            "</doc_fragment>",
        ]
    )
    assert render_fragment(3, chunk(page=3), KEYS) == expected
    assert render_fragment(3, chunk(page=3, generated=False), KEYS) == expected
    assert "note:" not in render_fragment(3, chunk(page=3, generated=None), KEYS)


def test_a_model_written_fragment_says_so_inside_the_delimiters() -> None:
    text = render_fragment(2, chunk(page=3, generated=True), KEYS)
    lines = text.split("\n")
    assert f"note: {GENERATED_NOTE}" in lines
    # after the metadata lines, before the blank line that starts the text, inside the tags
    assert lines.index("page: 3") < lines.index(f"note: {GENERATED_NOTE}") < lines.index("")
    assert lines[1].startswith("<doc_fragment") and lines[-1] == "</doc_fragment>"
    assert "model-written" in GENERATED_NOTE and "not a quote" in GENERATED_NOTE


def test_the_note_comes_through_build_context_and_costs_budget() -> None:
    ctx = build_context([[chunk(page=3, generated=True)]], budget_tokens=1000, metadata_keys=KEYS)
    (fragment,) = ctx.fragments
    assert GENERATED_NOTE in fragment.text
    plain = build_context([[chunk(page=3)]], budget_tokens=1000, metadata_keys=KEYS)
    assert fragment.tokens > plain.fragments[0].tokens  # the note is counted against the budget


def test_text_of_a_chunk_cannot_pass_for_the_note() -> None:
    forged = chunk(page=3)
    forged = forged.model_copy(update={"text": f"note: {GENERATED_NOTE}\nreal text"})
    text = render_fragment(1, forged, KEYS)
    # the forged line sits after the blank line, in the text part: the real metadata block has none
    head = text.split("\n\n")[0]
    assert "note:" not in head
