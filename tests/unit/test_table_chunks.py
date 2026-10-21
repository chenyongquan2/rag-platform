"""Spec P6-01: tables as their own chunks, header repeat, summary chunks, generated-text flag."""

from __future__ import annotations

import pytest
from llama_index.core.schema import Document, MetadataMode, NodeRelationship, TextNode

from rag.ingest.chunk import AssignChunkIds, MarkdownSplit, MergeTiny, TokenSplit
from rag.ingest.describe import ModelTextCache, ModelTextService
from rag.ingest.tables import (
    IMAGE_MARKER_PREFIX,
    MarkGeneratedText,
    TableChunks,
    TableSummary,
    rule_summary,
    split_table,
)
from rag.testing.pdf import FakeSummarizer

DOC = "pdf_demo:paper.pdf#p3"


def words(text: str) -> list[str]:
    return text.split()


def table(rows: int, *, cols: tuple[str, ...] = ("Model", "Error")) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    body = [f"| name{i} | {i}.5 |" for i in range(rows)]
    return "\n".join([head, *body])


def document(text: str, doc_id: str = DOC) -> Document:
    meta = {
        "kb": "pdf_demo",
        "doc_id": doc_id,
        "source_rel": "paper.pdf",
        "title_path": "Paper > 3 Results",
        "page": 3,
        "pipeline_version": "v1-test",
    }
    return Document(
        id_=doc_id,
        text=text,
        metadata=meta,
        excluded_embed_metadata_keys=[k for k in meta if k != "title_path"],
        excluded_llm_metadata_keys=[k for k in meta if k not in ("title_path", "source_rel")],
    )


def run(nodes, *steps):
    for step in steps:
        nodes = step(nodes)
    return list(nodes)


def table_chunks(max_tokens: int = 60) -> TableChunks:
    return TableChunks(
        max_tokens=max_tokens,
        tokenizer=words,
        embed_keys=["title_path"],
        llm_keys=["title_path", "source_rel"],
    )


# ---- split_table (AC7, AC8) --------------------------------------------------------------------


def test_split_table_returns_the_table_whole_when_it_fits() -> None:
    text = table(3)
    assert split_table(text, 100, lambda t: len(words(t))) == [text]


def test_split_table_repeats_the_header_and_keeps_every_row_in_order() -> None:
    text = table(12)
    parts = split_table(text, 30, lambda t: len(words(t)))
    assert len(parts) > 1
    header = text.split("\n")[:2]
    rows: list[str] = []
    for part in parts:
        lines = part.split("\n")
        assert lines[:2] == header
        assert len(words(part)) <= 30
        rows.extend(lines[2:])
    assert rows == text.split("\n")[2:]


def test_split_table_without_a_repeat_has_no_header_in_later_parts() -> None:
    parts = split_table(table(12), 30, lambda t: len(words(t)), repeat_header=False)
    assert parts[0].startswith("| Model")
    assert not parts[1].startswith("| Model")
    assert "\n".join(parts) == table(12)


def test_split_table_puts_an_oversized_row_in_a_part_of_its_own() -> None:
    big = "| " + " ".join(f"w{i}" for i in range(80)) + " | 1 |"
    text = "\n".join([table(2), big, "| last | 9 |"])
    parts = split_table(text, 30, lambda t: len(words(t)))
    assert any(big in p and p.count("\n") == 2 for p in parts)
    assert parts[-1].endswith("| last | 9 |")


def test_split_table_text_without_separator_row_repeats_nothing() -> None:
    text = "| a | b |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
    parts = split_table(text, 8, lambda t: len(words(t)))
    assert len(parts) > 1
    assert "\n".join(parts) == text


# ---- table_chunks (AC7) ----------------------------------------------------------------------


def test_table_chunks_make_the_table_its_own_chunk_with_caption() -> None:
    text = (
        "Intro paragraph about results.\n\n"
        "Table 2. Error by model.\n\n" + table(3) + "\n\nAfter the table we discuss it."
    )
    nodes = run([document(text)], MarkdownSplit(), table_chunks())
    kinds = [n.metadata["chunk_kind"] for n in nodes]
    assert kinds == ["text", "table", "text"]
    table_node = nodes[1]
    body = table_node.get_content(MetadataMode.NONE)
    assert body.startswith("Table 2. Error by model.\n\n| Model | Error |")
    assert body.endswith("| name2 | 2.5 |")
    assert table_node.metadata["table_id"] == f"{DOC}#t0"
    assert "Table 2" not in nodes[0].get_content(MetadataMode.NONE)
    assert nodes[0].get_content(MetadataMode.NONE) == "Intro paragraph about results."
    assert nodes[2].get_content(MetadataMode.NONE) == "After the table we discuss it."
    assert "table_id" not in nodes[0].metadata
    assert nodes[1].metadata["title_path"] == "Paper > 3 Results"


def test_table_chunks_keep_a_paragraph_that_is_not_a_caption() -> None:
    text = "The next table shows errors.\n\n" + table(2)
    nodes = run([document(text)], MarkdownSplit(), table_chunks())
    assert [n.metadata["chunk_kind"] for n in nodes] == ["text", "table"]
    assert nodes[1].get_content(MetadataMode.NONE).startswith("| Model")


def test_table_chunks_split_a_long_table_with_header_and_caption_in_every_part() -> None:
    text = "Table 9. Long one.\n\n" + table(20)
    nodes = run([document(text)], MarkdownSplit(), table_chunks(max_tokens=40))
    parts = [n for n in nodes if n.metadata["chunk_kind"] == "table"]
    assert len(parts) > 1
    assert {n.metadata["table_id"] for n in parts} == {f"{DOC}#t0"}
    for part in parts:
        body = part.get_content(MetadataMode.NONE)
        assert body.startswith("Table 9. Long one.\n\n| Model | Error |")
        meta_tokens = len(words(part.get_metadata_str(MetadataMode.LLM)))
        assert len(words(body)) + meta_tokens <= 40
    rows = [ln for n in parts for ln in n.get_content(MetadataMode.NONE).split("\n")[4:]]
    assert rows == table(20).split("\n")[2:]


def test_table_chunks_number_tables_over_the_whole_page() -> None:
    text = "## A\n\n" + table(2) + "\n\n## B\n\n" + table(2, cols=("X", "Y"))
    nodes = run([document(text)], MarkdownSplit(), table_chunks())
    ids = [n.metadata["table_id"] for n in nodes if n.metadata["chunk_kind"] == "table"]
    assert ids == [f"{DOC}#t0", f"{DOC}#t1"]


def test_table_chunks_leave_tables_inside_code_fences_alone() -> None:
    text = "```\n" + table(3) + "\n```"
    nodes = run([document(text)], MarkdownSplit(), table_chunks())
    assert [n.metadata["chunk_kind"] for n in nodes] == ["text"]


def test_table_chunks_without_tables_keep_the_text_as_it_is() -> None:
    text = "Just text.\n\nMore text | with a bar."
    (node,) = run([document(text)], MarkdownSplit(), table_chunks())
    assert node.get_content(MetadataMode.NONE) == text
    assert node.metadata["chunk_kind"] == "text"


def test_table_chunks_are_not_merged_with_their_neighbours() -> None:
    text = "Short intro.\n\n" + table(1) + "\n\nShort outro."
    chain = [
        MarkdownSplit(),
        table_chunks(max_tokens=60),
        TokenSplit(max_tokens=60, tokenizer=words),
        MergeTiny(min_tokens=50, max_tokens=60, tokenizer=words),
        AssignChunkIds(embed_keys=["title_path"], llm_keys=["title_path", "source_rel"]),
    ]
    nodes = run([document(text)], *chain)
    assert [n.metadata["chunk_kind"] for n in nodes] == ["text", "table", "text"]
    assert [n.node_id for n in nodes] == [f"{DOC}#0", f"{DOC}#1", f"{DOC}#2"]
    assert nodes[1].metadata["table_id"] == f"{DOC}#t0"


def test_merge_tiny_still_merges_plain_chunks() -> None:
    text = "# A\n\nTiny.\n\n## B\n\nAlso tiny."
    nodes = run(
        [document(text)],
        MarkdownSplit(),
        TokenSplit(max_tokens=60, tokenizer=words),
        MergeTiny(min_tokens=50, max_tokens=60, tokenizer=words),
    )
    assert len(nodes) == 1


# ---- token_split header repeat (AC8) -----------------------------------------------------------


def split_text(text: str, **kwargs) -> list[str]:
    node = TextNode(text=text, id_="n", metadata={"title_path": "T"})
    node.excluded_embed_metadata_keys = []
    step = TokenSplit(max_tokens=30, tokenizer=words, **kwargs)
    return [n.get_content(MetadataMode.NONE) for n in step([node])]


def test_token_split_default_table_output_unchanged() -> None:
    text = "Before the table.\n\n" + table(12) + "\n\nAfter."
    default = split_text(text)
    explicit = split_text(text, repeat_table_header=False)
    assert default == explicit
    # The behaviour before this spec: rows are cut with no header in the later parts.
    table_parts = [p for p in default if "name" in p]
    assert len(table_parts) > 1
    assert not table_parts[-1].lstrip().startswith("| Model")


def test_token_split_repeats_table_header() -> None:
    text = "Before the table.\n\n" + table(12) + "\n\nAfter."
    chunks = split_text(text, repeat_table_header=True)
    table_parts = [p for p in chunks if "name" in p]
    assert len(table_parts) > 1
    for part in table_parts:
        assert "| Model | Error |" in part
    rows = [ln for p in table_parts for ln in p.split("\n") if ln.startswith("| name")]
    assert rows == table(12).split("\n")[2:]


# ---- table_summary (AC9) ---------------------------------------------------------------------


def with_tables(text: str, max_tokens: int = 60) -> list:
    return run([document(text)], MarkdownSplit(), table_chunks(max_tokens=max_tokens))


def summary_step(mode: str, summarizer: FakeSummarizer | None = None, **kwargs) -> TableSummary:
    service = ModelTextService(cache=ModelTextCache(":memory:"), summarizer=summarizer)
    return TableSummary(mode=mode, service=service, **kwargs)


def test_table_summary_adds_one_linked_chunk_per_table_after_the_pages_chunks() -> None:
    text = "Table 2. Error by model.\n\n" + table(30) + "\n\nTail text."
    nodes = with_tables(text, max_tokens=40)
    n_parts = sum(n.metadata["chunk_kind"] == "table" for n in nodes)
    assert n_parts > 1
    fake = FakeSummarizer("Errors of 30 models.")
    out = summary_step("model", fake)(nodes)
    assert fake.calls == 1  # one call per table, not per part
    summaries = [n for n in out if n.metadata["chunk_kind"] == "table_summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert out[-1] is summary
    assert summary.metadata["table_id"] == f"{DOC}#t0"
    assert summary.metadata["has_generated_text"] is True
    assert summary.metadata["title_path"] == "Paper > 3 Results"
    assert summary.get_content(MetadataMode.NONE) == "Table 2. Error by model. Errors of 30 models."
    assert [n for n in out if n.metadata["chunk_kind"] != "table_summary"] == nodes
    # the model saw the whole table once, without repeated headers or the caption twice
    seen = fake.tables[0]
    assert seen.count("| Model | Error |") == 1
    assert seen.count("| name29 | 29.5 |") == 1


def test_table_summary_keeps_the_source_link_of_the_page() -> None:
    nodes = with_tables(table(2))
    for n in nodes:
        n.relationships[NodeRelationship.SOURCE] = document("x").as_related_node_info()
    out = summary_step("rule")(nodes)
    summary = out[-1]
    assert summary.ref_doc_id == DOC
    assert summary.metadata["doc_id"] == DOC
    assert summary.metadata["page"] == 3


def test_table_summary_rule_mode_uses_caption_columns_and_first_column() -> None:
    nodes = with_tables("Table 2. Error by model.\n\n" + table(3))
    fake = FakeSummarizer()
    out = summary_step("rule", fake)(nodes)
    assert fake.calls == 0
    text = out[-1].get_content(MetadataMode.NONE)
    assert text.startswith("Table 2. Error by model.")
    assert "Model, Error" in text
    assert "name0" in text and "name2" in text


def test_table_summary_rule_text_limits_the_rows_it_lists() -> None:
    text = rule_summary("", table(40), max_rows=5)
    assert "name4" in text and "name5" not in text
    assert "40 rows" in text


def test_table_summary_off_adds_nothing() -> None:
    nodes = with_tables(table(3))
    assert summary_step("off")(nodes) == nodes


def test_table_summary_model_error_falls_back_to_the_rule_text() -> None:
    class Broken(FakeSummarizer):
        def summarize(self, table_markdown: str, *, caption: str):
            raise TimeoutError("slow")

    step = summary_step("model", Broken())
    out = step(with_tables("Table 2. Error by model.\n\n" + table(3)))
    text = out[-1].get_content(MetadataMode.NONE)
    assert "Columns" in text and text.startswith("Table 2.")
    assert step.stats["model_errors"] == 1


def test_table_summary_budget_error_is_not_swallowed() -> None:
    from rag.ingest.describe import DescribeBudgetError, SpendGuard

    service = ModelTextService(
        cache=ModelTextCache(":memory:"),
        summarizer=FakeSummarizer(),
        guard=SpendGuard(0.001),
        table_estimate_cny=0.01,
    )
    step = TableSummary(mode="model", service=service)
    with pytest.raises(DescribeBudgetError):
        step(with_tables(table(3)))


def test_table_summary_second_run_is_served_from_the_cache() -> None:
    fake = FakeSummarizer()
    service = ModelTextService(cache=ModelTextCache(":memory:"), summarizer=fake)
    step = TableSummary(mode="model", service=service)
    step(with_tables(table(3)))
    step(with_tables(table(3)))
    assert fake.calls == 1
    assert service.stats.table_cache_hits == 1
    assert step.stats["tables"] == 2


def test_table_summary_pages_without_tables_are_untouched() -> None:
    nodes = with_tables("Only text here.")
    assert summary_step("model", FakeSummarizer())(nodes) == nodes


def test_table_summary_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        TableSummary(mode="magic")


def test_table_summary_model_mode_needs_a_service_or_a_model() -> None:
    with pytest.raises(ValueError, match="model"):
        TableSummary(mode="model")


# ---- mark_generated_text (AC9) -----------------------------------------------------------------


def test_mark_generated_text_flags_chunks_with_a_description() -> None:
    marked = f"{IMAGE_MARKER_PREFIX}, not part of the original document): A bar chart.]"
    nodes = [
        TextNode(text="plain text", id_="a", metadata={"chunk_kind": "text"}),
        TextNode(text=f"Figure 1.\n\n{marked}", id_="b", metadata={"chunk_kind": "text"}),
    ]
    out = MarkGeneratedText(embed_keys=[], llm_keys=[])(nodes)
    assert out[0].metadata.get("has_generated_text") is False
    assert out[1].metadata["has_generated_text"] is True


def test_mark_generated_text_keeps_the_flag_of_summary_chunks() -> None:
    node = TextNode(
        text="A summary.",
        id_="s",
        metadata={"chunk_kind": "table_summary", "has_generated_text": True},
    )
    (out,) = MarkGeneratedText(embed_keys=[], llm_keys=[])([node])
    assert out.metadata["has_generated_text"] is True


def test_token_split_default_output_equals_the_output_before_p6_01() -> None:
    """Golden chunks made with `chunk.py` of commit 6c2375a (before this spec) for a fixed text:
    the default behaviour for the hf_docs and mt5 pipelines must not change."""
    rows = "\n".join(f"| name{i} | {i}.5 |" for i in range(12))
    text = (
        "Before the table.\n\n| Model | Error |\n|---|---|\n"
        + rows
        + "\n\nAfter.\n\n```\ncode\n```"
    )
    assert split_text(text) == [
        "Before the table.\n\n| Model | Error |\n|---|---|\n"
        "| name0 | 0.5 |\n| name1 | 1.5 |\n| name2 | 2.5 |",
        "| name3 | 3.5 |\n| name4 | 4.5 |\n| name5 | 5.5 |\n| name6 | 6.5 |\n| name7 | 7.5 |",
        "| name8 | 8.5 |\n| name9 | 9.5 |\n| name10 | 10.5 |\n| name11 | 11.5 |"
        "\n\nAfter.\n\n```\ncode\n```",
    ]
