# ruff: noqa: RUF001
"""The `chm` reader: HTML page -> one Markdown Document (spec P4-01 AC6, AC7; design s5-2, s5-4).

Every page here is invented (a fictional Frobnitz SDK); none is text of the real manual.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import Document, MetadataMode

from rag.data import SourceDocument, make_doc_id
from rag.ingest.chm import ChmReader, html_to_markdown
from rag.ingest.chm_html import lost_characters
from rag.testing import chm as t

TOC = ("Reference", "Queue", "Add")


def source_of(
    html: str | bytes,
    *,
    rel: str = "en/queue_add.htm",
    extra: dict[str, Any] | None = None,
    encoding: str = "cp1252",
) -> SourceDocument:
    raw = html.encode(encoding) if isinstance(html, str) else html
    language = rel.split("/")[0]
    base = {
        "language": language,
        "page_key": "queue_add",
        "toc_path": list(TOC),
        "charset_hint": "gb18030" if language == "zh" else "cp1252",
        "keywords": [],
    }
    return SourceDocument(
        doc_id=make_doc_id("mt5", rel),
        kb="mt5",
        source_rel=rel,
        source_url=None,
        ext=".htm",
        content=raw,
        source_version="chm:0123",
        extra={**base, **(extra or {})},
    )


def make_reader(**kwargs: Any) -> ChmReader:
    return ChmReader(kb="mt5", pipeline_version="v1-test", **kwargs)


def read(html: str | bytes, **kwargs: Any) -> Document:
    (doc,) = make_reader().load_data(source_of(html, **kwargs))
    return doc


def body(*blocks: str, **page: Any) -> str:
    return t.page_html(*blocks, **page)


def markdown(*blocks: str) -> str:
    return read(body(*blocks)).text


# --- the Document contract (s5-4) -------------------------------------------------------------


def test_reader_contract_fields() -> None:
    doc = read(body(t.h1("Queue::Add"), t.short("Add a widget."), t.link("queue.htm", "Queue")))
    assert doc.id_ == "mt5:en/queue_add.htm"
    assert doc.metadata == {
        "kb": "mt5",
        "doc_id": "mt5:en/queue_add.htm",
        "title_path": "Reference > Queue > Add",
        "source_rel": "en/queue_add.htm",
        "unit": "queue_add.htm",
        "source_url": None,
        "doc_type": "chm",
        "language": "en",
        "acl_groups": ["public"],
        "images": [],
        "pipeline_version": "v1-test",
        "page_key": "queue_add",
        "links_to": ["queue"],
    }
    for forbidden in ("mtime", "size", "source_version", "ingest_run_id", "content_hash"):
        assert forbidden not in doc.metadata


def test_reader_stable_id_and_hash() -> None:
    page = body(t.h1("Queue::Add"), t.short("Add a widget."))
    first, second = read(page), read(page)
    assert (first.id_, first.hash) == (second.id_, second.hash)
    # What the page says outside its body (navigation, head) does not change the hash ...
    other_nav = body(
        t.h1("Queue::Add"),
        t.short("Add a widget."),
        crumbs=("Other", "Crumbs"),
        prev="a.htm",
        next="b.htm",
        title="Another title",
    )
    assert read(other_nav).hash == first.hash
    # ... and the body does.
    assert read(body(t.h1("Queue::Add"), t.short("Add two widgets."))).hash != first.hash


def test_reader_embed_text_is_title_path_and_body_only() -> None:
    doc = read(body(t.h1("Queue::Add"), t.short("Add a widget."), t.link("queue.htm", "Queue")))
    embed = doc.get_content(MetadataMode.EMBED)
    assert embed.startswith("title_path: Reference > Queue > Add\n\n# Queue::Add")
    for hidden in ("page_key", "links_to", "pipeline_version", "acl_groups", "chm", "mt5"):
        assert hidden not in embed.split("\n\n", 1)[0]
    llm = doc.get_content(MetadataMode.LLM)
    assert "source_rel: en/queue_add.htm" in llm and "page_key" not in llm


def test_reader_language_and_page_key_come_from_the_connector() -> None:
    zh = read(
        body(t.h1("Queue::Add"), t.short("把部件添加到队列。"), charset="gb2312"),
        rel="zh/queue_add_cn.htm",  # the file name says nothing: the connector's key is used
        encoding="gb18030",
        extra={"toc_path": ["参考", "队列", "添加"], "page_key": "queue_add"},
    )
    assert zh.metadata["language"] == "zh"
    assert zh.metadata["page_key"] == "queue_add"
    assert zh.metadata["title_path"] == "参考 > 队列 > 添加"
    assert "把部件添加到队列。" in zh.text


def test_reader_without_connector_hints_derives_what_it_can() -> None:
    page = body(t.h1("Add"), t.short("把部件添加到队列，然后返回。"), charset="gb2312")
    src = source_of(page, rel="zh/Sub/Queue_Add.htm", encoding="gb18030")
    bare = src.model_copy(update={"extra": {}})
    (doc,) = make_reader().load_data(bare)
    assert doc.metadata["language"] == "zh"  # detected from the text
    assert doc.metadata["page_key"] == "sub/queue_add"  # the language folder is not part of it
    assert doc.metadata["title_path"] == "Add"  # the page's own heading


def test_reader_title_path_fallbacks_are_counted() -> None:
    reader = make_reader()
    no_toc = source_of(body(t.h1("Own Heading"), t.short("x y z")), extra={"toc_path": None})
    no_heading = source_of(
        body(t.short("only text here"), title="Head Title"),
        rel="en/b.htm",
        extra={"toc_path": None, "page_key": "b"},
    )
    nothing = source_of(
        body(t.short("only text here"), title=""),
        rel="en/c.htm",
        extra={"toc_path": None, "page_key": "c"},
    )
    paths = [reader.load_data(s)[0].metadata["title_path"] for s in (no_toc, no_heading, nothing)]
    assert paths == ["Own Heading", "Head Title", "c"]
    assert reader.stats.not_in_toc == 3
    assert reader.stats.no_title == 1  # neither a heading nor a <title>: the file name is used


def test_reader_records_encodings_fallbacks_and_decode_errors() -> None:
    reader = make_reader()
    zh_text = "把部件添加到队列，然后返回成功。" * 2
    honest = source_of(
        body(t.short(zh_text), charset="gb2312"),
        rel="zh/a.htm",
        encoding="gb18030",
        extra={"page_key": "a"},
    )
    lying = source_of(
        body(t.short(zh_text), charset="iso-8859-1"),
        rel="zh/b.htm",
        encoding="gb18030",
        extra={"page_key": "b"},
    )
    ascii_page = source_of(body(t.short("plain")), rel="en/c.htm", extra={"page_key": "c"})
    broken = source_of(
        b"<html><body><div id='help'><p>ok \x81\x20 \x81\x81\x81 end</p></div></body></html>",
        rel="en/d.htm",
        extra={"page_key": "d", "charset_hint": None},
    )
    for src in (honest, lying, ascii_page, broken):
        reader.load_data(src)
    stats = reader.stats
    assert stats.pages == 4
    assert stats.encodings == {"zh:gb18030": 2, "en:ascii": 1, "en:cp1252": 1}
    assert stats.declared == {"gb2312": 1, "iso-8859-1": 2, "none": 1}
    assert stats.fallbacks == 2  # the lying page and the broken one
    assert stats.decode_errors == 1
    assert stats.replacement_chars > 0
    assert zh_text in reader.load_data(lying)[0].text


def test_reader_counts_duplicate_page_keys_within_a_language() -> None:
    reader = make_reader()
    page = body(t.short("same"))
    reader.load_data(source_of(page, rel="en/Queue_Add.htm"))
    reader.load_data(source_of(page, rel="en/queue_add.htm"))  # same key, other spelling
    reader.load_data(source_of(page, rel="zh/queue_add.htm"))  # other language: fine
    assert reader.stats.duplicate_page_keys == ["mt5:en/queue_add.htm"]


def test_reader_empty_body_gives_an_empty_document_and_is_counted() -> None:
    reader = make_reader()
    (doc,) = reader.load_data(source_of(body(t.empty_paragraph())))
    assert doc.text == ""
    assert reader.stats.empty_pages == 1


# --- Markdown conversion -----------------------------------------------------------------------


def test_reader_headings_and_paragraphs() -> None:
    out = markdown(
        t.h1("Queue::Add"),
        t.short("Add a widget."),
        t.h2("Details"),
        t.text("First   line,\nsecond&nbsp;line."),
        t.h3("More"),
        t.bold("Note"),
    )
    assert out == (
        "# Queue::Add\n\nAdd a widget.\n\n## Details\n\nFirst line, second line.\n\n"
        "### More\n\n**Note**"
    )


def test_reader_code_block_keeps_lines_and_indentation() -> None:
    out = markdown(
        t.code("bool  Queue::Add(", "   const void *w,   // a widget", "   int n", "   )"),
    )
    assert out == ("```\nbool  Queue::Add(\n   const void *w,   // a widget\n   int n\n   )\n```")


def test_reader_code_with_angle_brackets_entities_and_fences() -> None:
    out = markdown(t.code("if (a < b && c > d) {", "  Vec<int> v; // ``` not a fence", "}"))
    assert "if (a < b && c > d) {" in out
    assert "  Vec<int> v; // ``` not a fence" in out
    lines = out.split("\n")
    assert lines[0] == lines[-1] == "````"  # a longer fence, so the inner ``` cannot close it


def test_reader_two_code_boxes_are_two_code_blocks() -> None:
    out = markdown(t.code("first()"), t.short("between"), t.code("second()"))
    assert out == "```\nfirst()\n```\n\nbetween\n\n```\nsecond()\n```"


def test_reader_tables_become_markdown_tables() -> None:
    out = markdown(
        t.table(
            ["Method", "Purpose"],
            [
                [t.link("queue_add.htm", "Add"), "Add a widget | or two."],
                ["Clear", "Remove every widget."],
            ],
        )
    )
    assert out == (
        "| Method | Purpose |\n| --- | --- |\n| Add | Add a widget \\| or two. |\n"
        "| Clear | Remove every widget. |"
    )


def test_reader_table_without_header_row_and_ragged_rows() -> None:
    raw = (
        '<div><table class="table"><tr><td><p class="p_fortable">a</p></td>'
        '<td><p class="p_fortable">b</p></td></tr>'
        '<tr><td><p class="p_fortable">c</p></td></tr></table></div>'
    )
    assert markdown(raw) == "| a | b |\n| --- | --- |\n| c |  |"


def test_reader_note_box_becomes_a_quote() -> None:
    assert markdown(t.note("A queue grows by one step.")) == "> A queue grows by one step."


def test_reader_note_box_keeps_lists_and_several_paragraphs() -> None:
    box = (
        '<div><table class="attentable" border="1"><tr class="attentable">'
        '<td class="attentable"><p class="p_tableatten">One of these holds:</p>'
        "<ul><li>the first condition</li><li>the second condition</li></ul>"
        '<p class="p_tableatten">Otherwise nothing happens.</p></td></tr></table></div>'
    )
    assert markdown(box) == (
        "> One of these holds:\n>\n> - the first condition\n> - the second condition\n>\n"
        "> Otherwise nothing happens."
    )


def test_reader_note_box_keeps_code_tables_and_nested_notes() -> None:
    box = (
        '<div><table class="attentable"><tr><td class="attentable">'
        '<p class="p_tableatten">See the call:</p>'
        + t.code("run(1)", "  stop()")
        + t.table(["Key", "Value"], [["a", "b"]])
        + t.note("Inner warning.")
        + "</td></tr></table></div>"
    )
    assert markdown(box) == (
        "> See the call:\n>\n> ```\n> run(1)\n>   stop()\n> ```\n>\n"
        "> | Key | Value |\n> | --- | --- |\n> | a | b |\n>\n> > Inner warning."
    )


def test_reader_note_box_with_text_outside_paragraphs_and_an_empty_one() -> None:
    bare = (
        '<div><table class="attentable"><tr><td class="attentable">Loose note text.</td></tr>'
        "</table></div>"
    )
    assert markdown(bare) == "> Loose note text."
    empty = (
        '<div><table class="attentable"><tr><td class="attentable"><p>&nbsp;</p></td></tr>'
        "</table></div>"
    )
    assert markdown(t.short("Before."), empty, t.short("After.")) == "Before.\n\nAfter."


def test_reader_lists() -> None:
    out = markdown(
        t.bullets("first", "second with " + t.link("queue.htm", "a link")),
        t.numbered("one", "two"),
    )
    assert out == "- first\n- second with a link\n\n1. one\n2. two"


def test_reader_list_item_with_title_line_break_and_description() -> None:
    raw = (
        '<ul><li class="p_li"><span class="f_li" style="font-weight: bold;">'
        + t.link("queue.htm", "CQueue")
        + '</span><br>\n<span class="f_li">A queue of widgets.</span></li></ul>'
    )
    assert markdown(raw) == "- CQueue A queue of widgets."


def test_reader_parameter_name_and_description_become_one_item() -> None:
    out = markdown(
        t.bold("Parameters"),
        t.param("w", "[in] A pointer to the widget."),
        t.param("n", "[in] How many."),
        t.bold("Results"),
        t.short("True on success."),
    )
    assert out == (
        "**Parameters**\n\n- `w`: [in] A pointer to the widget.\n- `n`: [in] How many.\n\n"
        "**Results**\n\nTrue on success."
    )


def test_reader_images_are_dropped_with_a_marker_and_listed_once() -> None:
    doc = read(
        body(
            t.h1("Queue::Add"),
            t.image("queue.png", "Queue overview"),
            t.image("img/Inner.png"),
            t.image("queue.png", "again"),
        )
    )
    assert doc.text == (
        "# Queue::Add\n\n[image: Queue overview]\n\n[image: Inner.png]\n\n[image: again]"
    )
    assert doc.metadata["images"] == ["en/queue.png", "en/img/Inner.png"]


def test_reader_image_paths_are_relative_to_the_page_folder() -> None:
    page = body(t.image("a.png"), t.image("../b.png"), t.image("sub/c.png"))
    doc = read(page, rel="en/deep/page.htm", extra={"page_key": "deep/page"})
    assert doc.metadata["images"] == ["en/deep/a.png", "en/b.png", "en/deep/sub/c.png"]


def test_reader_links_become_text_and_their_page_keys_go_to_metadata() -> None:
    doc = read(
        body(
            t.text(
                f"See {t.link('Queue.htm', 'the queue')}, "
                f"{t.link('queue_clear.htm#top', 'clear')}, "
                f"{t.link('queue_clear.htm', 'clear again')}, {t.link('queue_add.htm', 'itself')}, "
                f"{t.link('%71ueue_remove.HTM', 'remove')}, "
                f"{t.link('http://example.com/a?b=1', 'the site')}, "
                f"{t.link('mailto:help@example.com', 'mail')}, {t.link('shot.png', 'the shot')}."
            )
        )
    )
    assert doc.text == (
        "See the queue, clear, clear again, itself, remove, "
        "[the site](http://example.com/a?b=1), mail, the shot."
    )
    # in order of appearance, once each, the page itself left out, images are not pages
    assert doc.metadata["links_to"] == ["queue", "queue_clear", "queue_remove"]


def test_reader_links_resolve_relative_to_the_page_folder() -> None:
    doc = read(
        body(t.text(f"{t.link('../top.htm', 'up')} {t.link('sib.htm', 'sibling')}")),
        rel="en/sub/page.htm",
        extra={"page_key": "sub/page"},
    )
    assert doc.metadata["links_to"] == ["top", "sub/sib"]


def test_reader_text_with_angle_brackets_and_entities_stays() -> None:
    out = markdown(
        t.text(
            "A value &lt; 0 means failure; use Vec&lt;T&gt; and &lt;b&gt;bold&lt;/b&gt; &amp; more."
        ),
    )
    assert out == "A value < 0 means failure; use Vec<T> and <b>bold</b> & more."


def test_reader_prose_that_looks_like_markdown_is_kept_as_prose() -> None:
    out = markdown(t.text("# not a heading"), t.text("1) first &amp; 2) second"))
    assert out == "\\# not a heading\n\n1) first & 2) second"


def test_reader_page_without_help_container_uses_the_body() -> None:
    raw = "<html><head><title>x</title></head><body><p>Loose paragraph text.</p></body></html>"
    reader = make_reader()
    (doc,) = reader.load_data(source_of(raw))
    assert doc.text == "Loose paragraph text."
    assert reader.stats.no_container == 1


# --- noise removed while reading, with counts (AC7) ----------------------------------------------


def test_html_noise_navigation_bar_is_dropped_and_counted() -> None:
    reader = make_reader()
    page = body(
        t.h1("Queue::Add"),
        t.short("Add a widget."),
        crumbs=("Reference", "Queue"),
    )
    (doc,) = reader.load_data(source_of(page))
    for gone in ("Reference /", "Previous", "Next"):
        assert gone not in doc.text
    stat = reader.stats.html_rules["nav_bar"]
    assert stat.pages_hit == 1 and stat.chars_removed > 10


def test_html_noise_navigation_words_in_the_body_survive() -> None:
    out = markdown(
        t.text("Go to the Previous or Next page of the settings."),
        t.table(["Previous", "Next"], [["old value", "new value"]]),
        t.h2("Next steps"),
    )
    assert "Go to the Previous or Next page of the settings." in out
    assert "| Previous | Next |" in out
    assert "## Next steps" in out


def test_html_noise_empty_paragraphs_are_dropped_and_counted() -> None:
    reader = make_reader()
    page = body(
        t.short("Kept."),
        t.empty_paragraph(),
        '<p class="p_Text"><span class="f_Text">  &nbsp; </span></p>',
        t.image("x.png", "A picture"),  # no text, but not empty
        t.text(t.link("queue.htm", "only a link")),
    )
    (doc,) = reader.load_data(source_of(page))
    assert doc.text == "Kept.\n\n[image: A picture]\n\nonly a link"
    assert reader.stats.html_rules["empty_paragraph"].pages_hit == 1


def test_html_noise_anchor_targets_are_dropped_text_links_stay() -> None:
    reader = make_reader()
    page = body(
        t.text(f"{t.anchor('top')}Start {t.anchor('mid')}here and "),
        '<p class="p_Text"><a name="named" class="topiclink">Named text</a> stays.</p>',
    )
    (doc,) = reader.load_data(source_of(page))
    assert doc.text == "Start here and\n\nNamed text stays."
    assert reader.stats.html_rules["anchor"].pages_hit == 1
    assert doc.metadata["links_to"] == []


def test_html_noise_scripts_and_styles_are_dropped_but_the_word_script_stays() -> None:
    reader = make_reader()
    page = body(
        t.text("Write the script file."),
        "<script>var hidden = 1;</script><style>.x { color: red }</style>",
    )
    (doc,) = reader.load_data(source_of(page))
    assert doc.text == "Write the script file."
    assert reader.stats.html_rules["script_style"].pages_hit == 1


def test_html_to_markdown_is_pure_and_reports_images_and_links() -> None:
    converted = html_to_markdown(
        "<div id='help'><p class='p_Text'><a href='b.htm'>b</a></p></div>", page_dir=""
    )
    assert converted.markdown == "b"
    assert converted.links_to == ["b"]
    assert converted.images == []


def test_reader_stats_count_images_links_tables_and_code() -> None:
    reader = make_reader()
    page = body(
        t.image("a.png", "A"),
        t.text(t.link("b.htm", "b") + " " + t.link("http://example.com", "e")),
        t.table(["h"], [["c"]]),
        t.code("x()"),
    )
    reader.load_data(source_of(page))
    stats = reader.stats
    assert (stats.images, stats.internal_links, stats.external_links) == (1, 1, 1)
    assert (stats.tables, stats.code_blocks) == (1, 1)


def test_reader_is_registered_under_the_name_chm() -> None:
    from rag.registry import readers

    assert "chm" in readers
    assert isinstance(readers.create("chm", kb="mt5", pipeline_version="v"), ChmReader)


# --- fidelity: the words of a page that reach the Markdown ---------------------------------------


def test_fidelity_every_construct_of_a_rich_page_keeps_its_words() -> None:
    page = body(
        t.h1("Queue::Add"),
        t.short("Add a widget."),
        t.code("run(1)", "  stop()"),
        t.bold("Parameters"),
        t.param("w", "[in] The widget."),
        t.table(["Key", "Value"], [["a", t.link("b.htm", "bee")]]),
        t.bullets("first", "second"),
        t.numbered("one", "two"),
        t.image("x.png", "A picture"),
        t.note("Careful with " + t.link("http://example.com", "the site") + "."),
        '<div><table class="attentable"><tr><td class="attentable">'
        "<ul><li>inside a note</li></ul></td></tr></table></div>",
    )
    converted = html_to_markdown(page, page_dir="")
    assert converted.lost_chars == 0
    assert converted.visible_chars > 100


def test_fidelity_counts_words_the_converter_dropped() -> None:
    assert lost_characters("alpha beta gamma", "alpha gamma") == 4  # "beta"
    assert lost_characters("alpha", "alpha [image: a caption] ](http://x.example/p)") == 0
    assert lost_characters("a b", "a b extra words") == 0  # only loss counts, not additions
    assert lost_characters("队列 保存", "队列") == 2


def test_fidelity_flags_a_page_that_loses_words() -> None:
    reader = make_reader()
    kept = source_of(body(t.short("All of this text is kept.")), rel="en/ok.htm")
    # a <head> element inside the help text: the converter has no way to carry it
    lossy = source_of(
        body(t.short("Visible."), "<head>hidden words that vanish</head>"), rel="en/lossy.htm"
    )
    for src in (kept, lossy):
        reader.load_data(src)
    stats = reader.stats
    assert stats.lossy_pages == ["mt5:en/lossy.htm"]
    assert stats.lost_chars == len("hiddenwordsthatvanish")
    assert stats.visible_chars > stats.lost_chars
    assert stats.to_dict()["lossy_pages"] == ["mt5:en/lossy.htm"]
