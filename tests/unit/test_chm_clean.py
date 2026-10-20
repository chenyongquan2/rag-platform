"""CHM cleaning rules, the removal guard and `dedupe(group_by)` (spec P4-01 AC8, AC9, AC10).

Rules are named, switchable, counted and tested both ways (design s6-1, s6-3): a positive case
(the noise goes) and a negative case (text that looks like it stays). Text here is invented.
"""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.schema import Document, MetadataMode

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.data import SourceDocument, make_doc_id
from rag.ingest import KbProcessor, pipeline_version
from rag.ingest.clean import Dedupe, RemovalGuardError, SkipEmpty, Whitespace
from rag.ingest.clean_chm import ChmTitleEcho
from rag.registry import steps
from rag.testing.chm import h1, page_html, text


def doc(
    body: str,
    *,
    title_path: str = "Reference > Queue > Add",
    language: str = "en",
    name: str = "queue_add",
) -> Document:
    doc_id = make_doc_id("mt5", f"{language}/{name}.htm")
    return Document(
        id_=doc_id,
        text=body,
        metadata={"doc_id": doc_id, "title_path": title_path, "language": language},
    )


def texts(docs: list[Any]) -> list[str]:
    return [d.get_content(MetadataMode.NONE) for d in docs]


# --- chm_title_echo ----------------------------------------------------------------------------


def test_title_echo_drops_the_heading_that_repeats_the_last_title() -> None:
    rule = ChmTitleEcho()
    (out,) = rule([doc("# Add\n\nAdd a widget to the queue.\n\n## Details\n\nMore text.")])
    assert (
        out.get_content(MetadataMode.NONE)
        == "Add a widget to the queue.\n\n## Details\n\nMore text."
    )
    assert rule.stats.units_hit == 1
    assert rule.stats.chars_removed == len("# Add\n\n")


def test_title_echo_ignores_case_and_spacing() -> None:
    rule = ChmTitleEcho()
    (out,) = rule([doc("#   ADD  \n\nBody here.")])
    assert texts([out]) == ["Body here."]


def test_title_echo_keeps_a_heading_that_says_more_than_the_title() -> None:
    body = "# Queue::Add\n\nAdd a widget."
    rule = ChmTitleEcho()
    assert texts(rule([doc(body)])) == [body]  # "Queue::Add" is not "Add"
    assert rule.stats.units_hit == 0


def test_title_echo_only_touches_the_first_heading_of_the_page() -> None:
    # Two overloads of one method: both sections carry the method's name as heading.
    body = "# Add\n\nFirst form.\n\n# Add\n\nSecond form."
    (out,) = ChmTitleEcho()([doc(body)])
    assert texts([out]) == ["First form.\n\n# Add\n\nSecond form."]


def test_title_echo_keeps_other_levels_and_headings_that_are_not_first() -> None:
    rule = ChmTitleEcho()
    for body in ("## Add\n\nText.", "Intro.\n\n# Add\n\nText.", "Add\n\nText.", "- Add\n\nText."):
        assert texts(rule([doc(body)])) == [body], body
    assert rule.stats.units_hit == 0


def test_title_echo_needs_a_title_path_and_ignores_headings_inside_code() -> None:
    rule = ChmTitleEcho()
    assert texts(rule([doc("# Add\n\nText.", title_path="")])) == ["# Add\n\nText."]
    fenced = "```\n# Add\n```\n\nText."
    assert texts(rule([doc(fenced)])) == [fenced]


def test_title_echo_works_on_chinese_titles() -> None:
    (out,) = ChmTitleEcho()(
        [doc("# 添加\n\n把部件添加到队列。", title_path="参考 > 队列 > 添加", language="zh")]
    )
    assert texts([out]) == ["把部件添加到队列。"]


def test_title_echo_can_be_switched_off_and_is_registered() -> None:
    rule = ChmTitleEcho(enabled=False)
    assert texts(rule([doc("# Add\n\nText.")])) == ["# Add\n\nText."]
    assert "chm_title_echo" in steps


# --- dedupe(group_by) --------------------------------------------------------------------------


def test_dedupe_group_by_language_keeps_the_same_text_in_both_languages() -> None:
    rule = Dedupe(group_by=["language"])
    kept = rule(
        [
            doc("Same code only.", language="en", name="a"),
            doc("Same code only.", language="zh", name="a"),
            doc("Same code only.", language="en", name="b"),
        ]
    )
    assert [d.node_id for d in kept] == ["mt5:en/a.htm", "mt5:zh/a.htm"]
    assert rule.stats.skipped == {"mt5:en/b.htm": "duplicate"}
    assert rule.stats.duplicate_of == {"mt5:en/b.htm": "mt5:en/a.htm"}


def test_dedupe_group_by_title_path_keeps_equal_text_under_different_titles() -> None:
    rule = Dedupe(group_by=["language", "title_path"])
    kept = rule(
        [
            doc("Same text.", title_path="Admin API > Clients", name="admin_clients"),
            doc("Same text.", title_path="Manager API > Clients", name="manager_clients"),
            doc("Same text.", title_path="Manager API > Clients", name="manager_clients_again"),
        ]
    )
    assert [d.node_id for d in kept] == ["mt5:en/admin_clients.htm", "mt5:en/manager_clients.htm"]
    assert rule.stats.duplicate_of == {
        "mt5:en/manager_clients_again.htm": "mt5:en/manager_clients.htm"
    }


def test_dedupe_without_group_by_still_collapses_across_languages() -> None:
    kept = Dedupe()([doc("Same.", language="en", name="a"), doc("Same.", language="zh", name="a")])
    assert [d.node_id for d in kept] == ["mt5:en/a.htm"]  # the smallest doc_id, as before


def test_dedupe_group_by_is_a_yaml_parameter() -> None:
    rule = steps.create("dedupe", group_by=["language"])
    assert isinstance(rule, Dedupe) and rule.group_by == ["language"]
    assert Dedupe().group_by == []


# --- the removal guard (design s6-3) -----------------------------------------------------------


def echo_page(body_chars: int, *, name: str = "p") -> Document:
    return doc("# Add\n\n" + "x" * body_chars, name=name)


def test_guard_is_off_by_default() -> None:
    rule = ChmTitleEcho()
    rule([echo_page(1)])  # 7 of 8 characters are removed: nobody complains without a limit


def test_guard_trips_when_one_page_loses_too_large_a_share() -> None:
    rule = ChmTitleEcho(max_page_share=0.05, guard_min_chars=50)
    with pytest.raises(RemovalGuardError) as err:
        rule([echo_page(8, name="small_ok"), echo_page(60, name="big_loss")])  # 7/67 = 10.4%
    exc = err.value
    assert (exc.rule, exc.scope, exc.doc_id) == ("chm_title_echo", "page", "mt5:en/big_loss.htm")
    assert exc.share == pytest.approx(7 / 67)
    assert exc.limit == 0.05
    message = str(exc)
    assert "chm_title_echo" in message and "mt5:en/big_loss.htm" in message
    assert "10.4%" in message and "5.0%" in message
    assert "xxxx" not in message  # never page text


def test_guard_page_share_passes_below_the_limit_and_for_pages_under_the_minimum() -> None:
    rule = ChmTitleEcho(max_page_share=0.1, guard_min_chars=50)
    rule([echo_page(200)])  # 7/207 = 3%
    rule([echo_page(3)])  # 70% of a 10-character page, but the page is below guard_min_chars
    assert rule.stats.units_hit == 2


def test_guard_trips_when_the_corpus_loses_too_large_a_share() -> None:
    def pages() -> list[Document]:
        return [echo_page(60, name=f"p{i}") for i in range(10)]  # 10 x 7 of 67 = 10.4%

    rule = ChmTitleEcho(max_corpus_share=0.05, guard_min_chars=10_000)
    with pytest.raises(RemovalGuardError) as err:
        rule(pages())
    assert (err.value.scope, err.value.doc_id) == ("corpus", None)
    assert err.value.share == pytest.approx(70 / 670)
    ChmTitleEcho(max_corpus_share=0.2)(pages())  # under the limit: passes


def test_guard_counts_dropped_pages_in_the_corpus_share() -> None:
    big = doc("same " * 100, name="a")
    twin = doc("same " * 100, name="b")
    other = doc("other text", name="c")
    with pytest.raises(RemovalGuardError, match="corpus"):
        Dedupe(max_corpus_share=0.3)([big, twin, other])
    assert len(Dedupe(max_corpus_share=0.6)([big, twin, other])) == 2


def test_guard_ignores_a_disabled_rule() -> None:
    ChmTitleEcho(enabled=False, max_page_share=0.01)([echo_page(1)])


def test_guard_also_watches_the_generic_rules() -> None:
    padded = doc("a b" + " \n" * 30 + "c d")
    with pytest.raises(RemovalGuardError, match="whitespace"):
        Whitespace(max_page_share=0.3, guard_min_chars=10)([padded])
    toc = doc("- [a](a.htm)\n- [b](b.htm)\n" * 30, name="toc")  # dropped as a table of contents
    real = doc("A real sentence with enough words to count as content.", name="real")
    with pytest.raises(RemovalGuardError, match="skip_empty"):
        SkipEmpty(max_corpus_share=0.5)([toc, real])
    assert len(SkipEmpty(max_corpus_share=0.95)([toc, real])) == 1


def test_guard_limits_are_checked_when_the_rule_is_built() -> None:
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ChmTitleEcho(max_page_share=bad)
        with pytest.raises(ValueError):
            ChmTitleEcho(max_corpus_share=bad)


def test_pipeline_version_ignores_the_guard_limits_but_not_other_parameters() -> None:
    """A limit decides whether a run stops, not what it writes: tuning it must not make every
    page look changed (the version is part of every page's hash, design s5-4)."""

    def version(clean: list[Any]) -> str:
        data = {
            **CHAIN_KB,
            "pipelines": {".htm": {**CHAIN_KB["pipelines"][".htm"], "clean": clean}},
        }
        return pipeline_version(
            KnowledgeBaseConfig.model_validate(data), ".htm", tokenizer_model="m"
        )

    plain = ["whitespace", {"skip_empty": {"min_chars": 20}}]
    guarded = [
        {"whitespace": {"max_page_share": 0.3, "max_corpus_share": 0.05}},
        {"skip_empty": {"min_chars": 20, "max_corpus_share": 0.02, "guard_min_chars": 50}},
    ]
    retuned = [
        {"whitespace": {"max_page_share": 0.9}},
        {"skip_empty": {"min_chars": 20, "max_corpus_share": 0.5}},
    ]
    assert version(plain) == version(guarded) == version(retuned)
    assert version(plain) != version(["whitespace", {"skip_empty": {"min_chars": 30}}])
    assert version(plain) != version(["whitespace", {"skip_empty": {"min_chars": 20}}, "dedupe"])


# --- the whole chain: text that looks like markup survives (design s6-1) ----------------------


CHAIN_KB: dict[str, Any] = {
    "kb": "mt5",
    "source": {"connector": "test_list"},
    "language": "auto",
    "pipelines": {
        ".htm": {
            "reader": "chm",
            "clean": [
                "whitespace",
                "chm_title_echo",
                {"skip_empty": {"min_chars": 20}},
                {"dedupe": {"group_by": ["language"]}},
            ],
            "steps": [
                "markdown_split",
                {"token_split": {"max_tokens": 60}},
                {"merge_tiny": {"min_tokens": 3}},
                "assign_chunk_ids",
            ],
        }
    },
}


def words(text_: str) -> list[int]:
    return list(range(len(text_.split())))


def test_chain_keeps_angle_brackets_in_prose() -> None:
    cfg = KnowledgeBaseConfig.model_validate(CHAIN_KB)
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    prose = (
        "A value &lt; 0 means failure, a value &gt; 9 is invalid, use Vec&lt;T&gt; for lists "
        "and write &lt;b&gt;bold&lt;/b&gt; &amp; &lt;i&gt;italic&lt;/i&gt; as plain text."
    )
    html = page_html(h1("Add"), text(prose), text("Call &lt;widget_id&gt; first."))
    source = SourceDocument(
        doc_id=make_doc_id("mt5", "en/queue_add.htm"),
        kb="mt5",
        source_rel="en/queue_add.htm",
        source_url=None,
        ext=".htm",
        content=html.encode("cp1252"),
        source_version="v1",
        extra={"language": "en", "page_key": "queue_add", "toc_path": ["Queue", "Add"]},
    )
    read = proc.read([source])
    assert not read.errors
    (kept,) = proc.clean(read.documents)
    expected = (
        "A value < 0 means failure, a value > 9 is invalid, use Vec<T> for lists "
        "and write <b>bold</b> & <i>italic</i> as plain text.\n\nCall <widget_id> first."
    )
    assert kept.get_content(MetadataMode.NONE) == expected  # only the title echo is gone
    chunk_text = " ".join(n.get_content(MetadataMode.NONE) for n in proc.chunk([kept]))
    for fragment in ("A value < 0", "a value > 9", "Vec<T>", "<b>bold</b>", "<widget_id>"):
        assert fragment in chunk_text
