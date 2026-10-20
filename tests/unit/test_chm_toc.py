"""Table of contents (.hhc), index (.hhk) and page keys (spec P4-01 AC2, AC3; design s5-4, s8-4)."""

from __future__ import annotations

import pytest

from rag.data.chm import page_key_of, parse_index, parse_toc
from rag.testing.chm import hhc, hhk

TREE = [
    ("Frobnitz SDK", "index.htm", []),
    (
        "Getting Started",
        "start.htm",
        [
            ("Requirements", "req.htm", []),
            ("Setup &amp; Removal", "setup.htm", [("Silent Mode", "setup_silent.htm#modes", [])]),
        ],
    ),
    ("Folder Without Page", None, [("Child", "child.htm", [])]),
    ("Reference", "ref.htm", [("Queue", "queue.htm", [("Add", "queue_add.htm", [])])]),
]


def test_parse_toc_builds_title_paths_from_nesting() -> None:
    entries = parse_toc(hhc(TREE))
    by_local = {e.local: e for e in entries if e.local}
    assert by_local["index.htm"].path == ("Frobnitz SDK",)
    assert by_local["req.htm"].path == ("Getting Started", "Requirements")
    assert by_local["queue_add.htm"].path == ("Reference", "Queue", "Add")
    assert by_local["queue_add.htm"].depth == 3
    assert by_local["start.htm"].depth == 1


def test_parse_toc_keeps_folders_without_a_page_as_ancestors() -> None:
    entries = parse_toc(hhc(TREE))
    folder = next(e for e in entries if e.name == "Folder Without Page")
    assert folder.local is None
    child = next(e for e in entries if e.local == "child.htm")
    assert child.path == ("Folder Without Page", "Child")


def test_parse_toc_unescapes_names_and_keeps_anchors_in_local() -> None:
    entries = {e.local: e for e in parse_toc(hhc(TREE)) if e.local}
    assert entries["setup.htm"].name == "Setup & Removal"
    assert entries["setup_silent.htm#modes"].path == (
        "Getting Started",
        "Setup & Removal",
        "Silent Mode",
    )


def test_parse_toc_siblings_after_a_deep_branch_return_to_the_right_level() -> None:
    entries = {e.local: e for e in parse_toc(hhc(TREE)) if e.local}
    assert entries["ref.htm"].path == ("Reference",)
    assert entries["ref.htm"].depth == 1


def test_parse_toc_reads_chinese_names_and_upper_case_markup() -> None:
    text = (
        '<HTML><BODY><UL><LI><OBJECT TYPE="text/sitemap">'
        '<PARAM NAME="Name" VALUE="开始"><PARAM NAME="Local" VALUE="start.htm"></OBJECT>'
        '<UL><LI><OBJECT type="text/sitemap"><param name="Name" value="系统要求">'
        '<param name="Local" value="req.htm"></OBJECT></UL></UL></BODY></HTML>'
    )
    entries = parse_toc(text)
    assert [e.path for e in entries] == [("开始",), ("开始", "系统要求")]


def test_parse_toc_returns_duplicates_as_they_are() -> None:
    tree = [("A", "same.htm", []), ("B", "same.htm", [])]
    assert [e.local for e in parse_toc(hhc(tree))] == ["same.htm", "same.htm"]


def test_parse_toc_empty_or_foreign_text_gives_nothing() -> None:
    assert parse_toc("") == []
    assert parse_toc("<html><body>no list here</body></html>") == []


def test_parse_index_maps_page_keys_to_keywords() -> None:
    text = hhk(
        [
            ("Add", ["queue_add.htm", "list_add.htm"]),
            ("Queue", ["queue.htm"]),
            ("Add", ["queue_add.htm"]),  # again: no duplicate keyword for the page
            ("See only", []),  # no target: ignored
        ]
    )
    index = parse_index(text)
    assert index["queue_add"] == ["Add"]
    assert index["list_add"] == ["Add"]
    assert index["queue"] == ["Queue"]
    assert "See only" not in {k for ks in index.values() for k in ks}


@pytest.mark.parametrize(
    ("local", "key"),
    [
        ("queue_add.htm", "queue_add"),
        ("Queue_Add.HTM", "queue_add"),
        ("sub/Page.html", "sub/page"),
        (".\\sub\\Page.htm", "sub/page"),
        ("/queue_add.htm#top", "queue_add"),
        ("queue%5Fadd.htm", "queue_add"),
        ("%E4%B8%AD%E6%96%87.htm", "中文"),
        ("中文.htm", "中文"),
        ("e\u0301cole.htm", "\u00e9cole"),  # NFC: the same name in two forms is one key
        ("mk:@MSITStore:C:\\x\\api.chm::/queue_add.htm", "queue_add"),
        ("ms-its:api.chm::/queue_add.htm#a", "queue_add"),
        ("logo.png", "logo.png"),
    ],
)
def test_page_key_normalizes_case_slashes_encoding_anchors_and_names(local: str, key: str) -> None:
    assert page_key_of(local) == key


@pytest.mark.parametrize(
    "external",
    [
        "http://example.com/a.htm",
        "https://example.com",
        "mailto:support@example.com",
        "javascript:void(0)",
        "",
        "#top",
    ],
)
def test_page_key_of_external_or_empty_targets_is_none(external: str) -> None:
    assert page_key_of(external) is None
