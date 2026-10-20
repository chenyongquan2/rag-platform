# ruff: noqa: RUF001
"""CHM page decoding (spec P4-01 AC1, design s6-2 encoding repair, s5-2).

The real manual declares `iso-8859-1` for the English pages (which are windows-1252: curly
quotes and dashes in bytes 0x92 to 0x97) and `gb2312` for the Chinese ones (which need the
GB18030 superset). These tests use invented text only.
"""

from __future__ import annotations

import pytest

from rag.data.chm import declared_charset, decode_html

ZH = "插件接口：添加一个队列元素，返回值为真表示成功。"


def page(body: str, charset: str | None) -> str:
    meta = (
        f'<meta http-equiv="Content-Type" content="text/html; charset={charset}" />'
        if charset
        else ""
    )
    return f"<html><head>{meta}<title>t</title></head><body><p>{body}</p></body></html>"


def test_declared_charset_reads_meta_variants() -> None:
    cases = {
        b'<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1" />': "iso-8859-1",
        b'<META HTTP-EQUIV="content-type" CONTENT="text/html; charset=GB2312">': "gb2312",
        b'<meta charset="utf-8">': "utf-8",
        b"<meta charset=utf-8>": "utf-8",
        b"<html><head></head></html>": None,
    }
    for raw, expected in cases.items():
        assert declared_charset(raw) == expected, raw


def test_declared_charset_only_looks_at_the_head() -> None:
    late = b"<html><head></head><body>" + b"x" * 5000 + b'<meta charset="utf-8"></body></html>'
    assert declared_charset(late) is None


def test_decode_ascii_page_needs_no_guess() -> None:
    raw = page("Add an element.", "iso-8859-1").encode("ascii")
    out = decode_html(raw)
    assert out.text == raw.decode("ascii")
    assert out.encoding == "ascii"
    assert out.declared == "iso-8859-1"
    assert out.fallback is False
    assert out.replaced == 0


def test_decode_gbk_page_with_honest_charset() -> None:
    raw = page(ZH, "gb2312").encode("gb18030")
    out = decode_html(raw)
    assert ZH in out.text
    assert out.encoding == "gb18030"
    assert out.declared == "gb2312"
    assert out.fallback is False  # gb2312 -> gb18030 is the standard alias, not a guess


@pytest.mark.parametrize("lie", ["iso-8859-1", "utf-8", "windows-1252", None])
def test_decode_gbk_page_whose_charset_lies(lie: str | None) -> None:
    raw = page(ZH * 3, lie).encode("gb18030")
    for hint in (None, "gb18030"):
        out = decode_html(raw, hint=hint)
        assert ZH * 3 in out.text, (lie, hint)
        assert out.encoding == "gb18030"
        assert out.fallback is (lie is not None)
        assert out.replaced == 0


def test_decode_cp1252_page_declared_latin1() -> None:
    body = "It’s “done” — really"
    raw = page(body, "iso-8859-1").encode("cp1252")
    out = decode_html(raw)
    assert body in out.text
    assert out.encoding == "cp1252"
    assert out.fallback is False  # latin-1 labels are read as windows-1252 (HTML standard)


def test_decode_cp1252_page_that_is_also_valid_gbk() -> None:
    # 0x97 (em dash) followed by an ASCII letter is a legal GBK pair: the guess must not flip.
    body = "Add—Remove"
    raw = page(body, "iso-8859-1").encode("cp1252")
    assert raw.decode("gb18030")  # the trap exists
    for hint in (None, "cp1252"):
        out = decode_html(raw, hint=hint)
        assert body in out.text
        assert out.encoding == "cp1252"


def test_decode_utf8_page_declared_gb2312() -> None:
    raw = page(ZH, "gb2312").encode("utf-8")
    out = decode_html(raw, hint="gb18030")
    assert ZH in out.text
    assert out.encoding == "utf-8"
    assert out.fallback is True


def test_decode_utf8_is_tried_before_the_declared_charset() -> None:
    # Strict GB18030 also accepts these UTF-8 bytes, and reads them as garbage: only trying UTF-8
    # first gets the page right.
    text = "中文中文"
    raw = page(text, "gb2312").encode("utf-8")
    assert raw.decode("gb18030", errors="strict") != text  # the trap exists
    for hint in (None, "gb18030"):
        out = decode_html(raw, hint=hint)
        assert text in out.text and out.encoding == "utf-8", hint
        assert out.fallback is True


def test_decode_rejects_a_reading_with_control_characters_or_replacement_characters() -> None:
    # Valid UTF-8, but the text holds a C1 control character: that reading is not trusted and
    # the page is read as windows-1252 (which cannot produce a control character).
    raw = page("x \u0085 y é", "iso-8859-1").encode("utf-8")
    out = decode_html(raw)
    assert out.encoding == "cp1252" and "\u0085" not in out.text
    # Same for a literal replacement character in otherwise valid UTF-8.
    raw = page("ok � é", "iso-8859-1").encode("utf-8")
    out = decode_html(raw)
    assert out.encoding == "cp1252" and out.text.count("�") == 0


def test_decode_utf8_bom_is_removed() -> None:
    raw = b"\xef\xbb\xbf" + page(ZH, "utf-8").encode("utf-8")
    out = decode_html(raw)
    assert not out.text.startswith("﻿")
    assert ZH in out.text
    assert out.fallback is False


def test_decode_unknown_declared_charset_is_ignored() -> None:
    raw = page(ZH, "x-no-such-charset").encode("gb18030")
    out = decode_html(raw, hint="gb18030")
    assert ZH in out.text
    assert out.fallback is True


def test_decode_undecodable_bytes_are_replaced_and_counted() -> None:
    # 0x81 0x20 is not GB18030, 0x81 is undefined in windows-1252, and it is not UTF-8.
    raw = b"<html><head></head><body>ok \x81\x20 and \x81\x81\x81 end</body></html>"
    out = decode_html(raw)
    assert out.replaced > 0
    assert out.text.count("�") == out.replaced
    assert "ok" in out.text and "end" in out.text
    assert out.fallback is True


def test_decode_hint_does_not_beat_a_clean_declared_charset() -> None:
    raw = page("Café à la carte", "iso-8859-1").encode("cp1252")
    out = decode_html(raw, hint="gb18030")
    assert "Café à la carte" in out.text
    assert out.encoding == "cp1252"
