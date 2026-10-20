# ruff: noqa: RUF001
"""Synthetic CHM-like manuals for tests (spec P4-01): invented sentences, never manual text.

The pages copy the *shape* of what a help-authoring tool writes (a navigation table, then a
`div#help` with classed paragraphs, code in bordered one-cell tables, `&nbsp;` for blanks and
indentation) about an invented product, the Frobnitz SDK. `write_volume` writes a folder in
the layout `7z x file.chm` produces (pages, `.hhc`, `.hhk`, `#SYSTEM`, CHM internals).
"""

from __future__ import annotations

import html
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CODE_BOX = (
    '<div style="border-style: solid;"><table cellspacing="0"><tr><td>'
    '<p class="p_CodeExample">{lines}</p></td></tr></table></div>'
)


def esc(text: str) -> str:
    """HTML-escape text, writing spaces as `&nbsp;` the way the authoring tool does in code."""
    return html.escape(text, quote=True).replace(" ", "&nbsp;")


def h1(text: str) -> str:
    return f'<p class="p_H1"><span class="f_H1">{html.escape(text)}</span></p>'


def h2(text: str) -> str:
    return f'<p class="p_H2"><span class="f_H2">{html.escape(text)}</span></p>'


def h3(text: str) -> str:
    return f'<p class="p_H3"><span class="f_H3">{html.escape(text)}</span></p>'


def text(body: str) -> str:
    """A body paragraph; `body` is HTML (use `link`, `html.escape` for literal `<`)."""
    return f'<p class="p_Text"><span class="f_Text">{body}</span></p>'


def short(body: str) -> str:
    return f'<p class="p_Function"><span class="f_Function">{body}</span></p>'


def bold(label: str) -> str:
    return f'<p class="p_BoldTitles"><span class="f_BoldTitles">{html.escape(label)}</span></p>'


def param(name: str, description: str) -> str:
    return (
        f'<p class="p_FunctionParameter"><span class="f_FunctionParameter">{html.escape(name)}'
        f'</span></p><p class="p_ParameterDesrciption"><span class="f_ParameterDesrciption">'
        f"{description}</span></p>"
    )


def link(href: str, label: str) -> str:
    return f'<a href="{href}" class="topiclink">{html.escape(label)}</a>'


def anchor(name: str) -> str:
    return f'<a name="{name}" class="topiclink"></a>'


def code(*lines: str) -> str:
    """A code box: every line with `&nbsp;` for spaces, lines separated by `<br>`."""
    inner = "\n<br>".join(f'<span class="f_CodeExample">{esc(line)}</span>' for line in lines)
    return CODE_BOX.format(lines=inner)


def note(body: str) -> str:
    return (
        '<div><table class="attentable" border="1"><tr class="attentable"><td class="attentable">'
        f'<p class="p_tableatten"><span class="f_tableatten">{body}</span></p>'
        "</td></tr></table></div>"
    )


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    def cell(tag: str, value: str) -> str:
        return f'<{tag}><p class="p_fortable"><span class="f_fortable">{value}</span></p></{tag}>'

    head = "".join(cell("th", c) for c in header)
    body = "".join("<tr>" + "".join(cell("td", c) for c in row) + "</tr>" for row in rows)
    return (
        f'<div><table class="table" border="1"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def bullets(*items: str) -> str:
    inner = "".join(f'<li class="p_li"><span class="f_li">{i}</span></li>' for i in items)
    return f'<ul style="list-style-type:disc">{inner}</ul>'


def numbered(*items: str) -> str:
    inner = "".join(f'<li class="p_li"><span class="f_li">{i}</span></li>' for i in items)
    return f'<ol style="list-style-type:decimal">{inner}</ol>'


def image(src: str, alt: str = "") -> str:
    return f'<p class="p_image"><img class="help" alt="{html.escape(alt)}" src="{src}"/></p>'


def empty_paragraph() -> str:
    return '<p class="p_Text"><span class="f_Text">&nbsp;</span></p>'


def page_html(
    *body: str,
    title: str = "Page",
    charset: str | None = "iso-8859-1",
    crumbs: Sequence[str] = ("Reference", "Widgets"),
    prev: str | None = "prev_page.htm",
    next: str | None = "next_page.htm",
) -> str:
    """A whole page: head, navigation table (breadcrumbs, previous/next images), `div#help`."""
    meta = (
        f'<meta http-equiv="Content-Type" content="text/html; charset={charset}" />'
        if charset
        else ""
    )
    trail = " / ".join(f'<a class="h_m" href="x.htm"> {html.escape(c)} </a>' for c in crumbs)
    arrows = ""
    if prev:
        arrows += f'<a href="{prev}"><img src="previous.png" alt="Previous" width="27"></a>&nbsp;'
    if next:
        arrows += f'<a href="{next}"><img src="next.png" alt="Next" width="27"></a>'
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        f"{meta}\n<title>{html.escape(title)}</title>\n"
        '<link type="text/css" href="default.css" rel="stylesheet" />\n'
        '<script type="text/javascript" src="jquery.js"></script>\n</head>\n'
        '<body style="margin:0px;">\n'
        '<table width="100%"><tr><td class="nav">'
        f'{trail}</td><td align="right">{arrows}</td></tr></table>\n'
        '<div id="help">\n' + "\n".join(body) + "\n</div>\n</body>\n</html>\n"
    )


TocTree = Sequence[tuple[str, str | None, "TocTree"]]
"""`(name, local, children)`; `local` None for a folder without a page."""


def hhc(tree: TocTree) -> str:
    """A table-of-contents file in the HHC shape: nested `<UL>`, unclosed `<LI>`, params."""

    def walk(nodes: TocTree) -> str:
        out = ["<UL>"]
        for name, local, children in nodes:
            out.append('<LI><OBJECT type="text/sitemap">')
            out.append(f'        <param name="Name" value="{name}">')
            if local is not None:
                out.append(f'        <param name="Local" value="{local}">')
            out.append("</OBJECT>\n")
            if children:
                out.append(walk(children))
        out.append("</UL>")
        return "\n".join(out)

    return (
        '<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">\n<HTML>\n<BODY>\n'
        '<OBJECT type="text/site properties">\n\t<param name="Window Styles" value="0x800025">\n'
        "</OBJECT>\n" + walk(tree) + "\n</BODY></HTML>\n"
    )


def hhk(entries: Iterable[tuple[str, Sequence[str]]]) -> str:
    """An index file: keyword -> pages it points to."""
    out = ['<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">\n<HTML>\n<BODY>\n<UL>']
    for keyword, locals_ in entries:
        out.append('<LI><OBJECT type="text/sitemap">')
        out.append(f'    <param name="Name" value="{keyword}">')
        for local in locals_:
            out.append(f'    <param name="Local" value="{local}">')
        out.append("    </OBJECT></LI>")
    out.append("</UL>\n</BODY></HTML>\n")
    return "\n".join(out)


def system_file(lcid: int, title: str = "Frobnitz SDK - User Guide") -> bytes:
    """A `#SYSTEM` file with the entries that matter: locale (4), title (3), default page (2)."""

    def entry(code_: int, data: bytes) -> bytes:
        return struct.pack("<HH", code_, len(data)) + data

    return (
        struct.pack("<I", 3)
        + entry(4, struct.pack("<I", lcid) + b"\x00" * 4)
        + entry(2, b"index.htm\x00")
        + entry(3, title.encode("ascii") + b"\x00")
    )


def write_volume(
    root: Path,
    pages: dict[str, bytes],
    *,
    toc: tuple[str, str] | None = None,
    index: tuple[str, str] | None = None,
    toc_encoding: str = "cp1252",
    lcid: int | None = None,
    internals: bool = True,
) -> Path:
    """Write `pages` (name -> raw bytes) and the optional `(file name, text)` toc and index
    next to CHM internals, as an unpacked CHM folder."""
    root.mkdir(parents=True, exist_ok=True)
    for name, raw in pages.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if toc:
        (root / toc[0]).write_bytes(toc[1].encode(toc_encoding))
    if index:
        (root / index[0]).write_bytes(index[1].encode(toc_encoding))
    if internals:
        (root / "#SYSTEM").write_bytes(system_file(lcid) if lcid else b"")
        (root / "$WWKeywordLinks").mkdir(exist_ok=True)
        (root / "$WWKeywordLinks" / "BTree").write_bytes(b"\x00")
        (root / "default.css").write_text("body {}", encoding="utf-8", newline="")
        (root / "previous.png").write_bytes(b"\x89PNG")
    return root


def demo_page(title: str, *body: str, charset: str = "iso-8859-1") -> str:
    return page_html(h1(title), *body, title=title, charset=charset)


def demo_en_volume(root: Path) -> Path:
    same = text("Frobnicate the gizmo, then stop. This text is the same on two pages.")
    pages = {
        "a.htm": demo_page(
            "A Page", text("Frobnicate the gizmo carefully. " + link("b.htm", "Read page B") + ".")
        ),
        "b.htm": demo_page("B Page", same),
        "b_copy.htm": demo_page("B Page", same),
        "empty.htm": page_html(empty_paragraph()),
        "only_en.htm": demo_page("Only English", text("The sprocket has no translation.")),
        "orphan.htm": demo_page("Orphan", text("ZETA-9000 is not in the contents.")),
    }
    tree = [
        ("A", "a.htm", []),
        ("B", "b.htm", []),
        ("B copy", "b_copy.htm", []),
        ("Empty", "empty.htm", []),
        ("Only English", "only_en.htm", []),
        ("Gone", "gone.htm", []),  # in the contents, no file
    ]
    return write_volume(
        root,
        {k: v.encode("cp1252") for k, v in pages.items()},
        toc=("api.hhc", hhc(tree)),
        lcid=0x409,
    )


def demo_zh_volume(root: Path) -> Path:
    pages = {
        "a.htm": demo_page(
            "A Page",
            text(
                "闪电 Frobnicate 部件，然后把结果写回到队列里，最后返回成功。"
                + link("b.htm", "参见乙")
            ),
            charset="gb2312",
        ),
        "b.htm": demo_page(
            "B Page",
            text("第二页的内容，说明 gizmo 怎么用，以及什么时候不能用它。"),
            charset="gb2312",
        ),
        "lie.htm": demo_page(
            "Lie",
            text("这页的字符集声明是错的，但是内容可以正常读出来，闪电。"),
            charset="iso-8859-1",
        ),
    }
    tree = [("甲", "a.htm", []), ("乙", "b.htm", []), ("丙", "lie.htm", [])]
    return write_volume(
        root,
        {k: v.encode("gb18030") for k, v in pages.items()},
        toc=("api_chinese.hhc", hhc(tree)),
        toc_encoding="gb18030",
        lcid=0x804,
    )


# --- a small invented bilingual manual for tests of the reader, report and ingest job ------------

DEMO_WORDS = ["Frobnicate", "gizmo", "sprocket", "ZETA-9000", "闪电"]
"""Distinctive words of the demo pages: a report or log that holds one holds page text."""


def demo_manual(root: Path) -> dict[str, Path]:
    """Write the demo manual: an English volume (6 pages: a normal one, b and a copy of b, an
    empty page, one without a Chinese version, one missing from the contents, and a contents
    entry without a file) and a Chinese one (a, b and a page whose charset label lies)."""
    return {"en": demo_en_volume(root / "en"), "zh": demo_zh_volume(root / "zh")}


def demo_kb_dict(
    volumes: dict[str, Path], *, max_tokens: int = 40, **overrides: Any
) -> dict[str, Any]:
    """KB config (as a dict for `KnowledgeBaseConfig.model_validate`) over the demo volumes."""
    data: dict[str, Any] = {
        "kb": "mt5",
        "source": {
            "connector": "chm_dir",
            "volumes": [
                {"language": "zh", "path": str(volumes["zh"])},
                {"language": "en", "path": str(volumes["en"])},
            ],
        },
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
                    {"token_split": {"max_tokens": max_tokens}},
                    {"merge_tiny": {"min_tokens": 3}},
                    "assign_chunk_ids",
                ],
            }
        },
    }
    data.update(overrides)
    return data
