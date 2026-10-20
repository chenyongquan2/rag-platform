"""HTML page of a help-authoring tool (CHM) -> Markdown (spec P4-01, design s5-2, s6-2).

The conversion works on the parsed page (a DOM), never on text with the tags cut out by regular
expressions: that is how the first experiments with this manual lost body text (`&lt;` turned
back into `<` and then removed as if it were a tag, design s6-1). Text stays text.

What the page looks like (the MT5 manual, authored with a tool that writes classed paragraphs):
a navigation table (breadcrumbs, previous/next images), then `<div id="help">` with `p_H1`..`p_H3`
headings, `p_Text`/`p_Function`/`p_FunctionRemark` paragraphs, `p_BoldTitles` labels, name and
description paragraphs for parameters, code in bordered one-cell tables (`p_CodeExample`, blanks
and indentation written as `&nbsp;`), data tables, note boxes (`attentable`), lists and images.
The rules below are written for that shape, and every piece of noise they drop is counted
(`Converted.removed`) so the quality report can show it:

- `nav_bar`: everything in `<body>` outside `div#help` (kept: a table *inside* the help text,
  whatever its words are);
- `script_style`: `script`, `style`, `noscript` inside the help text;
- `empty_paragraph`: paragraphs with nothing but blanks (and no image);
- `anchor`: `<a name>` targets without text.

Output: headings as `#`, code as fenced blocks (the fence is longer than any run of backticks in
the code), tables as Markdown tables (a header row is made when the table has none), note boxes
as quotes, lists as `-` / `1.`, a parameter's name and description as one list item, images as
`[image: caption]` (the file is listed in `images`), internal links as their text (the target
page keys in `links_to`), external links as `[text](url)`.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import unquote

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from rag.data.chm.toc import page_key_of

_WS = re.compile(r"[\s​]+")
_SOURCE_WS = re.compile(r"[ \t\r\n]+")
_BACKTICKS = re.compile(r"`+")
_LEADING_HASH = re.compile(r"^#{1,6}\s")
_EXTERNAL = re.compile(r"^https?://", re.I)
_NON_PAGE = re.compile(r"\.(?!html?$)[a-z0-9]{1,5}$", re.I)
_DROPPED_TAGS = ("script", "style", "noscript")
_HEADING_CLASSES = {"p_H1": 1, "p_H2": 2, "p_H3": 3}
_CODE_CLASS = "p_CodeExample"
_BLOCK_TAGS = frozenset({"p", "div", "table", "ul", "ol", "pre", "hr"})


@dataclass
class Converted:
    markdown: str
    heading: str | None = None
    """Text of the first level-1 heading."""
    html_title: str | None = None
    images: list[str] = field(default_factory=list)
    """Image files of the page, relative to the CHM's root, once each, in page order."""
    links_to: list[str] = field(default_factory=list)
    """Page keys of the pages this page links to (not itself), once each, in page order."""
    internal_links: int = 0
    external_links: int = 0
    image_markers: int = 0
    tables: int = 0
    code_blocks: int = 0
    has_container: bool = True
    visible_chars: int = 0
    """Letters and digits of the page's visible text (scripts and styles left out)."""
    lost_chars: int = 0
    """Of those, the ones that did not reach the Markdown (see `lost_characters`)."""
    removed: dict[str, tuple[int, int]] = field(default_factory=dict)
    """Noise rule -> (elements dropped, visible characters dropped)."""


_WORD_CHARS = re.compile(r"\w")
_IMAGE_MARKER = re.compile(r"\[image: [^\]]*\]")
_LINK_TARGET = re.compile(r"\]\(https?://[^)]*\)")


def lost_characters(visible: str, markdown: str) -> int:
    """Letters and digits (every Chinese character counts as one) of the page's visible text that
    are missing from its Markdown: a multiset comparison, so a word that moved or whose line
    breaks changed is not a loss, a word that vanished is. What the conversion adds (image
    markers, link targets) is taken out of the Markdown first. Design s6-3: a cleaning step must
    never lose body text without anyone noticing."""
    kept = _LINK_TARGET.sub("", _IMAGE_MARKER.sub("", markdown))
    missing = Counter(_WORD_CHARS.findall(visible)) - Counter(_WORD_CHARS.findall(kept))
    return sum(missing.values())


def norm(text: str) -> str:
    """Collapse every run of whitespace (non-breaking spaces included) into one space."""
    return _WS.sub(" ", text).strip()


def fence_for(code: str) -> str:
    """A code fence longer than every run of backticks inside `code` (at least three)."""
    longest = max((len(m.group(0)) for m in _BACKTICKS.finditer(code)), default=0)
    return "`" * max(3, longest + 1)


def _is_page_target(href: str) -> bool:
    """A link to a page (no extension, or `.htm`/`.html`), not to an image or another file."""
    path = unquote(href).split("#", 1)[0].split("?", 1)[0]
    return not _NON_PAGE.search(path)


class _Converter:
    def __init__(self, page_dir: str, self_key: str | None) -> None:
        self.page_dir = page_dir
        self.self_key = self_key
        self.blocks: list[tuple[str, str]] = []  # (kind, text); kind "item" joins with "\n"
        self.pending_param: str | None = None
        self.heading: str | None = None
        self.out = Converted(markdown="")

    # -- bookkeeping --------------------------------------------------------------------------

    def drop(self, rule: str, chars: int, elements: int = 1) -> None:
        old = self.out.removed.get(rule, (0, 0))
        self.out.removed[rule] = (old[0] + elements, old[1] + chars)

    def emit(self, kind: str, text: str) -> None:
        self.flush_param()
        self.blocks.append((kind, text))

    def flush_param(self) -> None:
        if self.pending_param is not None:
            name, self.pending_param = self.pending_param, None
            self.blocks.append(("item", f"- `{name}`"))

    def render(self) -> str:
        self.flush_param()
        parts: list[str] = []
        previous = ""
        for kind, text in self.blocks:
            if parts:
                parts.append("\n" if kind == previous == "item" else "\n\n")
            parts.append(text)
            previous = kind
        return "".join(parts).strip()

    # -- inline text --------------------------------------------------------------------------

    def inline(self, node: Tag, *, code: bool = False) -> str:
        parts: list[str] = []
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                # Source line breaks and runs of blanks are not significant in HTML, in code
                # too (its real blanks are `&nbsp;`); only `<br>` ends a line.
                parts.append(_SOURCE_WS.sub(" ", str(child)) if code else str(child))
            elif isinstance(child, Tag):
                parts.append(self.inline_tag(child, code=code))
        return "".join(parts)

    def inline_tag(self, tag: Tag, *, code: bool) -> str:
        name = tag.name
        if name == "br":
            return "\n"
        if name in _DROPPED_TAGS:
            return ""
        if name == "a":
            return self.link(tag, code=code)
        if name == "img":
            return "" if code else self.image(tag)
        if name in ("code", "tt") and not code:
            inner = norm(self.inline(tag))
            return f"`{inner}`" if inner else ""
        return self.inline(tag, code=code)

    def link(self, tag: Tag, *, code: bool) -> str:
        inner = self.inline(tag, code=code)
        href = tag.get("href")
        href = href.strip() if isinstance(href, str) else ""
        if not href:
            if not norm(inner):
                self.drop("anchor", len(str(tag.get("name") or "")))
                return ""
            return inner
        if _EXTERNAL.match(href):
            self.out.external_links += 1
            return inner if code or not norm(inner) else f"[{norm(inner)}]({href})"
        key = page_key_of(href, base=self.page_dir) if _is_page_target(href) else None
        if key is not None:
            self.out.internal_links += 1
            if key != self.self_key and key not in self.out.links_to:
                self.out.links_to.append(key)
        return inner

    def image(self, tag: Tag) -> str:
        src = str(tag.get("src") or "").strip()
        alt = norm(str(tag.get("alt") or ""))
        label = alt or posixpath.basename(unquote(src).split("?", 1)[0]) or "image"
        if src and not re.match(r"^[a-z][a-z0-9+.-]*:", src, re.I):
            path = posixpath.normpath(
                posixpath.join(self.page_dir, unquote(src).replace("\\", "/"))
            )
            if path and not path.startswith("..") and path not in self.out.images:
                self.out.images.append(path)
        self.out.image_markers += 1
        return f" [image: {label}] "

    # -- blocks -------------------------------------------------------------------------------

    def walk(self, parent: Tag) -> None:
        for child in parent.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                text = norm(str(child))
                if text:
                    self.emit("block", self.prose(text))
            elif isinstance(child, Tag):
                self.block(child)

    def block(self, el: Tag) -> None:
        name = el.name
        if name in _DROPPED_TAGS or name in ("hr", "br", "head"):
            return
        if name == "p":
            self.paragraph(el)
        elif name == "table":
            self.table(el)
        elif name in ("ul", "ol"):
            self.list_block(el, indent="")
        elif name == "pre":
            self.code_block(el.get_text())
        elif name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading_block(int(name[1]), norm(self.inline(el)))
        elif name in ("div", "section", "center", "blockquote", "body", "html", "tbody", "thead"):
            self.walk(el)
        else:  # span, font, b, a ... outside any paragraph: its text is a paragraph
            text = norm(self.inline_tag(el, code=False))
            if text:
                self.emit("block", self.prose(text))

    @staticmethod
    def prose(text: str) -> str:
        """Paragraph text; a leading `#` is escaped so it is not read as a heading later."""
        return "\\" + text if _LEADING_HASH.match(text) else text

    def heading_block(self, level: int, text: str) -> None:
        if not text:
            return
        if level == 1 and self.heading is None:
            self.heading = text
        self.emit("block", f"{'#' * level} {text}")

    def paragraph(self, el: Tag) -> None:
        classes = el.get("class") or []
        cls = next((c for c in classes if c.startswith("p_")), "")
        if cls == _CODE_CLASS:
            self.code_block(self.code_text(el))
            return
        text = norm(self.inline(el))
        if cls in _HEADING_CLASSES:
            self.heading_block(_HEADING_CLASSES[cls], text)
            return
        if not text:
            self.drop("empty_paragraph", len(el.get_text()))
            return
        if cls == "p_BoldTitles":
            self.emit("block", f"**{text}**")
        elif cls == "p_FunctionParameter":
            self.flush_param()
            self.pending_param = text
        elif cls.startswith("p_ParameterDesrciption") and self.pending_param is not None:
            name, self.pending_param = self.pending_param, None
            self.blocks.append(("item", f"- `{name}`: {text}"))
        else:
            self.emit("block", self.prose(text))

    # -- code ---------------------------------------------------------------------------------

    def code_text(self, el: Tag) -> str:
        text = self.inline(el, code=True)
        lines = [line.strip(" ").replace("\xa0", " ").rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip("\n")

    def code_block(self, code: str) -> None:
        code = code.rstrip()
        if not code.strip():
            return
        fence = fence_for(code)
        self.out.code_blocks += 1
        self.emit("block", f"{fence}\n{code}\n{fence}")

    # -- tables -------------------------------------------------------------------------------

    def table(self, el: Tag) -> None:
        paragraphs = el.find_all("p")
        classes = [(p.get("class") or [""])[0] for p in paragraphs]
        if paragraphs and all(c == _CODE_CLASS for c in classes):
            self.code_block("\n".join(self.code_text(p) for p in paragraphs))
            return
        if "attentable" in (el.get("class") or []):
            self.note_box(el)
            return
        rows = self.table_rows(el)
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = [_row(rows[0]), _row(["---"] * width)]
        lines += [_row(r) for r in rows[1:]]
        self.out.tables += 1
        self.emit("block", "\n".join(lines))

    def note_box(self, table: Tag) -> None:
        """A note box (`attentable`): everything in its cells, as quoted Markdown. Paragraphs,
        lists, code, tables and nested notes are converted like anywhere else, then every line
        gets a `> ` (blank lines `>`), so no text of the box is lost."""
        inner = _Converter(self.page_dir, self.self_key)
        inner.out = self.out
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            for cell in tr.find_all(["td", "th"], recursive=False):
                inner.walk(cell)
        text = inner.render()
        if text:
            self.emit("block", "\n".join(f"> {ln}" if ln else ">" for ln in text.split("\n")))

    def table_rows(self, table: Tag) -> list[list[str]]:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue  # a row of a table nested in a cell
            cells = [
                norm(self.inline(td)).replace("|", "\\|")
                for td in tr.find_all(["td", "th"], recursive=False)
            ]
            if any(cells):
                rows.append(cells)
        return rows

    # -- lists --------------------------------------------------------------------------------

    def list_block(self, el: Tag, *, indent: str) -> None:
        text = "\n".join(self.list_lines(el, indent))
        if text.strip():
            self.emit("block", text)

    def list_lines(self, el: Tag, indent: str) -> list[str]:
        lines: list[str] = []
        ordered = el.name == "ol"
        number = 1
        for li in el.find_all("li", recursive=False):
            marker = f"{number}." if ordered else "-"
            number += 1
            inline_parts: list[str] = []
            nested: list[str] = []
            for child in li.children:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, Tag) and child.name in ("ul", "ol"):
                    nested += self.list_lines(child, indent + "  ")
                elif isinstance(child, Tag) and child.name in _BLOCK_TAGS:
                    inner = _Converter(self.page_dir, self.self_key)
                    inner.out = self.out
                    inner.block(child)
                    nested += [
                        indent + "  " + ln if ln else ln for ln in inner.render().split("\n")
                    ]
                elif isinstance(child, Tag):
                    inline_parts.append(self.inline_tag(child, code=False))
                else:
                    inline_parts.append(str(child))
            text = norm("".join(inline_parts))
            if text or nested:
                lines.append(f"{indent}{marker} {text}".rstrip())
            lines += nested
        return lines


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def html_to_markdown(
    html_text: str, *, page_dir: str = "", self_key: str | None = None
) -> Converted:
    """Convert one page (see the module doc). `page_dir`: the page's folder inside the CHM, for
    relative image and link targets; `self_key`: the page's own key (a link to itself is not
    listed in `links_to`)."""
    soup = BeautifulSoup(html_text, "html.parser")
    converter = _Converter(page_dir, self_key)
    out = converter.out
    if soup.title is not None:
        out.html_title = norm(soup.title.get_text()) or None
    help_div = soup.find(id="help")
    body = soup.body or soup
    if isinstance(help_div, Tag):
        container: Tag = help_div
        nav_chars = sum(
            len(norm(el.get_text(" ")))
            for el in body.children
            if isinstance(el, Tag) and el is not help_div
        )
        if nav_chars:
            converter.drop("nav_bar", nav_chars)
    else:
        container = body
        out.has_container = False
    for tag in container.find_all(_DROPPED_TAGS):
        converter.drop("script_style", len(tag.get_text()))
        tag.decompose()
    visible = container.get_text(" ")
    converter.walk(container)
    out.markdown = converter.render()
    out.visible_chars = len(_WORD_CHARS.findall(visible))
    out.lost_chars = lost_characters(visible, out.markdown)
    out.heading = converter.heading
    return out
