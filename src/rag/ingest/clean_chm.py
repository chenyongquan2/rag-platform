"""Cleaning rules written for what the MT5 manual's pages leave after reading (spec P4-01).

Most noise of the manual is dropped while the page is read (`rag.ingest.chm_html`: navigation
bar, empty paragraphs, anchors, scripts), because it is markup and a DOM sees it exactly. What
is left for rules on the Markdown text is small; on the real pages (6,326 English and 6,273
Chinese) only this showed up:

- `chm_title_echo`: the page's first heading repeats the last name of its title path (10% of the
  pages: the contents entry and the first heading carry the same name). The title path is
  already in every chunk's embed text, so the heading only costs tokens.

Not here, and why: there is no "see also" block, copyright footer, redirect stub or repeated
header in these pages (checked page by page, counts in docs/reports/P4-report.md); the repeated
sentences (the fixed "returns this response code on success" line of the function pages) are
real text. Rules that would match nothing are not written (design s6-3: a rule is kept only
with a positive test, and a rule without a real positive case has none).
"""

from __future__ import annotations

from typing import ClassVar

from llama_index.core.schema import BaseNode

from rag.ingest.clean import TextRule
from rag.ingest.markdown import FENCE_OPEN, heading_title
from rag.ingest.meta import title_parts
from rag.registry import steps


def _same_title(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


@steps.register("chm_title_echo")
class ChmTitleEcho(TextRule):
    """Drop the first line of the page when it is a level-1 heading that says the same as the
    last name of the page's `title_path` (ignoring case and spacing), and the blank lines after
    it. Other headings, later headings (the sections of an overloaded function are each headed
    with its name), other levels, a first line inside a code block and pages without a title
    path are left alone."""

    name: ClassVar[str] = "chm_title_echo"

    def clean(self, text: str, node: BaseNode) -> str:
        parts = title_parts(str(node.metadata.get("title_path") or ""))
        if not parts:
            return text
        lines = text.splitlines(keepends=True)
        first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
        if first is None or FENCE_OPEN.match(lines[first].rstrip("\r\n")):
            return text
        heading = heading_title(lines[first].rstrip("\r\n"))
        if heading is None or heading[0] != 1 or not _same_title(heading[1], parts[-1]):
            return text
        rest = first + 1
        while rest < len(lines) and not lines[rest].strip():
            rest += 1
        return "".join(lines[rest:])
