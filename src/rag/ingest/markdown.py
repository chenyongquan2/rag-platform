"""Markdown structure helpers shared by the cleaning rules and the chunking steps.

Every rule and step must agree on what is code: a cleaning rule that edits text inside a fenced
code block, or a splitter that cuts one, damages exactly the content that technical questions ask
about (design s5-2, s6-3). Fences are recognised the same way everywhere: a line that starts with
three or more backticks or tildes (any indentation) opens a block; a line of the same character,
at least as long, closes it; an unclosed block runs to the end of the text (CommonMark).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)((?:(?!\1)(?!\n[ \t]*\n)[\s\S])+?)\1(?!`)")
_START, _END = "\ue000", "\ue001"
_PLACEHOLDER = re.compile(f"{_START}(\\d+){_END}")


@dataclass(frozen=True)
class Segment:
    text: str
    code: bool


def is_closing_fence(line: str, fence: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {fence[0]} and len(stripped) >= len(fence)


def split_fenced(text: str) -> list[Segment]:
    """Split into prose and fenced-code segments; `"".join(s.text for s in ...) == text`.

    A code segment holds whole lines, from the opening fence line to the closing fence line
    including its line break.
    """
    segments: list[Segment] = []
    buf: list[str] = []
    fence = ""

    def flush(code: bool) -> None:
        if buf:
            segments.append(Segment("".join(buf), code))
            buf.clear()

    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if not fence:
            match = FENCE_OPEN.match(bare)
            # A backtick fence's info string cannot contain backticks: "```x``` y" is inline code.
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                flush(False)
                fence = match.group(1)
            buf.append(line)
        else:
            buf.append(line)
            if is_closing_fence(bare, fence):
                flush(True)
                fence = ""
    flush(bool(fence))
    return segments


@dataclass
class Masked:
    """Text with code replaced by placeholders; `restore` puts the code back."""

    text: str
    stash: list[str]

    def restore(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self.stash[int(m.group(1))], text)


def mask_code(text: str, *, inline: bool = True) -> Masked | None:
    """Replace fenced code blocks (and inline code spans) by placeholders.

    Rules run their regexes on the masked text, so they cannot touch code. A fenced block keeps
    its own line (the placeholder replaces it without its last line break). Returns None if the
    text already contains the placeholder characters (private-use code points); callers then
    leave the text unchanged.
    """
    if _START in text or _END in text:
        return None
    stash: list[str] = []

    def hide(code: str) -> str:
        stash.append(code)
        return f"{_START}{len(stash) - 1}{_END}"

    parts: list[str] = []
    for seg in split_fenced(text):
        if seg.code:
            body = seg.text[:-1] if seg.text.endswith("\n") else seg.text
            parts.append(hide(body) + seg.text[len(body) :])
        elif inline:
            parts.append(_INLINE_CODE.sub(lambda m: hide(m.group(0)), seg.text))
        else:
            parts.append(seg.text)
    return Masked("".join(parts), stash)


def has_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text))


def heading_title(line: str) -> tuple[int, str] | None:
    """`(level, title)` of an ATX heading line, None otherwise.

    The title loses closing hashes, images and link targets (`[#5279](https://...)` -> `#5279`):
    changelog headings carry several URLs, which would cost the title path ~100 tokens.
    """
    match = HEADING.match(line)
    if not match:
        return None
    title = _CLOSING_HASHES.sub("", match.group(2))
    title = _LINK.sub(r"\1", _IMAGE.sub("", title))
    title = " ".join(title.split())
    if not title:
        return None
    return len(match.group(1)), title


def is_title_only(text: str) -> bool:
    """True when every non-blank line outside code is a heading (and there is one)."""
    segments = split_fenced(text)
    if any(seg.code for seg in segments):
        return False
    content = [ln.strip() for seg in segments for ln in seg.text.splitlines() if ln.strip()]
    return bool(content) and all(heading_title(ln) for ln in content)


def is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2
