"""Cleaning rules (design s6-1, s6-2, s6-3): named, switchable, tested, with statistics.

Every rule is a LlamaIndex `TransformComponent` that works on Documents (one Document = one unit)
and runs before the IngestionPipeline, so the pipeline hashes cleaned text (s7-4). Rules never
touch fenced code or inline code: they work on text where code is replaced by placeholders
(`rag.ingest.markdown.mask_code`). Each rule counts units seen and hit, characters removed and
added, and skipped units with a reason, for the quality report (s6-4).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar

import yaml
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from rag.ingest.injection import find_suspicious_instructions, scan_text
from rag.ingest.language import UNDETERMINED, detect_language
from rag.ingest.markdown import close_open_fence, heading_title, mask_code
from rag.ingest.meta import add_metadata
from rag.registry import steps

__all__ = [
    "CleanRule",
    "Dedupe",
    "DetectLanguage",
    "FrontMatter",
    "HfFirstChar",
    "MdMarkup",
    "RuleStats",
    "SkipEmpty",
    "SuspiciousInstruction",
    "TextRule",
    "Whitespace",
    "detect_language",
    "find_suspicious_instructions",
    "md_clean",
]


class RuleStats(BaseModel):
    units_seen: int = 0
    units_hit: int = 0
    chars_removed: int = 0
    chars_added: int = 0
    skipped: dict[str, str] = Field(default_factory=dict)
    """doc_id -> reason, for units this rule dropped."""
    duplicate_of: dict[str, str] = Field(default_factory=dict)
    """doc_id of a dropped copy -> doc_id of the copy that was kept (`dedupe`)."""


class CleanRule(TransformComponent):
    """Base class: `enabled: false` turns a rule off without removing it from the YAML."""

    name: ClassVar[str] = ""
    context_params: ClassVar[dict[str, str]] = {}
    """Constructor parameter -> processor context key filled in when the YAML does not set it."""

    enabled: bool = True
    _stats: RuleStats = PrivateAttr(default_factory=RuleStats)

    @classmethod
    def class_name(cls) -> str:
        return cls.name or cls.__name__

    @property
    def stats(self) -> RuleStats:
        return self._stats

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        if not self.enabled:
            return list(nodes)
        self._stats.units_seen += len(nodes)
        return self.apply(list(nodes))

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        raise NotImplementedError

    def _skip(self, node: BaseNode, reason: str) -> None:
        self._stats.units_hit += 1
        self._stats.skipped[node.node_id] = reason


class TextRule(CleanRule):
    """A rule that rewrites the text of each unit."""

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        for node in nodes:
            before = node.get_content(MetadataMode.NONE)
            after = self.clean(before, node)
            if after != before:
                node.set_content(after)
                self._stats.units_hit += 1
                self._stats.chars_removed += max(0, len(before) - len(after))
                self._stats.chars_added += max(0, len(after) - len(before))
        return nodes

    def clean(self, text: str, node: BaseNode) -> str:
        raise NotImplementedError


# --- hf_first_char --------------------------------------------------------------------------

_LOST_LT = re.compile(r"[A-Za-z][\w-]*(?:\s[^<>\n]*)?/?>\s*")
_LOST_STAR = re.compile(r"\*[^*\s][^*\n]*\*\*\s*")
_SHORT_TITLE = re.compile(r" \S.{0,118}")
_LOWERCASE_SENTENCE = re.compile(r" [a-z].*[.!?]")
_FRONT_MATTER_END = re.compile(r"^---[ \t]*$", re.M)


@steps.register("hf_first_char")
class HfFirstChar(TextRule):
    """Put back the first character the HF dataset export dropped from most pages.

    In `m-ric/huggingface_doc` the first character of nearly every page is missing: 1,224 pages
    start with " Title" (was "# Title"), 801 with "!--" (was "<!--", the licence comment),
    398 with "--" (was "---", front matter), 18 with "``python". Only these unambiguous shapes are
    repaired; a lost letter ("hat is ...") cannot be known and is left alone.

    A leading space is ambiguous: it is also what a lost one-letter word leaves (" demo for ..."
    was "A demo for ..." in gradio's demo/*/DESCRIPTION.md). In the corpus every such page is a
    single line, or a lowercase sentence; titles are followed by more text and are capitalised,
    non-Latin, or lowercase package names without a full stop. So "#" is put back only when
    more text follows and the line is not a lowercase sentence ending in . ! or ?.
    """

    name: ClassVar[str] = "hf_first_char"

    def clean(self, text: str, node: BaseNode) -> str:
        first, _, rest = text.partition("\n")
        if text.startswith("!--"):
            return "<" + text
        if first.rstrip() == "--" and _FRONT_MATTER_END.search(rest):
            return "-" + text
        if text.startswith("``") and not text.startswith("```"):
            return "`" + text
        title = first.rstrip()
        if (
            _SHORT_TITLE.fullmatch(title)
            and rest.strip()
            and not _LOWERCASE_SENTENCE.fullmatch(title)
        ):
            return "#" + text
        if _LOST_LT.fullmatch(first):
            return "<" + text
        if _LOST_STAR.fullmatch(first):
            return "*" + text
        return text


# --- front_matter ---------------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.S)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    if isinstance(value, list) and all(
        isinstance(v, bool | int | float | str | dt.date) for v in value
    ):
        return [_json_safe(v) for v in value]
    return _DROP


_DROP = object()


@steps.register("front_matter")
class FrontMatter(TextRule):
    """Move YAML front matter out of the text into metadata `front_matter` (design s5-2).

    Only top-level scalar values (and lists of scalars) are kept; dates become ISO strings.
    A block that is not a YAML mapping (a horizontal rule, say) is left in the text.
    """

    name: ClassVar[str] = "front_matter"

    def clean(self, text: str, node: BaseNode) -> str:
        match = _FRONT_MATTER.match(text)
        if not match:
            return text
        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return text
        if not isinstance(data, dict) or not data:
            return text
        kept = {str(k): _json_safe(v) for k, v in data.items()}
        add_metadata(node, "front_matter", {k: v for k, v in kept.items() if v is not _DROP})
        return text[match.end() :].lstrip("\n")


# --- md_markup ------------------------------------------------------------------------------

_HTML_TAGS = frozenset(
    """a abbr address area article aside audio b bdi bdo blockquote body br button canvas caption
    center cite code col colgroup dd del details dfn dialog div dl dt em embed fieldset figcaption
    figure font footer form h1 h2 h3 h4 h5 h6 head header hr html i iframe img ins kbd li main mark
    meta nav noscript object ol p picture pre q s samp script section small source span strike
    strong style sub summary sup svg table tbody td template tfoot th thead tr tt u ul video
    wbr""".split()  # noqa: SIM905
)
_DOC_BUILDER_TAGS = frozenset(
    """frameworkcontent pt tf jax hfoptions hfoption literalinclude python rust node js curl
    tokenizerslangcontent inferencesnippet medium-zoom gradio-app gradio-lite""".split()  # noqa: SIM905
)
_DROP_WITH_CONTENT = frozenset(
    {"script", "style", "iframe", "video", "audio", "svg", "literalinclude", "noscript"}
    | {"template", "object"}
)
_NEEDS_ATTRIBUTES = frozenset({"input", "label", "select", "option", "textarea", "output"})
"""Form elements that double as placeholders in prose (`<input>`): tags only with attributes."""
_COMPONENT = re.compile(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*")
_TAG_START = re.compile(r"<(/?)([A-Za-z][\w.:-]*)")
_QUIZ_FIELD = re.compile(r"\b(text|explain)\s*:\s*([\"'])((?:(?!\2)[^\\]|\\.)*)\2")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING_ANCHOR = re.compile(
    r"^(#{1,6}[ \t]+.*?)[ \t]*(?:\[\[[\w.:-]+\]\]|\{#[\w.:-]+\})[ \t]*$", re.M
)
_COLAB_LINE = re.compile(r"^[ \t]*\[\[open-in-colab\]\][ \t]*$\n?", re.M)
_MDX_IMPORT = re.compile(
    r"^import\s+(?:[\w*{}\s,]+\s+from\s+)?[\"'][^\"'\n]+[\"'];?[ \t]*$\n?", re.M
)
_EMPTY_LINK = re.compile(r"(?<!!)\[\]\([^)\n]*\)")
_LINK_NO_TARGET = re.compile(r"(?<!!)\[([^\]\n]+)\]\(\s*\)")
_ENTITY = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_MAX_TAG_CHARS = 20_000
_ATTR_NAME = re.compile(r"[^\s\"'<>/=`{}]+")
_ATTR_VALUE = re.compile(r"[^\s\"'=<>`]+")


def _is_tag_name(name: str, *, closing: bool, before: str) -> bool:
    lower = name.lower()
    if lower in _HTML_TAGS or lower in _DOC_BUILDER_TAGS or lower in _NEEDS_ATTRIBUTES:
        return True
    # JSX components (`<Tip>`, `<Question>`): not right after a letter or digit, so
    # `List<String>` and `Vec<Tensor>` stay text.
    return bool(_COMPONENT.fullmatch(name)) and (closing or not before.isalnum())


def _skip_quoted(text: str, i: int) -> int:
    quote = text[i]
    i += 1
    while i < len(text) and text[i] != quote:
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _skip_braces(text: str, i: int) -> int:
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'`":
            i = _skip_quoted(text, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _skip_spaces(text: str, i: int, limit: int) -> int:
    while i < limit and text[i] in " \t":
        i += 1
    return i


def _tag_end(text: str, i: int, *, closing: bool, min_attributes: int = 0) -> int:
    """End index of a tag whose name ends at `i`, or -1 if this is not a tag."""
    limit = min(len(text), i + _MAX_TAG_CHARS)
    if i >= limit or text[i] not in " \t\n/>":
        return -1
    if closing:
        j = _skip_spaces(text, i, limit)
        return j + 1 if j < limit and text[j] == ">" else -1
    attributes = 0
    while i < limit:
        ch = text[i]
        if ch in " \t\n":
            if text.startswith("\n\n", i) or text.startswith("\n \n", i):
                return -1  # a blank line ends a tag that never closed: it was prose
            i += 1
        elif ch == ">" or text.startswith("/>", i):
            end = i + (1 if ch == ">" else 2)
            return end if attributes >= min_attributes else -1
        elif ch == "{":
            attributes += 1
            i = _skip_braces(text, i)
            if i < 0:
                return -1
        else:
            match = _ATTR_NAME.match(text, i)
            if not match:
                return -1
            attributes += 1
            i = _skip_spaces(text, match.end(), limit)  # `align = "center"` happens
            if i < limit and text[i] == "=":
                i = _skip_spaces(text, i + 1, limit)
                if i >= limit:
                    return -1
                if text[i] in "\"'":
                    i = _skip_quoted(text, i)
                elif text[i] == "{":
                    i = _skip_braces(text, i)
                    if i < 0:
                        return -1
                else:
                    value = _ATTR_VALUE.match(text, i)
                    if not value:
                        return -1
                    i = value.end()
    return -1


def _quiz_lines(tag: str) -> str:
    lines = []
    for field, _, value in _QUIZ_FIELD.findall(tag):
        value = value.replace("\\'", "'").replace('\\"', '"').strip()
        if value:
            lines.append(f"- {value}" if field == "text" else f"  {value}")
    return "\n" + "\n".join(lines) + "\n" if lines else ""


def _joiner(text: str, start: int, end: int) -> str:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return " " if before.isalnum() and after.isalnum() else ""


def strip_tags(text: str) -> str:
    """Remove HTML and MDX tags, keep the text between them.

    `<script>`, `<style>`, `<iframe>`, `<video>`, `<svg>`, `<literalinclude>` and similar lose
    their content too. Quiz components keep their `text:`/`explain:` strings as a list. Anything
    that does not parse as a tag of a known HTML, doc-builder or component name is left alone,
    so `a < 0`, `x<y`, `<model_id>`, `<https://...>` and `List<String>` stay.
    """
    out: list[str] = []
    pos = 0
    for match in _TAG_START.finditer(text):
        start = match.start()
        if start < pos:
            continue
        closing, name = match.group(1) == "/", match.group(2)
        before = text[start - 1] if start > 0 else " "
        if not _is_tag_name(name, closing=closing, before=before):
            continue
        needs = 1 if name.lower() in _NEEDS_ATTRIBUTES else 0
        end = _tag_end(text, match.end(), closing=closing, min_attributes=needs)
        if end < 0:
            continue
        replacement = _joiner(text, start, end)
        if not closing and name.lower() in _DROP_WITH_CONTENT and text[end - 2 : end] != "/>":
            close = re.compile(rf"</{re.escape(name)}\s*>", re.I).search(text, end)
            if close:
                end = close.end()
        elif not closing and _COMPONENT.fullmatch(name):
            replacement = _quiz_lines(text[start:end]) or replacement
        out.append(text[pos:start])
        out.append(replacement)
        pos = end
    out.append(text[pos:])
    return "".join(out)


@steps.register("md_markup")
class MdMarkup(TextRule):
    """Markup cleanup (design s6-2 标记清理): HTML comments, MDX/HTML tags and components,
    MDX `import` lines, doc-builder heading anchors and `[[open-in-colab]]`, empty links, HTML
    entities. Fenced code, inline code and Markdown tables are kept as they are; entities are
    decoded only after tags are gone, so `&lt;b&gt;` never turns into a tag that gets removed.
    """

    name: ClassVar[str] = "md_markup"

    def clean(self, text: str, node: BaseNode) -> str:
        masked = mask_code(text)
        if masked is None:
            return text
        out = _COMMENT.sub("", masked.text)
        out = _HEADING_ANCHOR.sub(r"\1", out)
        out = _COLAB_LINE.sub("", out)
        out = _MDX_IMPORT.sub("", out)
        out = strip_tags(out)
        out = _EMPTY_LINK.sub("", out)
        out = _LINK_NO_TARGET.sub(r"\1", out)
        out = _ENTITY.sub(lambda m: html.unescape(m.group(0)), out)
        return masked.restore(out)


# --- whitespace -----------------------------------------------------------------------------

_INVISIBLE = {0x200B: None, 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None, 0xA0: " "}
_TRAILING = re.compile(r"[ \t]+$", re.M)
_BLANK_RUN = re.compile(r"\n{3,}")


@steps.register("whitespace")
class Whitespace(TextRule):
    """Line breaks to `\\n`; outside code: drop zero-width characters and trailing spaces,
    non-breaking space to space, at most one blank line in a row; trim the unit. A code block
    left open at the end of the unit gets its closing fence (renders the same, CommonMark)."""

    name: ClassVar[str] = "whitespace"

    def clean(self, text: str, node: BaseNode) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        masked = mask_code(text, inline=False)
        if masked is None:
            return text.strip()
        out = masked.text.translate(_INVISIBLE)
        out = _TRAILING.sub("", out)
        out = _BLANK_RUN.sub("\n\n", out)
        return close_open_fence(masked.restore(out.strip()))


# --- skip_empty -----------------------------------------------------------------------------

_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
_LINK_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s+\[[^\]\n]*\]\([^)\n]*\)[.,;]?")


@steps.register("skip_empty")
class SkipEmpty(CleanRule):
    """Drop units with no real content (design s6-2 空内容和目录页).

    Reasons: `empty` (nothing left), `title_only` (only headings), `toc_only` (only lists of
    links), `too_short` (fewer than `min_chars` visible characters outside headings; code counts
    as content). Dropped units are not passed to the pipeline, so their old chunks are deleted.
    """

    name: ClassVar[str] = "skip_empty"
    min_chars: int = Field(default=20, ge=0)

    def reason(self, text: str) -> str | None:
        if not text.strip():
            return "empty"
        masked = mask_code(text, inline=False)
        if masked is None:
            return None
        lines = [ln.strip() for ln in masked.text.split("\n") if ln.strip()]
        body = [ln for ln in lines if not heading_title(ln)]
        if not body:
            return "title_only"
        if masked.stash:
            return None
        if len(body) >= 2 and all(_LINK_ITEM.fullmatch(ln) for ln in body):
            return "toc_only"
        visible = _LINK.sub(r"\1", " ".join(body))
        if len(re.sub(r"\s", "", visible)) < self.min_chars:
            return "too_short"
        return None

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        kept = []
        for node in nodes:
            reason = self.reason(node.get_content(MetadataMode.NONE))
            if reason:
                self._skip(node, reason)
            else:
                kept.append(node)
        return kept


# --- dedupe ---------------------------------------------------------------------------------


@steps.register("dedupe")
class Dedupe(CleanRule):
    """Keep one copy of units with exactly the same cleaned text (design s6-2 重复内容).

    The copy with the smallest doc_id is kept, so the choice does not depend on input order.
    Where each dropped copy went is recorded in `stats.duplicate_of` (and the quality report),
    not in the kept Document: its hash must not change when another page appears (s5-4).
    """

    name: ClassVar[str] = "dedupe"

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        groups: dict[str, list[BaseNode]] = defaultdict(list)
        for node in nodes:
            text = node.get_content(MetadataMode.NONE)
            groups[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(node)
        kept_id: dict[str, str] = {}
        for group in groups.values():
            first = min(n.node_id for n in group)
            for n in group:
                kept_id[n.node_id] = first
        kept = []
        for node in nodes:
            if kept_id[node.node_id] == node.node_id:
                kept.append(node)
            else:
                self._skip(node, "duplicate")
                self._stats.duplicate_of[node.node_id] = kept_id[node.node_id]
        return kept


# --- detect_language ------------------------------------------------------------------------


@steps.register("detect_language")
class DetectLanguage(CleanRule):
    """Write the detected language of each unit to metadata `language` (design s5-4, s6-2).

    When there is too little prose to tell (a code-only page), `default` is used: the knowledge
    base's language, or `und` for `language: auto`. Units hit = units whose language changed.
    """

    name: ClassVar[str] = "detect_language"
    context_params: ClassVar[dict[str, str]] = {"default": "kb_language"}
    default: str = UNDETERMINED

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        for node in nodes:
            found = detect_language(node.get_content(MetadataMode.NONE))
            language = self.default if found == UNDETERMINED else found
            if node.metadata.get("language") != language:
                self._stats.units_hit += 1
            if "language" in node.metadata:
                node.metadata["language"] = language
            else:
                add_metadata(node, "language", language)
        return nodes


# --- suspicious_instruction -----------------------------------------------------------------
# The patterns, views and false-positive guards live in `rag.ingest.injection` (spec P1-03).


@steps.register("suspicious_instruction")
class SuspiciousInstruction(CleanRule):
    """Set metadata `suspicious_instruction` (True/False) from the node's own text
    (`rag.ingest.injection.scan_text`: zh and en patterns, hidden-text tricks).

    In the clean chain it flags units; as a chunk step (after the splitters) it sets the flag per
    chunk, replacing the value inherited from the unit (design s11-3: 命中的块标记).
    """

    name: ClassVar[str] = "suspicious_instruction"

    def apply(self, nodes: list[BaseNode]) -> list[BaseNode]:
        for node in nodes:
            hit = bool(scan_text(node.get_content(MetadataMode.NONE)))
            self._stats.units_hit += hit
            add_metadata(node, "suspicious_instruction", hit)
        return nodes


# --- presets --------------------------------------------------------------------------------


def md_clean(*, kb_language: str = UNDETERMINED) -> list[CleanRule]:
    """The generic Markdown chain (the default `clean` of `.md` pipelines)."""
    return [
        FrontMatter(),
        MdMarkup(),
        Whitespace(),
        SkipEmpty(),
        Dedupe(),
        DetectLanguage(default=kb_language),
        SuspiciousInstruction(),
    ]


md_clean.context_params = {"kb_language": "kb_language"}  # type: ignore[attr-defined]
steps.add("md_clean", md_clean)
