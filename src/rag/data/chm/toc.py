"""Table of contents (`.hhc`), keyword index (`.hhk`) and page keys of a CHM (spec P4-01).

Both files are "sitemap" HTML: nested `<UL>` lists whose `<LI>` items are never closed and hold
`<OBJECT>` blocks of `<param name="Name">` / `<param name="Local">`. An item followed by a
nested `<UL>` is the parent of its items, so the title path of a page is the names from the top
down to it (design s5-4 `title_path`). The files are read by hand, not by an HTML parser, because
the unclosed `<LI>` make parsers close the list items in unhelpful places.

`page_key_of` turns a link target into the key two language versions of a page share (s8-4):
the path inside the CHM without extension, lower case, `/` separators, percent-encoding undone,
Unicode NFC.
"""

from __future__ import annotations

import html
import posixpath
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote

_TOKEN = re.compile(r"<\s*(/?)\s*(ul|li|param)\b([^>]*)>", re.I)
_ATTR = re.compile(r"""([A-Za-z_][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_EXTERNAL = re.compile(r"^(?:https?|ftp|mailto|javascript|tel|file|data):", re.I)
_PAGE_EXT = re.compile(r"\.html?$", re.I)


@dataclass(frozen=True)
class TocEntry:
    name: str
    local: str | None
    """Target as written in the file (may carry an anchor); None for a folder without a page."""
    path: tuple[str, ...]
    """Names from the top of the contents down to this entry, itself last."""
    depth: int
    """1 for the top level."""


@dataclass
class _Raw:
    depth: int
    params: list[tuple[str, str]]


def _attrs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _ATTR.finditer(text):
        value = next(g for g in match.groups()[1:] if g is not None)
        out[match.group(1).lower()] = value
    return out


def _raw_entries(text: str) -> list[_Raw]:
    entries: list[_Raw] = []
    current: _Raw | None = None
    depth = 0
    for match in _TOKEN.finditer(text):
        closing, tag, rest = match.group(1) == "/", match.group(2).lower(), match.group(3)
        if tag == "ul":
            depth = max(0, depth - 1) if closing else depth + 1
            current = None
        elif tag == "li" and not closing:
            current = _Raw(depth=max(depth, 1), params=[])
            entries.append(current)
        elif tag == "param" and not closing and current is not None:
            attrs = _attrs(rest)
            if "name" in attrs and "value" in attrs:
                current.params.append((attrs["name"].lower(), html.unescape(attrs["value"])))
    return entries


def _first(params: list[tuple[str, str]], name: str) -> str | None:
    return next((v for k, v in params if k == name), None)


def parse_toc(text: str) -> list[TocEntry]:
    """Every item of a `.hhc` in file order, with its title path (see the module doc)."""
    entries: list[TocEntry] = []
    last_at: dict[int, str] = {}
    for raw in _raw_entries(text):
        name = (_first(raw.params, "name") or "").strip()
        local = _first(raw.params, "local")
        for deeper in [d for d in last_at if d >= raw.depth]:
            del last_at[deeper]
        path = (*(last_at[d] for d in sorted(last_at)), name)
        last_at[raw.depth] = name
        entries.append(
            TocEntry(name=name, local=local.strip() if local else None, path=path, depth=raw.depth)
        )
    return entries


def parse_index(text: str) -> dict[str, list[str]]:
    """`page_key -> keywords` of a `.hhk`: every keyword whose item points at the page, once,
    in file order. Items without a target (`See Also` only) are ignored."""
    index: dict[str, list[str]] = {}
    for raw in _raw_entries(text):
        keyword = (_first(raw.params, "name") or "").strip()
        if not keyword:
            continue
        for name, value in raw.params:
            if name != "local":
                continue
            key = page_key_of(value)
            if key is not None and keyword not in index.setdefault(key, []):
                index[key].append(keyword)
    return {k: v for k, v in index.items() if v}


def page_key_of(local: str | None, *, base: str = "") -> str | None:
    """The page key of a link target, or None for an empty target, a pure anchor or an external
    address (`http:`, `mailto:`, `javascript:`).

    `base` is the folder of the page that holds the link (`sub`), for relative targets.
    CHM-internal forms (`mk:@MSITStore:C:\\x\\api.chm::/page.htm`, `ms-its:api.chm::/page.htm`)
    keep what follows `::/`. A target that climbs out of the CHM gives None.
    """
    if not local or not local.strip():
        return None
    target = local.strip()
    if "::/" in target:
        target = target.split("::/", 1)[1]
    elif target.startswith("#") or _EXTERNAL.match(target):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    target = unquote(target).replace("\\", "/")
    if target.startswith("/"):
        target, base = target.lstrip("/"), ""
    joined = posixpath.normpath(posixpath.join(base, target)) if target else ""
    if not joined or joined == "." or joined.startswith(".."):
        return None
    key = unicodedata.normalize("NFC", _PAGE_EXT.sub("", joined)).lower()
    return key or None
