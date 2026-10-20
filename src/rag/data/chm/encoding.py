"""Decode a page of an unpacked CHM (spec P4-01, design s5-2, s6-2 "encoding repair").

A page's `<meta charset>` can lie, and in the MT5 manual it half-does: the English pages say
`iso-8859-1` but are windows-1252 (curly quotes and dashes in the bytes 0x92 to 0x97), the
Chinese pages say `gb2312` but need its superset GB18030. Browsers apply the same two aliases
(WHATWG Encoding Standard), so they are not counted as fallbacks here; any other disagreement
between the label and the bytes is.

Order of trial, each strict (no replacement characters):

1. UTF-8: a page that is valid UTF-8 and has non-ASCII bytes is UTF-8 whatever it says; the
   chance that GBK or windows-1252 text is also valid UTF-8 is negligible.
2. the declared charset (through the aliases), then the volume's hint (from the CHM locale),
   then GB18030, then windows-1252.

A candidate that decodes strictly can still be wrong. windows-1252 reads GBK bytes as a run of
accented letters ("mojibake"); that is rejected when GB18030 would read the same bytes as
Chinese and the volume does not say it is Western. When no candidate decodes strictly the best
guess is used with replacement characters, and their number is reported (`replaced`).
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass

_META = re.compile(rb"<meta\b[^>]*?charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:\-]+)", re.I | re.S)
_HEAD_BYTES = 4096

_WESTERN = frozenset({"cp1252", "iso8859-1", "latin-1", "ascii", "latin1"})
_CHINESE = frozenset({"gbk", "gb2312", "gb18030", "cp936"})


@dataclass(frozen=True)
class DecodedPage:
    text: str
    encoding: str
    """The codec that read the page: `ascii`, `utf-8`, `gb18030`, `cp1252`, ..."""
    declared: str | None
    """The charset the page declares (lower case), None when it declares none."""
    fallback: bool
    """True when the bytes were not read with the declared charset's family (or had to be
    read with replacement characters). The standard aliases do not count."""
    replaced: int
    """Replacement characters written because no codec decoded the page strictly."""


def declared_charset(raw: bytes) -> str | None:
    """The charset named by a `<meta>` tag in the first 4 KB, lower case, or None."""
    match = _META.search(raw[:_HEAD_BYTES])
    return match.group(1).decode("ascii", "replace").lower() if match else None


def codec_family(name: str | None) -> str | None:
    """The codec that reads `name` the way browsers do (`iso-8859-1` -> `cp1252`,
    `gb2312` -> `gb18030`); None for an empty or unknown name."""
    if not name:
        return None
    try:
        canonical = codecs.lookup(name).name
    except LookupError:
        return None
    if canonical in _WESTERN or name.lower() in {"iso-8859-1", "us-ascii", "latin1"}:
        return "cp1252"
    if canonical in _CHINESE:
        return "gb18030"
    return canonical


def _strict(raw: bytes, codec: str) -> str | None:
    try:
        return raw.decode(codec)
    except UnicodeDecodeError:
        return None


def _high(text: str) -> list[str]:
    return [c for c in text if ord(c) > 127]


def _is_cjk(char: str) -> bool:
    """Ideographs, CJK punctuation and kana, and the full-width forms Chinese text is full of."""
    code = ord(char)
    return 0x3000 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xFF00 <= code <= 0xFFEF


def _looks_like_mojibake(text: str, raw: bytes) -> bool:
    """`text` is `raw` read as windows-1252, but the bytes are Chinese read as GB18030."""
    high = _high(text)
    if len(high) < 4:
        return False
    latin = sum(0xA1 <= ord(c) <= 0xFF for c in high)
    if latin / len(high) < 0.8:
        return False
    chinese = _strict(raw, "gb18030")
    if chinese is None:
        return False
    other = _high(chinese)
    return bool(other) and sum(_is_cjk(c) for c in other) / len(other) >= 0.8


def _plausible(text: str, codec: str, raw: bytes, hint: str | None) -> bool:
    if "�" in text or any(0x80 <= ord(c) <= 0x9F for c in text):
        return False
    return not (codec == "cp1252" and hint != "cp1252" and _looks_like_mojibake(text, raw))


def decode_html(raw: bytes, *, hint: str | None = None) -> DecodedPage:
    """Decode the bytes of a page (see the module doc). `hint` is the codec the volume's
    locale suggests (`gb18030` for a Chinese CHM); it breaks ties, it never beats UTF-8."""
    declared = declared_charset(raw)
    declared_family = codec_family(declared)
    hint_family = codec_family(hint)
    if raw.startswith(b"\xef\xbb\xbf"):
        return DecodedPage(raw[3:].decode("utf-8", "replace"), "utf-8", declared, False, 0)
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return DecodedPage(raw.decode("utf-16", "replace"), "utf-16", declared, False, 0)
    if raw.isascii():
        return DecodedPage(raw.decode("ascii"), "ascii", declared, False, 0)

    candidates: list[str] = []
    for codec in ("utf-8", declared_family, hint_family, "gb18030", "cp1252"):
        if codec and codec not in candidates:
            candidates.append(codec)

    first_strict: tuple[str, str] | None = None
    for codec in candidates:
        text = _strict(raw, codec)
        if text is None:
            continue
        if first_strict is None:
            first_strict = (codec, text)
        if _plausible(text, codec, raw, hint_family):
            return _page(text, codec, declared, declared_family, 0)
    if first_strict is not None:
        return _page(first_strict[1], first_strict[0], declared, declared_family, 0)
    best = declared_family or hint_family or "cp1252"
    text = raw.decode(best, "replace")
    return _page(text, best, declared, declared_family, text.count("�"), force_fallback=True)


def _page(
    text: str,
    codec: str,
    declared: str | None,
    declared_family: str | None,
    replaced: int,
    *,
    force_fallback: bool = False,
) -> DecodedPage:
    fallback = force_fallback or (declared is not None and declared_family != codec)
    return DecodedPage(text, codec, declared, fallback, replaced)
