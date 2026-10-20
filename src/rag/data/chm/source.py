"""Where the pages of a CHM come from (spec P4-01, design s5-2).

`ChmSource` is what the connector needs of a manual: page names, page bytes, the contents and
index files and the locale facts. `DirectoryChmSource` reads a folder that holds an unpacked CHM
(what `python -m rag.data.chm unpack` writes, or what `7z x` / `hh.exe -decompile` leave); tests
use it with small invented folders. Unpacking a real CHM file is a separate, thin step
(`rag.data.chm.unpack`), so nothing else depends on which tool did it or on the operating system.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.data.base import normalize_rel_path

PAGE_SUFFIXES = (".htm", ".html")


class ChmError(ValueError):
    """A CHM folder or a request to it is not usable."""


@dataclass(frozen=True)
class ChmInfo:
    """Facts from the CHM's `#SYSTEM` file (all None when the file is missing or empty)."""

    lcid: int | None = None
    """Windows locale id the help was compiled for: 0x409 English (US), 0x804 Chinese (PRC)."""
    title: str | None = None
    default_page: str | None = None
    charset_hint: str | None = None
    """The codec that locale's pages usually use (`charset_hint_for_lcid`)."""


class ChmSource(Protocol):
    def names(self) -> list[str]:
        """Relative paths (`/` separators) of every page, sorted."""
        ...

    def read(self, name: str) -> bytes: ...

    def toc(self) -> bytes | None:
        """The raw `.hhc` bytes, None when the manual has no contents file."""
        ...

    def index(self) -> bytes | None: ...

    def info(self) -> ChmInfo: ...


# Primary language ids of English, German, Spanish, French, Italian, Dutch, Portuguese, Swedish,
# Danish, Norwegian and Finnish: Windows locales that use windows-1252.
_WESTERN_LANGUAGES = frozenset({0x09, 0x07, 0x0A, 0x0C, 0x10, 0x13, 0x16, 0x1D, 0x06, 0x14, 0x0B})


def charset_hint_for_lcid(lcid: int | None) -> str | None:
    """The codec a CHM of this Windows locale usually uses, None when there is no safe guess."""
    if lcid is None:
        return None
    if lcid in (0x804, 0x1004):  # Chinese (PRC, Singapore): GBK, read as its superset
        return "gb18030"
    if lcid in (0x404, 0xC04, 0x1404):  # Chinese (Taiwan, Hong Kong, Macau)
        return "big5"
    return "cp1252" if lcid & 0x3FF in _WESTERN_LANGUAGES else None


def parse_system_file(data: bytes) -> ChmInfo:
    """Read the entries of `#SYSTEM`: a version word, then (code, length, data) records; code 4
    holds the locale id, 3 the title, 2 the default page."""
    lcid: int | None = None
    title = default_page = None
    pos = 4
    while pos + 4 <= len(data):
        code, length = struct.unpack("<HH", data[pos : pos + 4])
        record = data[pos + 4 : pos + 4 + length]
        pos += 4 + length
        if code == 4 and len(record) >= 4:
            lcid = struct.unpack("<I", record[:4])[0] or None
        elif code == 3:
            title = record.split(b"\x00")[0].decode("cp1252", "replace") or None
        elif code == 2:
            default_page = record.split(b"\x00")[0].decode("cp1252", "replace") or None
    return ChmInfo(lcid, title, default_page, charset_hint_for_lcid(lcid))


class DirectoryChmSource:
    """An unpacked CHM in a folder. CHM internals (`#SYSTEM`, `$WWKeywordLinks`, ...), images,
    scripts and the unpack manifest are not pages; pages are `.htm` / `.html` files."""

    def __init__(
        self, root: str | Path, *, toc: str | None = None, index: str | None = None
    ) -> None:
        self.root = Path(root)
        self._toc = toc
        self._index = index

    def names(self) -> list[str]:
        found: list[str] = []
        for folder, subdirs, files in os.walk(self.root):
            subdirs[:] = [d for d in subdirs if d[0] not in "#$"]
            for file in files:
                if file[0] in "#$" or not file.lower().endswith(PAGE_SUFFIXES):
                    continue
                rel = (Path(folder) / file).relative_to(self.root)
                found.append(rel.as_posix())
        return sorted(found)

    def read(self, name: str) -> bytes:
        try:
            rel = normalize_rel_path(name)
        except ValueError as exc:
            raise ChmError(f"not a page of this manual: {exc}") from exc
        path = self.root / rel
        if not path.is_file():
            raise ChmError(f"page {name!r} does not exist in {self.root.name}")
        return path.read_bytes()

    def _find(self, suffix: str, wanted: str | None) -> Path | None:
        if wanted:
            path = self.root / wanted
            if not path.is_file():
                raise ChmError(f"{wanted!r} does not exist in {self.root.name}")
            return path
        found = sorted(p for p in self.root.glob(f"*{suffix}") if p.is_file())
        if len(found) > 1:
            names = ", ".join(p.name for p in found)
            raise ChmError(
                f"several {suffix} files in {self.root.name} ({names}); name the one to use"
            )
        return found[0] if found else None

    @property
    def toc_name(self) -> str | None:
        path = self._find(".hhc", self._toc)
        return path.name if path else None

    @property
    def index_name(self) -> str | None:
        path = self._find(".hhk", self._index)
        return path.name if path else None

    def toc(self) -> bytes | None:
        path = self._find(".hhc", self._toc)
        return path.read_bytes() if path else None

    def index(self) -> bytes | None:
        path = self._find(".hhk", self._index)
        return path.read_bytes() if path else None

    def info(self) -> ChmInfo:
        path = self.root / "#SYSTEM"
        if not path.is_file():
            return ChmInfo()
        return parse_system_file(path.read_bytes())
