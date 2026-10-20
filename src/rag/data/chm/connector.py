"""`chm_dir` connector: unpacked CHM folders as `SourceDocument`s (spec P4-01, design s5-1, s5-4).

    source:
      connector: chm_dir
      volumes:
        - {language: zh, path: data/raw/mt5/zh}
        - {language: en, path: data/raw/mt5/en}
      limit: null          # first N pages (by file name) of each volume; most, not all, pairs

A volume may also name `sections: [Web API, Gateway API]`: only the pages under those top-level
entries of its contents are delivered (other pages and pages missing from the contents are left
out; a name the contents do not have is an error). It is how a part of a manual is ingested when
the whole does not fit the embedding budget; widening the list later adds the rest. For
experiments only: run against an index that already holds more pages, a narrower `sections` (or a
`limit`) leaves those pages out of the run, and the run deletes them from the index (the
mass-delete guard only stops a run that would pass fewer than half of the known pages).

One page = one document: `doc_id` `<kb>:<language>/<page file>`. The raw page bytes are handed on
undecoded (the reader detects the encoding and reports it); what the connector knows about the
page goes into `SourceDocument.extra`: `language` (from the volume), `page_key` (the path inside
the CHM without extension, shared by the versions of one page in the other volumes, s8-4),
`toc_path` (names from the `.hhc` hierarchy, None when the page is not in the contents),
`charset_hint` (from the CHM locale) and `keywords` (from the `.hhk`).

`source_version` is the CHM file's hash from the unpack manifest when there is one, else the
hash of the page, so an upgraded manual shows up in the audit fields (s5-4).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from rag.config import ServiceSettings, repo_path
from rag.data.base import ConnectorMode, SourceDocument, make_doc_id
from rag.data.chm.encoding import decode_html
from rag.data.chm.source import ChmError, DirectoryChmSource
from rag.data.chm.toc import TocEntry, page_key_of, parse_index, parse_toc
from rag.data.chm.unpack import read_unpack_manifest
from rag.data.hf.connector import check_limit
from rag.registry import connectors

_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)?$")


@dataclass
class VolumeStats:
    """What the connector found in one volume (counts and page keys; no page text)."""

    language: str
    pages: int = 0
    toc_entries: int = 0
    in_toc: int = 0
    not_in_toc: list[str] = field(default_factory=list)
    toc_without_page: list[str] = field(default_factory=list)
    duplicate_page_keys: list[str] = field(default_factory=list)
    selected: int = 0
    """Pages the volume's `sections` (all pages when none are given) let through."""
    index_keys_without_page: int = 0
    encodings_of_toc: str = ""
    chm_name: str | None = None
    chm_sha256: str | None = None
    lcid: int | None = None


@dataclass(frozen=True)
class _Volume:
    language: str
    source: DirectoryChmSource
    sections: tuple[str, ...] | None = None


def _volumes(volumes: Sequence[Mapping[str, Any]]) -> list[_Volume]:
    if not volumes:
        raise ValueError("chm_dir needs `volumes`: a list of {language, path}")
    out: list[_Volume] = []
    seen: set[str] = set()
    for entry in volumes:
        language = entry.get("language")
        if not isinstance(language, str) or not _LANGUAGE.match(language):
            raise ValueError(f"every volume needs a `language` such as zh or en, got {language!r}")
        if language in seen:
            raise ValueError(f"language {language!r} is listed twice in `volumes`")
        seen.add(language)
        sections = _sections(language, entry.get("sections"))
        path = repo_path(str(entry.get("path", "")))
        if not path.is_dir():
            raise ChmError(
                f"volume {language!r}: folder {path.name!r} does not exist (unpack the CHM first)"
            )
        out.append(
            _Volume(
                language,
                DirectoryChmSource(path, toc=entry.get("toc"), index=entry.get("index")),
                sections,
            )
        )
    return out


@connectors.register("chm_dir")
class ChmDirConnector:
    """Full-snapshot connector over unpacked CHM folders (see the module doc)."""

    mode: ClassVar[ConnectorMode] = "snapshot"

    def __init__(
        self,
        *,
        kb: str,
        volumes: Sequence[Mapping[str, Any]],
        limit: int | None = None,
        settings: ServiceSettings | None = None,
    ) -> None:
        self.kb = kb
        self.limit = check_limit(limit)
        self.settings = settings if settings is not None else ServiceSettings()
        self._volumes = _volumes(volumes)
        self.volume_stats: dict[str, VolumeStats] = {}

    def documents(self) -> Iterator[SourceDocument]:
        for volume in self._volumes:
            yield from self._volume_documents(volume)

    def doc_ids(self) -> list[str]:
        return [doc.doc_id for doc in self.documents()]

    @staticmethod
    def _select(
        volume: _Volume,
        names: list[str],
        entries: list[TocEntry],
        toc_by_key: dict[str, TocEntry],
    ) -> list[str]:
        """The pages the volume's `sections` let through: those listed in the contents under one
        of the named top-level entries (a page missing from the contents is left out). Every page
        when the volume names no sections. A name the contents do not have is an error."""
        if volume.sections is None:
            return names
        top = {e.path[0] for e in entries}
        unknown = sorted(set(volume.sections) - top)
        if unknown:
            raise ChmError(
                f"volume {volume.language!r}: no top-level contents entry named {unknown}; "
                f"known: {sorted(top)}"
            )
        wanted = set(volume.sections)
        return [
            name
            for name in names
            if (key := page_key_of(name)) is not None
            and key in toc_by_key
            and toc_by_key[key].path[0] in wanted
        ]

    def _volume_documents(self, volume: _Volume) -> Iterator[SourceDocument]:
        source, language = volume.source, volume.language
        info = source.info()
        hint = info.charset_hint
        manifest = read_unpack_manifest(source.root) or {}
        stats = VolumeStats(language=language, lcid=info.lcid)
        stats.chm_name = _text(manifest.get("chm_name"))
        stats.chm_sha256 = _text(manifest.get("chm_sha256"))
        self.volume_stats[language] = stats

        toc_by_key: dict[str, TocEntry] = {}
        toc_raw = source.toc()
        entries = parse_toc(decode_html(toc_raw, hint=hint).text) if toc_raw else []
        stats.toc_entries = len(entries)
        for entry in entries:
            key = page_key_of(entry.local)
            if key is not None:
                toc_by_key.setdefault(key, entry)
        index_raw = source.index()
        keywords = parse_index(decode_html(index_raw, hint=hint).text) if index_raw else {}

        names = source.names()
        keys: dict[str, str] = {}
        for name in names:
            key = page_key_of(name)
            if key is None:
                continue
            if key in keys:
                stats.duplicate_page_keys.append(key)
            keys.setdefault(key, name)
        stats.pages = len(names)
        stats.in_toc = sum(1 for k in keys if k in toc_by_key)
        stats.not_in_toc = sorted(k for k in keys if k not in toc_by_key) if toc_raw else []
        stats.toc_without_page = sorted(k for k in toc_by_key if k not in keys)
        stats.index_keys_without_page = sum(1 for k in keywords if k not in keys)

        version_of_chm = f"chm:{stats.chm_sha256[:16]}" if stats.chm_sha256 else None
        chosen = self._select(volume, names, entries, toc_by_key)
        stats.selected = len(chosen)
        chosen = chosen if self.limit is None else chosen[: self.limit]
        for name in chosen:
            key = page_key_of(name)
            raw = source.read(name)
            entry = toc_by_key.get(key) if key else None
            rel = f"{language}/{name}"
            yield SourceDocument(
                doc_id=make_doc_id(self.kb, rel),
                kb=self.kb,
                source_rel=rel,
                source_url=None,
                ext=Path(name).suffix.lower(),
                content=raw,
                source_version=version_of_chm or f"sha256:{hashlib.sha256(raw).hexdigest()[:16]}",
                extra={
                    "language": language,
                    "page_key": key,
                    "toc_path": list(entry.path) if entry else None,
                    "charset_hint": hint,
                    "keywords": list(keywords.get(key, [])) if key else [],
                },
            )


def _sections(language: str, value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list | tuple)
        or not value
        or not all(isinstance(v, str) and v for v in value)
    ):
        raise ValueError(
            f"volume {language!r}: `sections` must be a non-empty list of names of top-level "
            "contents entries"
        )
    return tuple(value)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
