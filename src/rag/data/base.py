"""What every connector hands to the readers (design s5-1, s5-4)."""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConnectorMode = Literal["snapshot", "changes"]
"""`snapshot`: every document on each run; `changes`: only what changed since last run (s7-4)."""


def normalize_rel_path(path: str) -> str:
    """Normalize a relative path to `a/b/c` form, or raise `ValueError` if it is not relative.

    Backslashes become `/`; empty and `.` segments are dropped (`./a//b/./c` -> `a/b/c`). Absolute
    paths (`/x`, `//server/x`, `C:/x`, `C:x`) and `..` segments are rejected, so the result can be
    joined under a directory without escaping it.
    """
    rel = path.replace("\\", "/")
    if rel.startswith("/") or PureWindowsPath(rel).anchor:
        raise ValueError(f"path must be relative, got {path!r}")
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"path must not contain '..', got {path!r}")
    if not parts:
        raise ValueError(f"path must not be empty, got {path!r}")
    return "/".join(parts)


def make_doc_id(kb: str, source_rel: str, unit: str | None = None) -> str:
    """Stable document id: knowledge base + relative path (+ unit inside the file), design s5-4.

    `hf_docs:huggingface/transformers/blob/main/docs/source/en/bert.md`,
    `mt5_manuals:help/mt5.chm#page.htm`. The path is normalized (`normalize_rel_path`), so the
    same file gets the same id on Windows and Linux; an id never depends on where the data lives.
    `#` separates the unit, so `%` and `#` inside the path are percent-encoded (`c#.md` ->
    `c%23.md`): the first `#` after the kb is always the unit separator and ids stay unique.
    """
    if not kb:
        raise ValueError("kb must not be empty")
    rel = normalize_rel_path(source_rel).replace("%", "%25").replace("#", "%23")
    if unit is None:
        return f"{kb}:{rel}"
    if not unit:
        raise ValueError("unit must not be empty; pass None for a file without units")
    return f"{kb}:{rel}#{unit}"


class SourceDocument(BaseModel):
    """One raw document as delivered by a connector, before any reader runs.

    `content` is the raw bytes; the reader picked by `ext` turns it into LlamaIndex Documents.
    `source_version` identifies the exact upstream version (file hash or commit) for audits; it is
    added to chunks at the end of processing, never to Document metadata (s5-4).
    `extra` holds what the connector knows about the document and the bytes do not say (for a CHM
    page: language, page key, title path from the contents, charset hint); the reader decides
    which of it becomes metadata. Empty for connectors that need nothing of the kind.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    kb: str
    source_rel: str
    source_url: str | None
    ext: str
    content: bytes
    source_version: str
    extra: dict[str, Any] = Field(default_factory=dict)
