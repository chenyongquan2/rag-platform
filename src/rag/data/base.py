"""What every connector hands to the readers (design s5-1, s5-4)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict

ConnectorMode = Literal["snapshot", "changes"]
"""`snapshot`: every document on each run; `changes`: only what changed since last run (s7-4)."""


def make_doc_id(kb: str, source_rel: str, unit: str | None = None) -> str:
    """Stable document id: knowledge base + relative path (+ unit inside the file), design s5-4.

    `hf_docs:huggingface/transformers/blob/main/docs/source/en/bert.md`,
    `mt5_manuals:help/mt5.chm#page.htm`. Backslashes become `/` and a leading `./` is dropped, so
    the same file gets the same id on Windows and Linux. Absolute paths and `..` are rejected: an
    id must not depend on where the data happens to live on this machine.
    """
    if not kb:
        raise ValueError("kb must not be empty")
    rel = source_rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel.startswith("/") or PurePosixPath(rel).is_absolute():
        raise ValueError(f"source_rel must be a non-empty relative path, got {source_rel!r}")
    if ".." in rel.split("/"):
        raise ValueError(f"source_rel must not contain '..', got {source_rel!r}")
    return f"{kb}:{rel}" if unit is None else f"{kb}:{rel}#{unit}"


class SourceDocument(BaseModel):
    """One raw document as delivered by a connector, before any reader runs.

    `content` is the raw bytes; the reader picked by `ext` turns it into LlamaIndex Documents.
    `source_version` identifies the exact upstream version (file hash or commit) for audits; it is
    added to chunks at the end of processing, never to Document metadata (s5-4).
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    kb: str
    source_rel: str
    source_url: str | None
    ext: str
    content: bytes
    source_version: str
