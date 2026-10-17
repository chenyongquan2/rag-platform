"""Test helpers for ingestion: an editable in-memory connector and small Markdown pages."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import PurePosixPath

from rag.data import SourceDocument, make_doc_id


def make_source(kb: str, rel: str, text: str, version: str = "src@1") -> SourceDocument:
    """A connector document for `rel` (its extension picks the pipeline)."""
    return SourceDocument(
        doc_id=make_doc_id(kb, rel),
        kb=kb,
        source_rel=rel,
        source_url=None,
        ext=PurePosixPath(rel).suffix,
        content=text.encode("utf-8"),
        source_version=version,
    )


class ListConnector:
    """Snapshot connector over documents a test adds, changes and removes between runs."""

    mode = "snapshot"

    def __init__(self, kb: str, folder: str = "docs") -> None:
        self.kb = kb
        self.folder = folder
        self.docs: dict[str, SourceDocument] = {}

    def doc_id(self, name: str) -> str:
        return make_doc_id(self.kb, f"{self.folder}/{name}")

    def put(self, name: str, text: str, version: str = "src@1") -> str:
        doc = make_source(self.kb, f"{self.folder}/{name}", text, version)
        self.docs[doc.doc_id] = doc
        return doc.doc_id

    def remove(self, name: str) -> None:
        del self.docs[self.doc_id(name)]

    def documents(self) -> Iterator[SourceDocument]:
        yield from list(self.docs.values())


def section(seed: str, n: int = 12) -> str:
    """A sentence of `n` distinct words, `seed0 seed1 ...`."""
    return " ".join(f"{seed}{i}" for i in range(n)) + "."


def page(title: str, *sections: str) -> str:
    """`# title` and one `## Part i` heading per section."""
    body = "\n\n".join(f"## Part {i}\n\n{text}" for i, text in enumerate(sections))
    return f"# {title}\n\n{body}\n"
