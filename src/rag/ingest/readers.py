"""Readers: raw connector bytes -> LlamaIndex Documents that meet the s5-4 contract.

Readers are looked up by file extension (or by the pipeline's `reader` name) in
`rag.registry.readers` (design s5-1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from rag.config import PipelineConfig
from rag.data import SourceDocument
from rag.ingest.meta import DEFAULT_EMBED_KEYS, DEFAULT_LLM_KEYS, apply_whitelist
from rag.registry import UnknownNameError, readers

DEFAULT_READERS = {".md": "markdown", ".markdown": "markdown", ".mdx": "markdown"}
"""Reader per extension when the pipeline does not name one."""

DOC_TYPES = {".md": "markdown", ".markdown": "markdown", ".mdx": "mdx", ".ipynb": "notebook"}

METADATA_TEMPLATE = "{key}: {value}"
METADATA_SEPARATOR = "\n"
TEXT_TEMPLATE = "{metadata_str}\n\n{content}"


class NotebookJsonError(ValueError):
    """A real (JSON) notebook was given to the Markdown reader."""


def reader_name_for(ext: str, pipeline: PipelineConfig) -> str:
    """The reader of a pipeline: its `reader` name, else the default for the extension."""
    if pipeline.reader:
        return pipeline.reader
    try:
        return DEFAULT_READERS[ext.lower()]
    except KeyError:
        known = ", ".join(sorted(DEFAULT_READERS))
        raise UnknownNameError(
            f"no default reader for {ext!r} (defaults exist for {known}); set `reader` in the "
            f"knowledge base's pipeline for {ext!r}"
        ) from None


def _looks_like_notebook_json(text: str) -> bool:
    if not text.lstrip().startswith("{"):
        return False
    try:
        data = json.loads(text)
    except ValueError:
        return False
    return isinstance(data, dict) and "cells" in data


@readers.register("markdown")
class MarkdownReader(BaseReader):
    """One Markdown (or MDX, or already flattened notebook) file -> one Document.

    The text is kept as delivered; front matter, MDX components and whitespace are handled by the
    cleaning rules, which run before the IngestionPipeline hashes the Document (s7-4).
    Metadata never carries modification times or sizes, and never the per-write fields
    `source_version`, `ingest_run_id`, `content_hash` (added to chunks at write time, s5-4).
    """

    def __init__(
        self,
        *,
        kb: str,
        pipeline_version: str,
        language: str = "und",
        acl_groups: Sequence[str] = ("public",),
        embed_keys: Sequence[str] = DEFAULT_EMBED_KEYS,
        llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
    ) -> None:
        self.kb = kb
        self.pipeline_version = pipeline_version
        self.language = language
        self.acl_groups = list(acl_groups)
        self.embed_keys = tuple(embed_keys)
        self.llm_keys = tuple(llm_keys)

    def load_data(self, source: SourceDocument, **_: Any) -> list[Document]:  # type: ignore[override]
        text = source.content.decode("utf-8", errors="replace")
        ext = source.ext.lower()
        if ext == ".ipynb" and _looks_like_notebook_json(text):
            raise NotebookJsonError(
                f"{source.source_rel}: JSON notebook; the markdown reader only reads notebooks "
                "already flattened to Markdown text (as the HF dataset stores them)"
            )
        metadata: dict[str, Any] = {
            "kb": self.kb,
            "doc_id": source.doc_id,
            "title_path": "",
            "source_rel": source.source_rel,
            "unit": None,
            "source_url": source.source_url,
            "doc_type": DOC_TYPES.get(ext, ext.lstrip(".") or "text"),
            "language": self.language,
            "acl_groups": list(self.acl_groups),
            "pipeline_version": self.pipeline_version,
        }
        doc = Document(
            id_=source.doc_id,
            text=text,
            metadata=metadata,
            metadata_template=METADATA_TEMPLATE,
            metadata_separator=METADATA_SEPARATOR,
            text_template=TEXT_TEMPLATE,
        )
        apply_whitelist(doc, self.embed_keys, self.llm_keys)
        return [doc]
