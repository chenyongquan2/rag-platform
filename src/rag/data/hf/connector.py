"""`hf_dataset` connector: the Hugging Face documentation corpus as `SourceDocument`s (s5-1)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import ClassVar

import pyarrow as pa

from rag.config import ServiceSettings, repo_path
from rag.data.base import ConnectorMode, SourceDocument, make_doc_id
from rag.data.hf.download import CachedFile, ensure_dataset, read_dataset
from rag.data.hf.manifest import HfDataError, Manifest, load_manifest
from rag.registry import connectors


def check_limit(limit: int | None) -> int | None:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError(f"limit must be a positive integer or null, got {limit!r}")
    return limit


def corpus_documents(
    table: pa.Table, manifest: Manifest, *, limit: int | None = None
) -> Iterator[SourceDocument]:
    """Turn corpus rows into documents; `limit` keeps the first N rows of the pinned files.

    Every `path` value must be unique, because it is the document id. The check runs over the
    whole table even with a limit, so a limited run never hides a duplicate.
    """
    entry = manifest.corpus
    column = entry.columns["path"]
    paths = table.column(column).to_pylist()
    if any(not p for p in paths):
        raise HfDataError(f"{entry.repo_id}: empty {column!r} values")
    dupes = sorted(p for p, n in Counter(paths).items() if n > 1)
    if dupes:
        raise HfDataError(f"{entry.repo_id}: duplicate {column!r} values {dupes}")
    if limit is not None:
        table = table.slice(0, limit)
        paths = paths[:limit]
    texts = table.column(entry.columns["text"]).to_pylist()
    version = f"{entry.repo_id}@{entry.revision}"
    for source_rel, text in zip(paths, texts, strict=True):
        yield SourceDocument(
            doc_id=make_doc_id(manifest.kb, source_rel),
            kb=manifest.kb,
            source_rel=source_rel,
            source_url=entry.doc_url_prefix + source_rel if entry.doc_url_prefix else None,
            ext=PurePosixPath(source_rel).suffix.lower(),
            content=(text or "").encode("utf-8"),
            source_version=version,
        )


@connectors.register("hf_dataset")
class HfDatasetConnector:
    """Full-snapshot connector over a pinned HF dataset listed in a data manifest.

    Built from the KB config: `connectors.create("hf_dataset", kb=cfg.kb, settings=settings,
    **cfg.source.model_extra)`. A relative `manifest` path is taken from the repo root.
    `dataset` must be the manifest's corpus repo and `kb` the manifest's kb, so a YAML edit cannot
    silently point the knowledge base at other data.
    """

    mode: ClassVar[ConnectorMode] = "snapshot"

    def __init__(
        self,
        *,
        kb: str,
        dataset: str,
        manifest: str | Path,
        limit: int | None = None,
        settings: ServiceSettings | None = None,
    ) -> None:
        self.limit = check_limit(limit)
        self.manifest_path = repo_path(manifest)
        self.manifest = load_manifest(self.manifest_path)
        if self.manifest.kb != kb:
            raise HfDataError(
                f"{self.manifest_path} is the manifest of kb {self.manifest.kb!r}, not kb {kb!r}"
            )
        if self.manifest.corpus.repo_id != dataset:
            raise HfDataError(
                f"dataset {dataset!r} does not match the corpus "
                f"{self.manifest.corpus.repo_id!r} in {self.manifest_path}"
            )
        self.kb = kb
        self.settings = settings if settings is not None else ServiceSettings()

    def download(self) -> list[CachedFile]:
        """Fetch the corpus files into the cache unless verified copies are already there."""
        return ensure_dataset(
            self.manifest.corpus,
            endpoint=self.settings.hf_endpoint,
            data_dir=self.settings.hf_data_dir,
            timeout_s=self.settings.hf_timeout_s,
        )

    def documents(self) -> Iterator[SourceDocument]:
        self.download()
        table = read_dataset(self.manifest.corpus, self.settings.hf_data_dir)
        yield from corpus_documents(table, self.manifest, limit=self.limit)

    def doc_ids(self) -> list[str]:
        return [doc.doc_id for doc in self.documents()]
