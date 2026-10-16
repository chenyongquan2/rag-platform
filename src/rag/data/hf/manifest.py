"""The committed data manifest: which files to download and what they must look like.

One JSON file per knowledge base (`data/manifests/<kb>.json`) names the dataset repos, the pinned
revision, every file's size, sha256 and row count, plus license and source URL (design s1-1: record
where public data comes from and under which license). Raw data stays out of git; the manifest
lets anyone re-download it and check it is byte-for-byte the same.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

_CHUNK = 1 << 20


class HfDataError(RuntimeError):
    """Base error for Hugging Face data problems."""


class ManifestMismatchError(HfDataError):
    """A file does not match the size, sha256 or row count recorded in the manifest."""


class NotDownloadedError(HfDataError):
    """A file listed in the manifest is not in the local cache."""


class NotPinnedError(HfDataError):
    """A manifest entry has no recorded size/sha256/rows yet."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEntry(_Strict):
    """One file inside a dataset repo. `None` values are filled by `download --pin`."""

    path: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rows: int | None = Field(default=None, ge=0)

    @property
    def pinned(self) -> bool:
        return None not in (self.size, self.sha256, self.rows)


class DatasetEntry(_Strict):
    """One dataset repo at a pinned revision (a commit id, so the files never change under us)."""

    repo_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(min_length=1)
    ref: str | None = None
    """The branch the revision was taken from (`main`, `refs/convert/parquet`); informational."""
    source_url: str
    license: str
    columns: dict[str, str] = Field(min_length=1)
    """Our field name -> column in the files. Reading fails when a file lacks a mapped column."""
    doc_url_prefix: str | None = None
    """Prefix that turns the corpus `path` column into the original web page."""
    rows: int = Field(ge=0)
    note: str | None = None
    files: list[FileEntry] = Field(min_length=1)

    @property
    def label(self) -> str:
        return f"{self.repo_id}@{self.revision[:7]}"


CORPUS_FIELDS = frozenset({"text", "path"})
EVAL_FIELDS = frozenset({"question", "answer", "source_doc", "context"})


class Manifest(_Strict):
    kb: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    corpus: DatasetEntry
    eval: DatasetEntry | None = None

    @model_validator(mode="after")
    def _column_maps_complete(self) -> Self:
        for name, entry, fields in (
            ("corpus", self.corpus, CORPUS_FIELDS),
            ("eval", self.eval, EVAL_FIELDS),
        ):
            if entry is not None and set(entry.columns) != fields:
                raise ValueError(
                    f"{name}.columns must map exactly {sorted(fields)}, got {sorted(entry.columns)}"
                )
        return self


def load_manifest(path: str | Path) -> Manifest:
    return Manifest.model_validate_json(Path(path).read_text("utf-8"))


def save_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write the manifest as stable, LF-terminated JSON (it is committed)."""
    text = json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    Path(path).write_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    """Row count from the parquet footer (no need to read the data).

    The file is closed before returning: on Windows an open handle blocks the rename into the cache.
    """
    with pq.ParquetFile(path) as parquet:
        return parquet.metadata.num_rows


def verify_file(path: Path, entry: FileEntry) -> None:
    """Check the recorded size, sha256 and row count; values still `None` are not checked.

    Size first (cheap), then sha256, then the parquet row count.
    """
    if not path.is_file():
        raise NotDownloadedError(
            f"{path} is missing; run `uv run python -m rag.data.hf download` first"
        )
    size = path.stat().st_size
    if entry.size is not None and size != entry.size:
        raise ManifestMismatchError(f"{path}: size {size} != manifest size {entry.size}")
    if entry.sha256 is not None:
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise ManifestMismatchError(
                f"{path}: sha256 {actual} != manifest sha256 {entry.sha256}"
            )
    if entry.rows is not None:
        rows = parquet_rows(path)
        if rows != entry.rows:
            raise ManifestMismatchError(f"{path}: {rows} rows != manifest rows {entry.rows}")
