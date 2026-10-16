"""Download dataset files from a Hugging Face endpoint (the mirror) into a local cache.

Cache layout: `<data_dir>/<repo_id>/<revision>/<path>`. The revision is part of the path, so a
new revision never overwrites an old file. Files are written to a temporary name, verified against
the manifest and only then renamed into place, so the cache never holds a file that failed
verification (two checkouts sharing one cache can download at the same time).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from rag.data.hf.manifest import (
    DatasetEntry,
    FileEntry,
    HfDataError,
    ManifestMismatchError,
    NotPinnedError,
    parquet_rows,
    sha256_file,
    verify_file,
)

_USER_AGENT = "rag-platform/0.1 (+urllib)"
_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY_S = 0.2


@dataclass(frozen=True)
class CachedFile:
    path: Path
    entry: FileEntry
    downloaded: bool
    """True when this call fetched the file; False when a verified copy was already cached."""


def file_url(endpoint: str, entry: DatasetEntry, file: FileEntry) -> str:
    """`<endpoint>/datasets/<repo_id>/resolve/<revision>/<path>` (the Hub's download URL)."""
    revision = quote(entry.revision, safe="")
    return f"{endpoint.rstrip('/')}/datasets/{entry.repo_id}/resolve/{revision}/{quote(file.path)}"


def cache_path(data_dir: str | Path, entry: DatasetEntry, file: FileEntry) -> Path:
    return Path(data_dir) / entry.repo_id / entry.revision / file.path


def dataset_paths(entry: DatasetEntry, data_dir: str | Path) -> list[Path]:
    return [cache_path(data_dir, entry, f) for f in entry.files]


def ensure_dataset(
    entry: DatasetEntry,
    *,
    endpoint: str,
    data_dir: str | Path,
    timeout_s: float,
    pin: bool = False,
) -> list[CachedFile]:
    """Make every file of `entry` present in the cache and verified against the manifest.

    A cached file that verifies is not fetched again. Without `pin`, every file must already have
    size, sha256 and rows in the manifest. With `pin`, missing values are filled in `entry` (in
    place) from the downloaded file; values already recorded are still checked, never overwritten.
    """
    if not pin:
        unpinned = [f.path for f in entry.files if not f.pinned]
        if unpinned:
            raise NotPinnedError(
                f"{entry.repo_id}: no size/sha256/rows recorded for {unpinned}; run "
                "`uv run python -m rag.data.hf download --pin` once and commit the manifest"
            )
    results: list[CachedFile] = []
    for file in entry.files:
        target = cache_path(data_dir, entry, file)
        if file.pinned and _verifies(target, file):
            results.append(CachedFile(target, file, downloaded=False))
            continue
        _download(file_url(endpoint, entry, file), target, file, timeout_s=timeout_s)
        results.append(CachedFile(target, file, downloaded=True))
    return results


def _verifies(path: Path, file: FileEntry) -> bool:
    if not path.is_file():
        return False
    try:
        verify_file(path, file)
    except ManifestMismatchError:
        return False  # corrupt or partial copy in the cache: fetch it again
    return True


def _download(url: str, target: Path, file: FileEntry, *, timeout_s: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    shutil.copyfileobj(response, out, length=1 << 20)
            except OSError as exc:  # URLError, HTTPError and timeouts are OSErrors
                raise HfDataError(f"download failed: {url}: {exc}") from exc
        verify_file(tmp, file)
        _check_parquet(tmp, url)
        if not file.pinned:
            _pin(tmp, file)
        _move_into_cache(tmp, target, file)
    finally:
        tmp.unlink(missing_ok=True)


def _move_into_cache(tmp: Path, target: Path, file: FileEntry) -> None:
    """Rename the verified download into place.

    On Windows the rename fails with PermissionError while another process (another checkout
    sharing the cache) has the target open or is renaming its own copy. Retry briefly; if the
    other process has meanwhile put a verified copy there, use it.
    """
    error: PermissionError | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except PermissionError as exc:
            if _verifies(target, file):
                return
            error = exc
            time.sleep(_REPLACE_DELAY_S * (attempt + 1))
    raise HfDataError(
        f"cannot move the download into {target}: {error}. Another process may have the file "
        "open (shared cache, HF_DATA_DIR); close it and run the download again"
    ) from error


def _check_parquet(path: Path, url: str) -> None:
    try:
        parquet_rows(path)
    except Exception as exc:  # pyarrow raises several types for a non-parquet body
        raise ManifestMismatchError(f"{url} did not return a parquet file: {exc}") from exc


def _pin(path: Path, file: FileEntry) -> None:
    if file.size is None:
        file.size = path.stat().st_size
    if file.sha256 is None:
        file.sha256 = sha256_file(path)
    if file.rows is None:
        file.rows = parquet_rows(path)


def read_dataset(entry: DatasetEntry, data_dir: str | Path) -> pa.Table:
    """Verify the cached files of `entry` (no network) and read the mapped columns.

    Only pinned files are read, so nothing unverified reaches the readers. Files are concatenated
    in manifest order, so row order is the pinned files' order. The total row count must match
    `entry.rows`.
    """
    unpinned = [f.path for f in entry.files if not f.pinned]
    if unpinned:
        raise NotPinnedError(f"{entry.repo_id}: {unpinned} not pinned in the manifest")
    tables = []
    wanted = list(entry.columns.values())
    for file, path in zip(entry.files, dataset_paths(entry, data_dir), strict=True):
        verify_file(path, file)
        names = pq.read_schema(path).names
        absent = [c for c in wanted if c not in names]
        if absent:
            raise ManifestMismatchError(f"{path}: columns {absent} missing (has {names})")
        tables.append(pq.read_table(path, columns=wanted))
    table = pa.concat_tables(tables)
    if table.num_rows != entry.rows:
        raise ManifestMismatchError(
            f"{entry.repo_id}: {table.num_rows} rows != manifest rows {entry.rows}"
        )
    return table
