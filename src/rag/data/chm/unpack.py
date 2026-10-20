"""Unpack a `.chm` file into a folder (spec P4-01, design s5-2): the thin, tool-specific step.

Which tool works is a fact about the machine, so it hides behind `ChmUnpacker`:
- `7z` (7-Zip, also `7zz`, `7za`): reads CHM natively, handles the Unicode file names, runs on
  Windows and in a Linux container (`p7zip`); first choice. Windows default install paths are
  tried when it is not on `PATH`.
- `extract_chmLib` (chmlib, Linux/macOS): second choice.
- `hh.exe -decompile` (Windows only, what the first experiments used): last choice.
The folder is then read by `DirectoryChmSource`; a manifest `_unpack.json` next to the pages
records the CHM's size and SHA-256 (the `source_version` of its pages), the tool and the counts.
It holds no absolute paths and no page text.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rag.data.chm.source import ChmError, DirectoryChmSource

MANIFEST_NAME = "_unpack.json"
UNPACK_TIMEOUT_S = 600.0
SEVEN_ZIP_NAMES = ("7z", "7zz", "7za")
SEVEN_ZIP_WINDOWS_PATHS = (
    "C:/Program Files/7-Zip/7z.exe",
    "C:/Program Files (x86)/7-Zip/7z.exe",
)
HH_EXE = "C:/Windows/hh.exe"


class UnpackError(ChmError):
    """The CHM could not be unpacked (no tool, tool failed, nothing came out)."""


class ChmUnpacker(Protocol):
    name: str

    def unpack(self, chm: Path, out_dir: Path) -> None: ...


def _run(cmd: list[str], tool: str) -> None:
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=UNPACK_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UnpackError(f"{tool} could not run: {type(exc).__name__}: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().replace("\n", " ")[:300]
        raise UnpackError(f"{tool} failed with exit code {done.returncode}: {detail}")


@dataclass(frozen=True)
class SevenZipUnpacker:
    exe: str = "7z"
    name: str = "7z"

    def unpack(self, chm: Path, out_dir: Path) -> None:
        _run([self.exe, "x", "-y", f"-o{out_dir}", str(chm)], "7z")


@dataclass(frozen=True)
class ChmLibUnpacker:
    exe: str = "extract_chmLib"
    name: str = "extract_chmLib"

    def unpack(self, chm: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _run([self.exe, str(chm), str(out_dir)], "extract_chmLib")


@dataclass(frozen=True)
class HhDecompileUnpacker:
    exe: str = HH_EXE
    name: str = "hh.exe"

    def unpack(self, chm: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _run([self.exe, "-decompile", str(out_dir), str(chm)], "hh.exe")


def find_unpacker(
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.exists,
) -> ChmUnpacker:
    """The first tool this machine has (see the module doc); `UnpackError` names all of them."""
    for name in SEVEN_ZIP_NAMES:
        if exe := which(name):
            return SevenZipUnpacker(exe)
    for path in SEVEN_ZIP_WINDOWS_PATHS:
        if exists(path):
            return SevenZipUnpacker(path)
    if exe := which("extract_chmLib"):
        return ChmLibUnpacker(exe)
    if exists(HH_EXE):
        return HhDecompileUnpacker()
    raise UnpackError(
        "no CHM unpacker found; tried 7z/7zz/7za (also the 7-Zip install folders), "
        "extract_chmLib and hh.exe. Install 7-Zip (Linux: p7zip-full) or chmlib"
    )


@dataclass(frozen=True)
class UnpackResult:
    tool: str
    chm_sha256: str
    chm_bytes: int
    files: int
    pages: int
    out_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_unpack_manifest(folder: Path) -> dict[str, object] | None:
    """The manifest `unpack_chm` wrote into `folder`, None when there is none."""
    path = folder / MANIFEST_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text("utf-8"))
    return data if isinstance(data, dict) else None


def unpack_chm(
    chm: str | Path,
    out_dir: str | Path,
    *,
    unpacker: ChmUnpacker | None = None,
    overwrite: bool = False,
) -> UnpackResult:
    """Unpack `chm` into `out_dir` and write the manifest. A folder that already has files is
    refused unless `overwrite` (files of the same name are then replaced, others stay)."""
    chm, out_dir = Path(chm), Path(out_dir)
    if not chm.is_file():
        raise UnpackError(f"no such file: {chm.name}")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise UnpackError(f"{out_dir.name} is not empty; pass overwrite to unpack over it")
    tool = unpacker or find_unpacker()
    tool.unpack(chm, out_dir)
    source = DirectoryChmSource(out_dir)
    pages = len(source.names())
    if pages == 0:
        raise UnpackError(f"{tool.name} produced no pages in {out_dir.name}")
    files = sum(1 for p in out_dir.rglob("*") if p.is_file())
    result = UnpackResult(
        tool=tool.name,
        chm_sha256=_sha256(chm),
        chm_bytes=chm.stat().st_size,
        files=files,
        pages=pages,
        out_dir=out_dir,
    )
    info = source.info()
    manifest = {
        "chm_name": chm.name,
        "chm_bytes": result.chm_bytes,
        "chm_sha256": result.chm_sha256,
        "tool": tool.name,
        "files": files,
        "pages": pages,
        "toc": source.toc_name,
        "index": source.index_name,
        "lcid": info.lcid,
        "title": info.title,
        "unpacked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline=""
    )
    return result
