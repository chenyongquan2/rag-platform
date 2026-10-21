"""PDF -> pages of Markdown plus figure images, cached by file hash (spec P6-01, design s5-2).

`convert_pdf` runs an engine (Docling, `rag.data.pdf.docling_engine`) once per file and keeps the
result under `<cache_dir>/<key>/`: `converted.json` (one entry per page) and the figure images as
PNG files. The key holds the file's sha256, the Docling version, our own post-processing version
and the options that change the output, so a new Docling or other options convert again and
everything else is read from disk. Docling is slow on a CPU; the cache is what makes tests and
re-ingests fast.

What the engine hands over (`RawPage`): the Markdown of each page (tables as Markdown tables,
every figure as `<!-- image -->`) and the figure images in the same order. This module then

- numbers the figures of a page (`<!-- figure N -->`, N counts from 1 on each page), so the
  reader can describe or skip a figure by number and the description lands at the figure's place;
- turns the headings into a hierarchy (`normalise_headings`): Docling writes every heading of a
  PDF as `##`; the first heading is the title (`#`), a numbered heading is as deep as its number
  (`3.1` is level 3), any other heading is level 2;
- works out for each page which headings are still open when the page starts (`carried`), so a
  page that continues a section knows the section it belongs to.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

CONVERT_VERSION = "1"
"""Raise when this module changes what it writes for the same engine output."""
PLACEHOLDER = "<!-- image -->"
"""What Docling writes for a figure (`export_to_markdown`)."""
FIGURE_MARKER = "<!-- figure {n} -->"
FIGURE_MARKER_RE = re.compile(r"<!-- figure (\d+) -->")


class ConversionError(RuntimeError):
    """The engine's output cannot be used (figures and placeholders do not match, no pages)."""


def docling_version() -> str:
    try:
        return metadata.version("docling")
    except metadata.PackageNotFoundError:  # pragma: no cover - docling is a locked dependency
        return "unknown"


@dataclass(frozen=True)
class DoclingOptions:
    """Docling settings. `num_threads` only changes the speed, and the three limits only decide
    whether a file is read, so none of them is in the cache key."""

    ocr: bool = False
    table_structure: bool = True
    images_scale: float = 1.0
    num_threads: int = 4
    max_file_mb: float = 50.0
    """A larger file is refused before Docling starts (a failed file, not a stopped run)."""
    max_pages: int = 300
    """A file with more pages is refused (Docling stops at this many pages)."""
    timeout_s: float | None = 900.0
    """Docling's time for one file; a file that takes longer is a failed file. None: no limit."""

    def fingerprint(self) -> str:
        text = f"ocr={self.ocr};tables={self.table_structure};scale={self.images_scale}"
        return hashlib.sha256(text.encode()).hexdigest()[:8]


@dataclass
class RawFigure:
    png: bytes
    width: int
    height: int
    caption: str = ""


@dataclass
class RawPage:
    number: int
    markdown: str
    figures: list[RawFigure] = field(default_factory=list)


class Engine(Protocol):
    def __call__(self, path: Path, options: DoclingOptions) -> list[RawPage]: ...


@dataclass
class FigureRef:
    """A figure image kept on disk; `path` is relative to the cache folder of the file."""

    index: int
    path: str
    sha256: str
    width: int
    height: int
    caption: str = ""


@dataclass
class PdfPage:
    number: int
    markdown: str
    figures: list[FigureRef]
    carried: list[str]
    """Headings open at the start of the page that are above the page's first heading, outermost
    first (all open headings when the page has no heading)."""
    tables: int = 0


@dataclass
class ConvertedPdf:
    sha256: str
    docling_version: str
    title: str
    pages: list[PdfPage]
    folder: Path
    seconds: float = 0.0
    """Time the engine took in this call; 0.0 when the result came from the cache."""
    cached: bool = False
    recorded_seconds: float = 0.0
    """Time the engine took when the result was made (also when it is read from the cache)."""

    def figure_path(self, ref: FigureRef) -> Path:
        return self.folder / ref.path


# ---- headings -------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?[ \t]+\S")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def heading_level(title: str, *, first: bool) -> int:
    """Level of a heading from its text: the document's first heading is the title (1); a
    numbered heading is as deep as its number plus one (`3` is 2, `3.1` is 3); others are 2."""
    if first and not _NUMBERED.match(title):
        return 1
    numbered = _NUMBERED.match(title)
    if numbered:
        return min(6, numbered.group(1).count(".") + 2)
    return 2


def normalise_headings(pages: list[str]) -> tuple[list[str], list[list[tuple[int, str]]]]:
    """Rewrite the heading levels of every page and list each page's headings `(level, title)`.

    `#` lines inside a fenced code block are text. The first heading of the document is the title
    unless it is numbered.
    """
    out: list[str] = []
    found: list[list[tuple[int, str]]] = []
    seen_any = False
    for text in pages:
        lines: list[str] = []
        headings: list[tuple[int, str]] = []
        fenced = False
        for line in text.split("\n"):
            if _FENCE.match(line):
                fenced = not fenced
            match = None if fenced else _HEADING.match(line)
            if match is None:
                lines.append(line)
                continue
            title = match.group(2).strip()
            level = heading_level(title, first=not seen_any)
            seen_any = True
            headings.append((level, title))
            lines.append(f"{'#' * level} {title}")
        out.append("\n".join(lines))
        found.append(headings)
    return out, found


def carried_titles(per_page: list[list[tuple[int, str]]]) -> list[list[str]]:
    """For each page, the headings still open at its start that sit above its first heading."""
    stack: list[tuple[int, str]] = []
    carried: list[list[str]] = []
    for headings in per_page:
        limit = headings[0][0] if headings else 7
        carried.append([title for level, title in stack if level < limit])
        for level, title in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
    return carried


def count_tables(markdown: str) -> int:
    """Markdown tables on a page: runs of table rows that have a separator row."""
    tables = 0
    in_table = False
    for line in markdown.split("\n"):
        if _TABLE_ROW.match(line):
            if not in_table and _TABLE_SEPARATOR.match(line) is None:
                in_table = True
                tables += 1
        else:
            in_table = False
    return tables


# ---- conversion and cache --------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_folder(cache_dir: Path, sha256: str, options: DoclingOptions) -> Path:
    key = f"{sha256[:16]}-docling{docling_version()}-c{CONVERT_VERSION}-{options.fingerprint()}"
    return cache_dir / key


def _number_figures(page: RawPage) -> str:
    placeholders = page.markdown.count(PLACEHOLDER)
    if placeholders != len(page.figures):
        raise ConversionError(
            f"page {page.number}: {placeholders} figure placeholders but {len(page.figures)} "
            "figure images; the engine's output cannot be matched"
        )
    parts = page.markdown.split(PLACEHOLDER)
    text = parts[0]
    for n, rest in enumerate(parts[1:], start=1):
        text += FIGURE_MARKER.format(n=n) + rest
    return text


def _build(path: Path, sha256: str, options: DoclingOptions, engine: Engine, folder: Path) -> None:
    started = time.perf_counter()
    raw = engine(path, options)
    seconds = time.perf_counter() - started
    if not raw:
        raise ConversionError(f"{path.name}: the engine returned no pages")
    if len(raw) > options.max_pages:
        raise ConversionError(
            f"{path.name}: {len(raw)} pages, more than max_pages ({options.max_pages})"
        )
    raw = sorted(raw, key=lambda p: p.number)
    numbered = [_number_figures(p) for p in raw]
    normalised, headings = normalise_headings(numbered)
    carried = carried_titles(headings)
    title = next((h[0][1] for h in headings if h), path.stem)
    tmp = folder.with_name(folder.name + ".tmp")
    if tmp.exists():
        for old in tmp.iterdir():
            old.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for page, text, open_titles in zip(raw, normalised, carried, strict=True):
        refs = []
        for n, figure in enumerate(page.figures, start=1):
            name = f"p{page.number}_f{n}.png"
            (tmp / name).write_bytes(figure.png)
            refs.append(
                {
                    "index": n,
                    "path": name,
                    "sha256": hashlib.sha256(figure.png).hexdigest(),
                    "width": figure.width,
                    "height": figure.height,
                    "caption": figure.caption,
                }
            )
        pages.append(
            {
                "number": page.number,
                "markdown": text,
                "figures": refs,
                "carried": open_titles,
                "tables": count_tables(text),
            }
        )
    document = {
        "sha256": sha256,
        "docling_version": docling_version(),
        "convert_version": CONVERT_VERSION,
        "title": title,
        "seconds": round(seconds, 2),
        "pages": pages,
    }
    payload = json.dumps(document, ensure_ascii=False, indent=1)
    (tmp / "converted.json").write_bytes(payload.encode("utf-8"))
    if folder.exists():
        for old in folder.iterdir():
            old.unlink()
        folder.rmdir()
    tmp.rename(folder)


def load_converted(folder: Path, *, cached: bool) -> ConvertedPdf:
    data = json.loads((folder / "converted.json").read_bytes().decode("utf-8"))
    pages = [
        PdfPage(
            number=int(p["number"]),
            markdown=str(p["markdown"]),
            figures=[FigureRef(**f) for f in p["figures"]],
            carried=list(p["carried"]),
            tables=int(p.get("tables", 0)),
        )
        for p in data["pages"]
    ]
    return ConvertedPdf(
        sha256=str(data["sha256"]),
        docling_version=str(data["docling_version"]),
        title=str(data["title"]),
        pages=pages,
        folder=folder,
        seconds=0.0 if cached else float(data.get("seconds", 0.0)),
        cached=cached,
        recorded_seconds=float(data.get("seconds", 0.0)),
    )


def convert_pdf(
    path: str | Path,
    *,
    cache_dir: str | Path,
    options: DoclingOptions | None = None,
    engine: Engine | Callable[[Path, DoclingOptions], list[RawPage]] | None = None,
) -> ConvertedPdf:
    """The pages of a PDF; the engine runs only when the cache has no result for this file hash,
    Docling version and options. `engine` defaults to Docling (`rag.data.pdf.docling_engine`)."""
    path = Path(path)
    options = options or DoclingOptions()
    sha256 = file_sha256(path)
    folder = cache_folder(Path(cache_dir), sha256, options)
    if (folder / "converted.json").is_file():
        return load_converted(folder, cached=True)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > options.max_file_mb:
        raise ConversionError(
            f"{path.name}: {size_mb:.1f} MB, larger than max_file_mb ({options.max_file_mb})"
        )
    if engine is None:
        from rag.data.pdf.docling_engine import docling_engine

        engine = docling_engine
    _build(path, sha256, options, engine, folder)
    return load_converted(folder, cached=False)
