"""`pdf_dir` connector: a folder of PDF files and images as `SourceDocument`s (spec P6-01,
design s5-1, s5-2, s5-4).

    source:
      connector: pdf_dir
      path: data/raw/pdf_demo        # PDF, PNG, JPG files (not recursive)
      files: null                    # or a list of file names
      limit: null                    # first N files by name
      docling: {ocr: false, table_structure: true, images_scale: 2.0, num_threads: 4}
      cache_dir: data/cache/pdf      # converted pages, keyed by file hash and Docling version

A PDF gives one `SourceDocument` per page (the document id of a page is `<kb>:<file>#p<N>`; the
unit of s5-4 is the page number): Docling converts the file once, the result is cached by file
hash (`rag.data.pdf.convert`), so listing the folder again costs a few file reads. `content` is
the page's Markdown, `extra` holds what the reader needs: `page`, `pages`, `title`, `carried`
(headings open at the start of the page), `figures` (index, path of the image file, size,
caption) and `tables`. An image file gives one `SourceDocument` with the file's bytes.
`source_version` is the file's sha256 (shortened), so an audit shows which file a chunk came from.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from rag.config import ServiceSettings, repo_path
from rag.data.base import ConnectorMode, SourceDocument, make_doc_id
from rag.data.hf.connector import check_limit
from rag.data.pdf.convert import ConvertedPdf, DoclingOptions, Engine, convert_pdf, file_sha256
from rag.registry import connectors

PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _options(docling: Mapping[str, Any] | None) -> DoclingOptions:
    return DoclingOptions(**dict(docling or {}))


@connectors.register("pdf_dir")
class PdfDirConnector:
    """Full-snapshot connector over a folder of PDFs and images (see the module doc)."""

    mode: ClassVar[ConnectorMode] = "snapshot"

    def __init__(
        self,
        *,
        kb: str,
        path: str | Path,
        files: Sequence[str] | None = None,
        limit: int | None = None,
        docling: Mapping[str, Any] | None = None,
        cache_dir: str | Path = "data/cache/pdf",
        settings: ServiceSettings | None = None,
        engine: Engine | None = None,
    ) -> None:
        self.kb = kb
        self.root = repo_path(path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root}: the folder of the PDF files does not exist")
        self.files = list(files) if files is not None else None
        self.limit = check_limit(limit)
        self.options = _options(docling)
        self.cache_dir = repo_path(cache_dir)
        self.settings = settings if settings is not None else ServiceSettings()
        self.engine = engine
        self.conversions: dict[str, dict[str, Any]] = {}
        """Per PDF file of the last listing: pages, figures, tables, seconds, cached."""
        self.failures: dict[str, str] = {}
        """PDF files that could not be converted, with the reason (one line each)."""
        self._failed: dict[tuple[str, str], str] = {}

    def _paths(self) -> list[Path]:
        suffixes = (*PDF_EXTENSIONS, *IMAGE_EXTENSIONS)
        if self.files is not None:
            chosen = [self.root / name for name in self.files]
            root = self.root.resolve()
            for name, path in zip(self.files, chosen, strict=True):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"{name!r} in `files` is outside the folder {self.root}")
            for path in chosen:
                if not path.is_file():
                    raise FileNotFoundError(f"{path}: listed in `files` but not found")
        else:
            chosen = [
                p
                for p in sorted(self.root.iterdir())
                if p.is_file() and p.suffix.lower() in suffixes
            ]
        return chosen if self.limit is None else chosen[: self.limit]

    def documents(self) -> Iterator[SourceDocument]:
        for path in self._paths():
            if path.suffix.lower() in PDF_EXTENSIONS:
                yield from self._pdf_or_failure(path)
            else:
                yield self._image(path)

    def doc_ids(self) -> list[str]:
        return [doc.doc_id for doc in self.documents()]

    def _image(self, path: Path) -> SourceDocument:
        raw = path.read_bytes()
        return SourceDocument(
            doc_id=make_doc_id(self.kb, path.name),
            kb=self.kb,
            source_rel=path.name,
            source_url=None,
            ext=path.suffix.lower(),
            content=raw,
            source_version=f"sha256:{file_sha256(path)[:16]}",
        )

    def _pdf_or_failure(self, path: Path) -> list[SourceDocument]:
        """The page documents of a PDF; when the file cannot be converted (corrupt, too large,
        too many pages, too slow) one document that stands for the file and carries the reason,
        which the reader raises: the ingest job then counts a read failure and goes on with the
        other files. A file that failed is not tried again in this process."""
        digest = file_sha256(path)
        key = (path.name, digest)
        reason = self._failed.get(key)
        if reason is None:
            try:
                return list(self._pdf(path))
            except Exception as exc:  # any failure of one file must not stop the others
                reason = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
                self._failed[key] = reason
        self.failures[path.name] = reason
        return [
            SourceDocument(
                doc_id=make_doc_id(self.kb, path.name),
                kb=self.kb,
                source_rel=path.name,
                source_url=None,
                ext=".pdf",
                content=b"",
                source_version=f"sha256:{digest[:16]}",
                extra={"conversion_error": reason},
            )
        ]

    def _pdf(self, path: Path) -> Iterator[SourceDocument]:
        started = time.perf_counter()
        converted: ConvertedPdf = convert_pdf(
            path, cache_dir=self.cache_dir, options=self.options, engine=self.engine
        )
        self.conversions[path.name] = {
            "pages": len(converted.pages),
            "figures": sum(len(p.figures) for p in converted.pages),
            "tables": sum(p.tables for p in converted.pages),
            "engine_seconds": round(converted.seconds, 2),
            "recorded_engine_seconds": round(converted.recorded_seconds, 2),
            "cached": converted.cached,
            "wall_seconds": round(time.perf_counter() - started, 2),
            "docling_version": converted.docling_version,
        }
        version = f"sha256:{converted.sha256[:16]}"
        for page in converted.pages:
            figures = [
                {
                    "index": ref.index,
                    "path": str(converted.figure_path(ref)),
                    "rel": f"{converted.folder.name}/{ref.path}",
                    "sha256": ref.sha256,
                    "width": ref.width,
                    "height": ref.height,
                    "caption": ref.caption,
                }
                for ref in page.figures
            ]
            yield SourceDocument(
                doc_id=make_doc_id(self.kb, path.name, f"p{page.number}"),
                kb=self.kb,
                source_rel=path.name,
                source_url=None,
                ext=".pdf",
                content=page.markdown.encode("utf-8"),
                source_version=version,
                extra={
                    "page": page.number,
                    "pages": len(converted.pages),
                    "title": converted.title,
                    "carried": list(page.carried),
                    "figures": figures,
                    "tables": page.tables,
                },
            )
