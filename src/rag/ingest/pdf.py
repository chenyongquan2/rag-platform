"""Readers `pdf` and `image`: a PDF page or an image file -> one Document (spec P6-01,
design s5-2, s5-3, s5-4).

`pdf`: the connector (`rag.data.pdf`) hands over the Markdown of one page from Docling and the
figure images of the page. The reader describes each figure with a vision model and writes the
description into the text at the figure's place (`[Image description (written by a model, not
part of the original document): ...]`), so BM25 and the vectors both find it and the prompt can
tell it from the document's own words. A figure below `min_image_px` (an icon, a rule line) is
left out, and so is every figure when `describe_images` is `off`. The descriptions are cached by
the image's hash (`rag.ingest.describe`): reading a page again costs nothing and gives the same
text, so the page hash stays stable. A model error on one figure leaves that figure without a
description (counted as `images_failed`); a budget stop is raised, the run must not go on with
half the figures described.

Document fields: `doc_id` `<kb>:<file>#p<N>`, `source_rel` the file, `unit` `p<N>`, `page` N (for
"page N" in citations), `title_path` the headings open at the start of the page (the page's own
headings are added by `markdown_split`; a PDF without headings gets its title), `doc_type` `pdf`,
`images` the figure files of the page.

`image`: a PNG or JPG file; the Document is the description of the picture. Both readers take
their options from `reader_params` of the pipeline in the knowledge base YAML.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from PIL import Image

from rag.data import SourceDocument
from rag.data.pdf.convert import FIGURE_MARKER_RE
from rag.ingest.describe import (
    DescribeBudgetError,
    ModelTextService,
    image_marker,
    shared_service,
)
from rag.ingest.meta import DEFAULT_EMBED_KEYS, DEFAULT_LLM_KEYS, apply_whitelist, join_title
from rag.ingest.readers import METADATA_SEPARATOR, METADATA_TEMPLATE, TEXT_TEMPLATE
from rag.registry import readers

__all__ = ["ImageReader", "PdfReader", "ReadStats"]

_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE)
DESCRIBE_MODES = ("model", "off")


@dataclass
class ReadStats:
    pages: int = 0
    tables: int = 0
    images_seen: int = 0
    images_skipped_small: int = 0
    images_failed: int = 0
    image_files: int = 0


class ReadStatsHolder:
    """`reader.stats`: the counters (`counts`, also readable as attributes) and `to_dict()`
    with the totals of the model service (what `KbProcessor.reader_stats` collects)."""

    def __init__(self, reader: Any) -> None:
        self.counts = ReadStats()
        self._reader = reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self.counts, name)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(vars(self.counts))
        service = self._reader._service
        out["model_text"] = service.stats.to_dict() if service is not None else {}
        return out


class _Base(BaseReader):
    """What both readers share: options, the model service and the counters."""

    def __init__(
        self,
        *,
        kb: str,
        pipeline_version: str,
        language: str = "und",
        acl_groups: Sequence[str] = ("public",),
        embed_keys: Sequence[str] = DEFAULT_EMBED_KEYS,
        llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
        describe_images: str = "model",
        vision_model: str = "",
        cache_path: str = "data/cache/model_text.sqlite3",
        budget_cny: float | None = None,
        min_image_px: int = 64,
        max_image_px: int = 1024,
        service: ModelTextService | None = None,
        settings: Any = None,
    ) -> None:
        if describe_images not in DESCRIBE_MODES:
            raise ValueError(
                f"describe_images must be one of {DESCRIBE_MODES}, got {describe_images!r}"
            )
        if describe_images == "model" and service is None and not vision_model:
            raise ValueError("describe_images `model` needs `vision_model` in reader_params")
        self.kb = kb
        self.pipeline_version = pipeline_version
        self.language = language
        self.acl_groups = list(acl_groups)
        self.embed_keys = tuple(embed_keys)
        self.llm_keys = tuple(llm_keys)
        self.describe_images = describe_images
        self.vision_model = vision_model
        self.cache_path = cache_path
        self.budget_cny = budget_cny
        self.min_image_px = min_image_px
        self.max_image_px = max_image_px
        self._service = service
        self._settings = settings
        self.stats = ReadStatsHolder(self)

    @property
    def service(self) -> ModelTextService | None:
        """The model service, built on the first figure that needs it (no gateway key is read
        when nothing is described)."""
        if self._service is None and self.describe_images == "model":
            self._service = shared_service(
                self._settings,
                vision_model=self.vision_model,
                cache_path=self.cache_path,
                budget_cny=self.budget_cny,
                max_px=self.max_image_px,
            )
        return self._service

    def _describe(self, png: bytes, caption: str) -> str | None:
        """The description of an image, or None when it cannot be had (counted)."""
        service = self.service
        assert service is not None
        try:
            return service.describe_image(png, caption=caption, language=self.language)
        except DescribeBudgetError:
            raise
        except Exception:
            self.stats.counts.images_failed += 1
            return None

    def _document(self, source: SourceDocument, text: str, metadata: dict[str, Any]) -> Document:
        doc = Document(
            id_=source.doc_id,
            text=text,
            metadata=metadata,
            metadata_template=METADATA_TEMPLATE,
            metadata_separator=METADATA_SEPARATOR,
            text_template=TEXT_TEMPLATE,
        )
        apply_whitelist(doc, self.embed_keys, self.llm_keys)
        return doc


@readers.register("pdf")
class PdfReader(_Base):
    """One PDF page -> one Document (see the module doc)."""

    def load_data(self, source: SourceDocument, **_: Any) -> list[Document]:  # type: ignore[override]
        extra = source.extra
        text = source.content.decode("utf-8")
        figures = {int(f["index"]): f for f in extra.get("figures", [])}
        counts = self.stats.counts

        def place(match: re.Match[str]) -> str:
            figure = figures.get(int(match.group(1)))
            if figure is None:
                return ""
            counts.images_seen += 1
            if min(int(figure["width"]), int(figure["height"])) < self.min_image_px:
                counts.images_skipped_small += 1
                return ""
            if self.describe_images == "off":
                return ""
            with open(figure["path"], "rb") as fh:
                described = self._describe(fh.read(), str(figure.get("caption", "")))
            return image_marker(described) if described else ""

        body = FIGURE_MARKER_RE.sub(place, text)
        body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")
        carried = list(extra.get("carried", []))
        if not carried and not _HEADING_LINE.search(body):
            carried = [str(extra.get("title") or source.source_rel)]
        counts.pages += 1
        counts.tables += int(extra.get("tables", 0))
        page = int(extra["page"])
        metadata: dict[str, Any] = {
            "kb": self.kb,
            "doc_id": source.doc_id,
            "title_path": join_title(carried),
            "source_rel": source.source_rel,
            "unit": f"p{page}",
            "source_url": source.source_url,
            "doc_type": "pdf",
            "language": self.language,
            "acl_groups": list(self.acl_groups),
            "images": [str(f.get("rel", f["path"])) for f in extra.get("figures", [])],
            "pipeline_version": self.pipeline_version,
            "page": page,
        }
        return [self._document(source, body, metadata)]


@readers.register("image")
class ImageReader(_Base):
    """One PNG or JPG file -> one Document that is the description of the picture."""

    def load_data(self, source: SourceDocument, **_: Any) -> list[Document]:  # type: ignore[override]
        try:
            image = Image.open(io.BytesIO(source.content))
            image.load()
        except Exception as exc:
            raise ValueError(f"{source.source_rel}: not an image file ({exc})") from exc
        counts = self.stats.counts
        counts.image_files += 1
        counts.images_seen += 1
        text = ""
        if min(image.size) < self.min_image_px:
            counts.images_skipped_small += 1
        elif self.describe_images == "model":
            described = self._describe(source.content, "")
            text = image_marker(described) if described else ""
        stem = source.source_rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        metadata: dict[str, Any] = {
            "kb": self.kb,
            "doc_id": source.doc_id,
            "title_path": stem,
            "source_rel": source.source_rel,
            "unit": None,
            "source_url": source.source_url,
            "doc_type": "image",
            "language": self.language,
            "acl_groups": list(self.acl_groups),
            "images": [source.source_rel],
            "pipeline_version": self.pipeline_version,
        }
        return [self._document(source, text, metadata)]
