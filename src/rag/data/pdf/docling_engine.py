"""The Docling engine of `rag.data.pdf.convert` (spec P6-01).

Docling is imported here only, on the first conversion: importing it takes seconds and its models
need about 2 GB of memory. The layout and table models are read from the local Hugging Face cache
(`HF_HUB_OFFLINE=1` unless the caller set it); the first run on a new machine needs
`HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0` once.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from rag.data.pdf.convert import (
    PLACEHOLDER,
    ConversionError,
    DoclingOptions,
    RawFigure,
    RawPage,
)


def _png(image: Any) -> tuple[bytes, int, int]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), int(image.width), int(image.height)


def docling_engine(path: Path, options: DoclingOptions) -> list[RawPage]:
    """Convert `path` with Docling: Markdown per page and the figure images in reading order."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline = PdfPipelineOptions()
    pipeline.do_ocr = options.ocr
    pipeline.do_table_structure = options.table_structure
    pipeline.generate_picture_images = True
    pipeline.images_scale = options.images_scale
    pipeline.accelerator_options = AcceleratorOptions(num_threads=options.num_threads, device="cpu")
    pipeline.document_timeout = options.timeout_s
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    result = converter.convert(
        str(path),
        max_num_pages=options.max_pages,
        max_file_size=int(options.max_file_mb * 1024 * 1024),
    )
    if result.status != ConversionStatus.SUCCESS:  # a partial result (timeout) is not cached
        raise ConversionError(
            f"{path.name}: Docling finished with status {result.status.value} "
            f"(timeout_s {options.timeout_s}); the file is not used"
        )
    document = result.document
    pictures: dict[int, list[Any]] = {}
    for picture in document.pictures:
        if picture.prov:
            pictures.setdefault(picture.prov[0].page_no, []).append(picture)
    pages: list[RawPage] = []
    for number in sorted(document.pages):
        markdown = document.export_to_markdown(page_no=number, image_placeholder=PLACEHOLDER)
        figures: list[RawFigure] = []
        for picture in pictures.get(number, []):
            image = picture.get_image(document)
            if image is None:  # keep the figure count equal to the placeholder count
                figures.append(RawFigure(b"", 0, 0, picture.caption_text(document)))
                continue
            data, width, height = _png(image)
            figures.append(RawFigure(data, width, height, picture.caption_text(document)))
        pages.append(RawPage(number, markdown.strip("\n"), figures))
    return pages
