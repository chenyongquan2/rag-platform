"""Test helpers for the PDF and image sources (spec P6-01): a counting fake conversion engine,
fake model clients, tiny PNG images and a tiny PDF made with fpdf2 (a dev dependency)."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from rag.data.pdf.convert import DoclingOptions, RawFigure, RawPage
from rag.ingest.describe import ModelReply


def tiny_png(
    width: int = 160, height: int = 120, *, color: tuple[int, int, int] = (40, 90, 200)
) -> bytes:
    """A small PNG with a coloured box; different colours give different bytes (hashes)."""
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle([10, 10, width - 10, height - 10], fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    """A conversion engine that returns the given pages and counts its calls."""

    def __init__(self, pages: Sequence[RawPage]) -> None:
        self.pages = list(pages)
        self.calls = 0

    def __call__(self, path: Path, options: DoclingOptions) -> list[RawPage]:
        self.calls += 1
        return [
            RawPage(
                p.number,
                p.markdown,
                [RawFigure(f.png, f.width, f.height, f.caption) for f in p.figures],
            )
            for p in self.pages
        ]


class FakeVision:
    """An image describer that answers a fixed text and counts its calls."""

    model = "fake-vision"

    def __init__(self, text: str = "A diagram of a small test figure.", cny: float = 0.003) -> None:
        self.text = text
        self.cny = cny
        self.calls = 0
        self.seen: list[tuple[str, str]] = []

    def describe(self, png: bytes, *, caption: str, language: str) -> ModelReply:
        self.calls += 1
        self.seen.append((caption, language))
        return ModelReply(self.text, prompt_tokens=100, completion_tokens=30, cny=self.cny)


class FakeSummarizer:
    """A table summarizer that answers a fixed text and counts its calls."""

    model = "fake-summary"

    def __init__(self, text: str = "A small table of test values.", cny: float = 0.002) -> None:
        self.text = text
        self.cny = cny
        self.calls = 0
        self.tables: list[str] = []

    def summarize(self, table_markdown: str, *, caption: str) -> ModelReply:
        self.calls += 1
        self.tables.append(table_markdown)
        return ModelReply(self.text, prompt_tokens=100, completion_tokens=30, cny=self.cny)


def sample_pages() -> list[RawPage]:
    """Three pages: a title and a numbered section with a figure and a table, a continuation
    page without headings, and a page that opens a deeper section."""
    red = tiny_png(color=(200, 40, 40))
    blue = tiny_png(color=(40, 40, 200))
    return [
        RawPage(
            1,
            "## Fast Training of Small Nets\n\n## Abstract\n\nWe train small nets quickly.\n\n"
            "## 1 Introduction\n\nSmall nets are useful.\n\n"
            "Figure 1: Training speed.\n\n<!-- image -->\n\n"
            "Table 1: Speed by model.\n\n| Model | Speed |\n|---|---|\n| A | 10 |\n| B | 20 |",
            [RawFigure(red, 160, 120, "Figure 1: Training speed.")],
        ),
        RawPage(2, "The introduction goes on here.\n\nMore text.", []),
        RawPage(
            3,
            "## 2.1 Details\n\nDetails of the method.\n\n<!-- image -->\n\n<!-- image -->",
            [RawFigure(blue, 160, 120, ""), RawFigure(red, 32, 32, "")],
        ),
    ]


def make_test_pdf(path: Path) -> Path:
    """A two-page PDF with a title, a heading, a table and a drawn figure (needs fpdf2)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Tiny Report on Pumps", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1 Introduction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6, "This report compares three pumps. The Zephyr pump moves the most water per hour."
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Table 1: Pump flow rates.", new_x="LMARGIN", new_y="NEXT")
    rows = [
        ("Pump", "Flow (l/min)", "Power (W)"),
        ("Zephyr", "120", "450"),
        ("Boreas", "95", "380"),
        ("Notus", "60", "210"),
    ]
    for number, row in enumerate(rows):
        pdf.set_font("Helvetica", "B" if number == 0 else "", 11)
        for cell in row:
            pdf.cell(55, 8, cell, border=1)
        pdf.ln(8)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2 Results", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Figure 1 shows the pressure over time for the Zephyr pump.")
    x0, y0 = 30, 60
    pdf.set_draw_color(0, 0, 0)
    pdf.line(x0, y0 + 60, x0 + 120, y0 + 60)
    pdf.line(x0, y0, x0, y0 + 60)
    pdf.set_draw_color(200, 0, 0)
    pdf.set_line_width(0.8)
    pdf.line(x0, y0 + 50, x0 + 40, y0 + 20)
    pdf.line(x0 + 40, y0 + 20, x0 + 120, y0 + 10)
    pdf.set_xy(25, y0 + 66)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Figure 1: Pressure rises quickly and then levels off.")
    path.write_bytes(bytes(pdf.output()))
    return path
