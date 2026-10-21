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
    chart = io.BytesIO()
    pressure_chart().save(chart, format="PNG")
    chart.seek(0)
    pdf.image(chart, x=x0, y=y0, w=120)
    pdf.set_xy(25, y0 + 86)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Figure 1: Pressure rises quickly and then levels off.")
    path.write_bytes(bytes(pdf.output()))
    return path


def pressure_chart() -> Image.Image:
    """A raster line chart for the test PDF (an embedded image, which Docling finds as a figure)."""
    image = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.line([40, 20, 40, 280], fill="black", width=3)
    draw.line([40, 280, 460, 280], fill="black", width=3)
    draw.line([40, 250, 160, 90, 460, 50], fill=(200, 0, 0), width=5)
    return image


DEMO_CHART_VALUES = {"Q1": 12, "Q2": 18, "Q3": 9, "Q4": 25}
"""The numbers drawn in the demo chart (million USD); the evaluation questions rely on them."""


def make_demo_chart(path: Path) -> Path:
    """The standalone image of the `pdf_demo` knowledge base: a bar chart of quarterly revenue
    drawn here (no outside source), so the answers to questions about it are known."""
    from PIL import ImageFont

    width, height = 640, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=22)
    label_font = ImageFont.load_default(size=18)
    draw.text(
        (40, 16), "Quarterly revenue (million USD), fiscal year 2025", fill="black", font=title_font
    )
    left, bottom, top = 70, 360, 80
    draw.line([left, top, left, bottom], fill="black", width=2)
    draw.line([left, bottom, width - 30, bottom], fill="black", width=2)
    unit = (bottom - top) / 30
    for i, (name, value) in enumerate(DEMO_CHART_VALUES.items()):
        x0 = left + 40 + i * 130
        y0 = bottom - value * unit
        draw.rectangle([x0, y0, x0 + 80, bottom], fill=(40, 90, 200))
        draw.text((x0 + 22, y0 - 26), str(value), fill="black", font=label_font)
        draw.text((x0 + 26, bottom + 8), name, fill="black", font=label_font)
    for tick in (0, 10, 20, 30):
        y = bottom - tick * unit
        draw.line([left - 6, y, left, y], fill="black", width=2)
        draw.text((left - 40, y - 9), str(tick), fill="black", font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path
