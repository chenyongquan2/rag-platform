from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.data.pdf import convert as conv
from rag.data.pdf.convert import (
    ConversionError,
    DoclingOptions,
    RawFigure,
    RawPage,
    carried_titles,
    convert_pdf,
    count_tables,
    heading_level,
    normalise_headings,
)
from rag.testing.pdf import FakeEngine, sample_pages, tiny_png


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4 not a real file, the engine is a fake")
    return path


# ---- headings (AC3) ------------------------------------------------------------------------


def test_normalise_headings_title_numbered_and_plain() -> None:
    pages = ["## My Paper\n\n## Abstract\n\ntext\n\n## 1 Introduction\n\n## 3.1 Training\n"]
    out, found = normalise_headings(pages)
    assert out[0].split("\n")[0] == "# My Paper"
    assert found[0] == [
        (1, "My Paper"),
        (2, "Abstract"),
        (2, "1 Introduction"),
        (3, "3.1 Training"),
    ]
    assert "### 3.1 Training" in out[0]
    assert "## Abstract" in out[0]


def test_normalise_headings_numbered_first_heading_is_not_the_title() -> None:
    out, found = normalise_headings(["## 1 Introduction\n\ntext", "## Appendix"])
    assert found == [[(2, "1 Introduction")], [(2, "Appendix")]]
    assert out[0].startswith("## 1 Introduction")


def test_normalise_headings_leaves_hashes_in_code_fences_alone() -> None:
    page = "## Title\n\n```\n## not a heading\n# comment\n```\n\n## 2 Real"
    out, found = normalise_headings([page])
    assert found[0] == [(1, "Title"), (2, "2 Real")]
    assert "```\n## not a heading\n# comment\n```" in out[0]


def test_heading_level_depth_of_numbers() -> None:
    assert heading_level("4 Experiments", first=False) == 2
    assert heading_level("4.2.1 Accelerating", first=False) == 4
    assert heading_level("4.2.1. Accelerating", first=False) == 4
    assert heading_level("1×1 convolutions", first=False) == 2  # not a section number  # noqa: RUF001
    assert heading_level("Abstract", first=True) == 1


def test_carried_title_keeps_open_sections_above_the_first_heading() -> None:
    per_page = [
        [(1, "Title"), (2, "1 Intro"), (3, "1.1 Background")],
        [],  # a page without headings continues 1.1
        [(3, "1.2 Aim")],  # the first heading is level 3: Title and 1 Intro are above it
        [(2, "2 Method")],
    ]
    assert carried_titles(per_page) == [
        [],
        ["Title", "1 Intro", "1.1 Background"],
        ["Title", "1 Intro"],
        ["Title"],
    ]


def test_count_tables_counts_runs_of_rows() -> None:
    text = "a\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\ntext\n\n| C |\n|---|\n| 3 |"
    assert count_tables(text) == 2
    assert count_tables("no table | here") == 0


# ---- conversion and cache (AC2) --------------------------------------------------------------


def test_convert_numbers_figures_and_fills_the_pages(pdf: Path, tmp_path: Path) -> None:
    engine = FakeEngine(sample_pages())
    result = convert_pdf(pdf, cache_dir=tmp_path / "cache", engine=engine)
    assert result.title == "Fast Training of Small Nets"
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert "<!-- figure 1 -->" in result.pages[0].markdown
    assert "<!-- image -->" not in result.pages[0].markdown
    third = result.pages[2].markdown
    assert third.index("<!-- figure 1 -->") < third.index("<!-- figure 2 -->")
    assert [(f.index, f.width) for f in result.pages[2].figures] == [(1, 160), (2, 32)]
    assert result.figure_path(result.pages[2].figures[0]).read_bytes() == tiny_png(
        color=(40, 40, 200)
    )
    assert result.pages[0].tables == 1
    assert result.pages[1].carried == ["Fast Training of Small Nets", "1 Introduction"]
    # "2.1 Details" is level 3: the title and "1 Introduction" are above it and still open
    assert result.pages[2].carried == ["Fast Training of Small Nets", "1 Introduction"]
    assert result.pages[0].markdown.startswith("# Fast Training of Small Nets")


def test_convert_second_call_does_not_run_the_engine(pdf: Path, tmp_path: Path) -> None:
    engine = FakeEngine(sample_pages())
    first = convert_pdf(pdf, cache_dir=tmp_path / "c", engine=engine)
    second = convert_pdf(pdf, cache_dir=tmp_path / "c", engine=engine)
    assert engine.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.seconds == 0.0
    assert [p.markdown for p in first.pages] == [p.markdown for p in second.pages]


def test_convert_runs_again_for_new_bytes_options_or_docling_version(
    pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine(sample_pages())
    cache = tmp_path / "c"
    convert_pdf(pdf, cache_dir=cache, engine=engine)
    convert_pdf(pdf, cache_dir=cache, options=DoclingOptions(num_threads=1), engine=engine)
    assert engine.calls == 1  # threads change the speed, not the output
    convert_pdf(pdf, cache_dir=cache, options=DoclingOptions(ocr=True), engine=engine)
    assert engine.calls == 2
    monkeypatch.setattr(conv, "docling_version", lambda: "9.9.9")
    convert_pdf(pdf, cache_dir=cache, engine=engine)
    assert engine.calls == 3
    monkeypatch.undo()
    pdf.write_bytes(pdf.read_bytes() + b"changed")
    convert_pdf(pdf, cache_dir=cache, engine=engine)
    assert engine.calls == 4


def test_convert_is_deterministic(pdf: Path, tmp_path: Path) -> None:
    a = convert_pdf(pdf, cache_dir=tmp_path / "a", engine=FakeEngine(sample_pages()))
    b = convert_pdf(pdf, cache_dir=tmp_path / "b", engine=FakeEngine(sample_pages()))
    read = lambda r: json.loads((r.folder / "converted.json").read_text("utf-8"))  # noqa: E731
    first, second = read(a), read(b)
    first.pop("seconds")
    second.pop("seconds")
    assert first == second
    assert a.folder.name == b.folder.name


def test_convert_rejects_figures_that_do_not_match_the_placeholders(
    pdf: Path, tmp_path: Path
) -> None:
    pages = [RawPage(1, "text\n\n<!-- image -->", [])]
    with pytest.raises(ConversionError, match="page 1"):
        convert_pdf(pdf, cache_dir=tmp_path, engine=FakeEngine(pages))
    pages = [RawPage(1, "text", [RawFigure(tiny_png(), 1, 1)])]
    with pytest.raises(ConversionError, match="page 1"):
        convert_pdf(pdf, cache_dir=tmp_path, engine=FakeEngine(pages))
    assert not list(tmp_path.glob("*/converted.json"))  # a failed run leaves no cache entry


def test_convert_rejects_an_engine_without_pages(pdf: Path, tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="no pages"):
        convert_pdf(pdf, cache_dir=tmp_path, engine=FakeEngine([]))


def test_convert_title_falls_back_to_the_file_name(pdf: Path, tmp_path: Path) -> None:
    result = convert_pdf(
        pdf, cache_dir=tmp_path, engine=FakeEngine([RawPage(1, "plain text only", [])])
    )
    assert result.title == "paper"
    assert result.pages[0].carried == []
