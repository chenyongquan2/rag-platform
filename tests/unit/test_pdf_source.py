"""Spec P6-01: the `pdf_dir` connector and the `pdf` and `image` readers (AC4, AC5, AC6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from llama_index.core.schema import MetadataMode

from rag.data import SourceDocument
from rag.data.pdf.connector import PdfDirConnector
from rag.ingest.describe import (
    IMAGE_MARKER_PREFIX,
    DescribeBudgetError,
    ModelTextCache,
    ModelTextService,
    SpendGuard,
)
from rag.ingest.pdf import ImageReader, PdfReader
from rag.registry import connectors, readers
from rag.testing.pdf import FakeEngine, FakeVision, sample_pages, tiny_png

KB = "pdf_demo"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"%PDF-1.4 fake A")
    (root / "b.pdf").write_bytes(b"%PDF-1.4 fake B")
    (root / "chart.png").write_bytes(tiny_png(300, 200))
    (root / "notes.txt").write_text("not a source", encoding="utf-8")
    return root


def connector(folder: Path, tmp_path: Path, engine: FakeEngine | None = None, **kwargs):
    return PdfDirConnector(
        kb=KB,
        path=folder,
        cache_dir=tmp_path / "cache",
        engine=engine or FakeEngine(sample_pages()),
        **kwargs,
    )


def reader(
    tmp_path: Path, vision: FakeVision | None = None, *, budget: float | None = None, **kwargs
):
    vision = vision if vision is not None else FakeVision()
    service = ModelTextService(
        cache=ModelTextCache(tmp_path / "text.sqlite3"),
        describer=vision,
        guard=SpendGuard(budget),
        image_estimate_cny=0.003,
    )
    params = {
        "kb": KB,
        "pipeline_version": "v-test",
        "language": "en",
        "embed_keys": ["title_path"],
        "llm_keys": ["title_path", "source_rel", "page"],
        "service": service,
        **kwargs,
    }
    return PdfReader(**params), vision, service


def pages_of(folder: Path, tmp_path: Path, name: str = "a.pdf") -> list[SourceDocument]:
    return [d for d in connector(folder, tmp_path).documents() if d.source_rel == name]


# ---- connector (AC4) -------------------------------------------------------------------------


def test_pdf_connector_is_registered_and_gives_one_document_per_page(
    folder: Path, tmp_path: Path
) -> None:
    assert "pdf_dir" in connectors
    docs = list(connector(folder, tmp_path).documents())
    ids = [d.doc_id for d in docs]
    assert ids == [
        "pdf_demo:a.pdf#p1",
        "pdf_demo:a.pdf#p2",
        "pdf_demo:a.pdf#p3",
        "pdf_demo:b.pdf#p1",
        "pdf_demo:b.pdf#p2",
        "pdf_demo:b.pdf#p3",
        "pdf_demo:chart.png",
    ]
    first = docs[0]
    assert (first.kb, first.source_rel, first.ext) == (KB, "a.pdf", ".pdf")
    assert first.content.decode("utf-8").startswith("# Fast Training of Small Nets")
    assert first.extra["page"] == 1 and first.extra["pages"] == 3
    assert first.extra["title"] == "Fast Training of Small Nets"
    assert docs[1].extra["carried"] == ["Fast Training of Small Nets", "1 Introduction"]
    figures = docs[2].extra["figures"]
    assert [f["index"] for f in figures] == [1, 2]
    assert Path(figures[0]["path"]).is_file()
    assert {d.source_version for d in docs[:3]} == {docs[0].source_version}
    assert docs[0].source_version.startswith("sha256:")
    assert docs[0].source_version != docs[3].source_version
    image = docs[-1]
    assert (image.ext, image.source_rel) == (".png", "chart.png")
    assert image.content == tiny_png(300, 200)


def test_pdf_connector_second_listing_uses_the_cache_and_gives_the_same_ids(
    folder: Path, tmp_path: Path
) -> None:
    engine = FakeEngine(sample_pages())
    conn = connector(folder, tmp_path, engine)
    first = [(d.doc_id, d.source_version) for d in conn.documents()]
    second = [(d.doc_id, d.source_version) for d in conn.documents()]
    assert first == second
    assert engine.calls == 2  # a.pdf and b.pdf once each
    assert conn.conversions["a.pdf"]["pages"] == 3
    assert conn.conversions["a.pdf"]["cached"] is True


def test_pdf_connector_limit_and_files(folder: Path, tmp_path: Path) -> None:
    only_first = {d.source_rel for d in connector(folder, tmp_path, limit=1).documents()}
    assert only_first == {"a.pdf"}
    chosen = {d.source_rel for d in connector(folder, tmp_path, files=["chart.png"]).documents()}
    assert chosen == {"chart.png"}
    with pytest.raises(FileNotFoundError, match=r"missing\.pdf"):
        list(connector(folder, tmp_path, files=["missing.pdf"]).documents())


def test_pdf_connector_needs_an_existing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfDirConnector(kb=KB, path=tmp_path / "nope", cache_dir=tmp_path / "c")


# ---- pdf reader (AC5) ------------------------------------------------------------------------


def test_pdf_reader_is_registered_and_fills_the_s5_4_contract(folder: Path, tmp_path: Path) -> None:
    assert "pdf" in readers and "image" in readers
    rd, _, _ = reader(tmp_path)
    source = pages_of(folder, tmp_path)[0]
    (doc,) = rd.load_data(source)
    assert doc.id_ == "pdf_demo:a.pdf#p1"
    meta = doc.metadata
    assert meta["kb"] == KB and meta["doc_id"] == doc.id_
    assert meta["source_rel"] == "a.pdf" and meta["unit"] == "p1" and meta["page"] == 1
    assert meta["doc_type"] == "pdf" and meta["language"] == "en"
    assert meta["acl_groups"] == ["public"] and meta["pipeline_version"] == "v-test"
    assert meta["images"] and meta["images"][0].endswith("p1_f1.png")
    assert not any(k in meta for k in ("mtime", "modified", "size", "source_version"))
    # a page that starts with the title heading has no carried path of its own
    assert meta["title_path"] == ""
    (again,) = rd.load_data(source)
    assert again.hash == doc.hash and again.id_ == doc.id_


def test_pdf_reader_embed_text_is_title_path_and_body_only(folder: Path, tmp_path: Path) -> None:
    rd, _, _ = reader(tmp_path)
    (doc,) = rd.load_data(pages_of(folder, tmp_path)[1])
    assert doc.metadata["title_path"] == "Fast Training of Small Nets > 1 Introduction"
    embedded = doc.get_content(MetadataMode.EMBED)
    assert embedded.startswith("title_path: Fast Training of Small Nets > 1 Introduction")
    assert "page" not in embedded.split("\n\n")[0]
    llm = doc.get_content(MetadataMode.LLM)
    assert "page: 2" in llm and "source_rel: a.pdf" in llm and "doc_id" not in llm


def test_pdf_reader_writes_the_description_at_the_figure_position(
    folder: Path, tmp_path: Path
) -> None:
    rd, vision, _ = reader(tmp_path, FakeVision("A line chart of training speed."))
    (doc,) = rd.load_data(pages_of(folder, tmp_path)[0])
    text = doc.text
    marker = (
        f"{IMAGE_MARKER_PREFIX}, not part of the original document): "
        "A line chart of training speed.]"
    )
    assert marker in text
    assert "<!-- figure" not in text
    assert text.index("Figure 1: Training speed.") < text.index(marker) < text.index("Table 1:")
    assert vision.seen == [("Figure 1: Training speed.", "en")]


def test_pdf_reader_skips_small_images_and_counts_them(folder: Path, tmp_path: Path) -> None:
    rd, vision, _ = reader(tmp_path, min_image_px=64)
    (doc,) = rd.load_data(pages_of(folder, tmp_path)[2])
    assert doc.text.count(IMAGE_MARKER_PREFIX) == 1  # the 32 x 32 figure is not described
    assert "<!-- figure" not in doc.text
    assert vision.calls == 1
    assert rd.stats.images_seen == 2 and rd.stats.images_skipped_small == 1


def test_pdf_reader_without_descriptions_drops_the_markers(folder: Path, tmp_path: Path) -> None:
    rd, vision, _ = reader(tmp_path, describe_images="off")
    (doc,) = rd.load_data(pages_of(folder, tmp_path)[0])
    assert IMAGE_MARKER_PREFIX not in doc.text and "<!-- figure" not in doc.text
    assert vision.calls == 0
    assert "Figure 1: Training speed." in doc.text


def test_pdf_reader_second_read_makes_no_model_call(folder: Path, tmp_path: Path) -> None:
    rd, vision, service = reader(tmp_path)
    sources = pages_of(folder, tmp_path)
    first = [rd.load_data(s)[0].text for s in sources]
    second = [rd.load_data(s)[0].text for s in sources]
    assert first == second
    assert vision.calls == 2  # two different figures on the three pages (the small one skipped)
    assert service.stats.image_cache_hits == 2


def test_pdf_reader_a_failed_description_leaves_the_page_without_it(
    folder: Path, tmp_path: Path
) -> None:
    class Broken(FakeVision):
        def describe(self, png: bytes, *, caption: str, language: str):
            raise TimeoutError("slow")

    rd, _, _ = reader(tmp_path, Broken())
    (doc,) = rd.load_data(pages_of(folder, tmp_path)[0])
    assert IMAGE_MARKER_PREFIX not in doc.text and "<!-- figure" not in doc.text
    assert rd.stats.images_failed == 1


def test_pdf_reader_budget_stop_is_not_swallowed(folder: Path, tmp_path: Path) -> None:
    rd, vision, _ = reader(tmp_path, budget=0.001)
    with pytest.raises(DescribeBudgetError):
        rd.load_data(pages_of(folder, tmp_path)[0])
    assert vision.calls == 0


def test_pdf_reader_page_without_headings_uses_the_document_title(
    tmp_path: Path,
) -> None:
    from rag.data.pdf.convert import RawPage

    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "plain.pdf").write_bytes(b"%PDF plain")
    pages = [RawPage(1, "Only text.", []), RawPage(2, "More text.", [])]
    conn = PdfDirConnector(kb=KB, path=folder, cache_dir=tmp_path / "c", engine=FakeEngine(pages))
    rd, _, _ = reader(tmp_path)
    docs = [rd.load_data(s)[0] for s in conn.documents()]
    assert [d.metadata["title_path"] for d in docs] == ["plain", "plain"]


def test_pdf_reader_stats_have_counts_and_no_page_text(folder: Path, tmp_path: Path) -> None:
    rd, _, _ = reader(tmp_path)
    for source in pages_of(folder, tmp_path):
        rd.load_data(source)
    stats = rd.stats.to_dict()
    assert stats["pages"] == 3 and stats["tables"] == 1
    assert stats["images_seen"] == 3 and stats["images_skipped_small"] == 1
    assert stats["model_text"]["images_described"] == 2
    assert "Fast Training" not in str(stats)


# ---- image reader (AC5) ------------------------------------------------------------------------


def image_reader(tmp_path: Path, **kwargs):
    _, vision, service = reader(tmp_path)
    img = ImageReader(
        kb=KB,
        pipeline_version="v-test",
        language="en",
        embed_keys=["title_path"],
        llm_keys=["title_path", "source_rel"],
        service=service,
        **kwargs,
    )
    return img, vision


def image_source(folder: Path, tmp_path: Path) -> SourceDocument:
    return next(d for d in connector(folder, tmp_path).documents() if d.ext == ".png")


def test_image_reader_makes_a_document_of_the_description(folder: Path, tmp_path: Path) -> None:
    img, vision = image_reader(tmp_path)
    (doc,) = img.load_data(image_source(folder, tmp_path))
    assert doc.id_ == "pdf_demo:chart.png"
    assert doc.text.startswith(IMAGE_MARKER_PREFIX)
    assert doc.metadata["doc_type"] == "image" and doc.metadata["title_path"] == "chart"
    assert doc.metadata["images"] == ["chart.png"] and doc.metadata["unit"] is None
    assert vision.seen == [("", "en")]
    img.load_data(image_source(folder, tmp_path))
    assert vision.calls == 1  # cache


def test_image_reader_off_gives_an_empty_text(folder: Path, tmp_path: Path) -> None:
    img, vision = image_reader(tmp_path, describe_images="off")
    (doc,) = img.load_data(image_source(folder, tmp_path))
    assert doc.text == "" and vision.calls == 0


def test_image_reader_refuses_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    img, _ = image_reader(tmp_path)
    source = SourceDocument(
        doc_id="pdf_demo:bad.png",
        kb=KB,
        source_rel="bad.png",
        source_url=None,
        ext=".png",
        content=b"not a png",
        source_version="sha256:x",
    )
    with pytest.raises(ValueError, match=r"bad\.png"):
        img.load_data(source)


# ---- a file that cannot be read (review round 1) -----------------------------------------------


def test_pdf_connector_a_corrupt_file_does_not_stop_the_files_after_it(
    folder: Path, tmp_path: Path
) -> None:
    (folder / "a.pdf").write_bytes(b"%PDF a")
    (folder / "b.pdf").write_bytes(b"corrupt")
    (folder / "c.pdf").write_bytes(b"%PDF c")
    conn = connector(folder, tmp_path, FakeEngine(sample_pages(), fail_on=["b.pdf"]))
    docs = list(conn.documents())
    ids = [d.doc_id for d in docs]
    assert ids[:3] == ["pdf_demo:a.pdf#p1", "pdf_demo:a.pdf#p2", "pdf_demo:a.pdf#p3"]
    assert "pdf_demo:b.pdf" in ids  # one document for the file that failed
    assert ids[-4:-1] == ["pdf_demo:c.pdf#p1", "pdf_demo:c.pdf#p2", "pdf_demo:c.pdf#p3"]
    failed = next(d for d in docs if d.doc_id == "pdf_demo:b.pdf")
    assert "Docling cannot read b.pdf" in failed.extra["conversion_error"]
    assert failed.source_version.startswith("sha256:")
    assert list(conn.failures) == ["b.pdf"]


def test_pdf_connector_does_not_convert_a_failed_file_twice(folder: Path, tmp_path: Path) -> None:
    (folder / "b.pdf").write_bytes(b"corrupt")
    engine = FakeEngine(sample_pages(), fail_on=["b.pdf"])
    conn = connector(folder, tmp_path, engine)
    list(conn.documents())
    calls = engine.calls
    list(conn.documents())
    assert engine.calls == calls  # a.pdf comes from the cache, b.pdf is not tried again


def test_pdf_reader_raises_for_the_document_of_a_failed_file(folder: Path, tmp_path: Path) -> None:
    (folder / "b.pdf").write_bytes(b"corrupt")
    conn = connector(folder, tmp_path, FakeEngine(sample_pages(), fail_on=["b.pdf"]))
    failed = next(d for d in conn.documents() if d.doc_id == "pdf_demo:b.pdf")
    rd, _, _ = reader(tmp_path)
    with pytest.raises(ValueError, match=r"b\.pdf.*Docling cannot read"):
        rd.load_data(failed)


def test_pdf_connector_limits_are_docling_config_keys(folder: Path, tmp_path: Path) -> None:
    conn = connector(
        folder, tmp_path, docling={"max_file_mb": 0.00001, "max_pages": 2, "timeout_s": 5}
    )
    assert conn.options.max_file_mb == 0.00001
    assert (conn.options.max_pages, conn.options.timeout_s) == (2, 5)
    docs = list(conn.documents())
    # a.pdf and b.pdf are 15 bytes: over the size limit, so each is one failed document
    assert [d.doc_id for d in docs if d.ext == ".pdf"] == ["pdf_demo:a.pdf", "pdf_demo:b.pdf"]
    assert "larger than max_file_mb" in docs[0].extra["conversion_error"]


def test_pdf_connector_files_must_stay_inside_the_folder(folder: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.pdf").write_bytes(b"%PDF outside")
    with pytest.raises(ValueError, match="outside"):
        list(connector(folder, tmp_path, files=["../outside.pdf"]).documents())
