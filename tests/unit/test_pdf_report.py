# ruff: noqa: RUF059 (the fixture returns a tuple; each test unpacks what it needs)
"""Spec P6-01 AC6: the estimate before a run and the quality report of a PDF knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import ServiceSettings, load_kb_config, repo_path
from rag.data.pdf.connector import PdfDirConnector
from rag.ingest import KbProcessor
from rag.ingest.describe import ModelTextCache, reset_shared_services
from rag.ingest.pdf_report import estimate_run, main, pdf_quality_report
from rag.testing.pdf import FakeEngine, sample_pages, tiny_png

KB_ROOT = repo_path("kb")


def words(text: str) -> list[str]:
    return text.split()


@pytest.fixture(autouse=True)
def _fresh_services():
    reset_shared_services()
    yield
    reset_shared_services()


@pytest.fixture
def env(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.pdf").write_bytes(b"%PDF fake a")
    (raw / "chart.png").write_bytes(tiny_png(300, 200, color=(9, 9, 9)))
    cache = tmp_path / "model_text.sqlite3"
    cfg = load_kb_config(repo_path("kb/private/pdf_demo.yaml"), kb_root=KB_ROOT)
    pipelines = {}
    for ext, pipe in cfg.pipelines.items():
        params = {**pipe.reader_params, "cache_path": str(cache), "budget_cny": 0.5}
        steps = [
            s.model_copy(update={"params": {**s.params, "cache_path": str(cache)}})
            if s.name == "table_summary"
            else s
            for s in pipe.steps
        ]
        pipelines[ext] = pipe.model_copy(update={"reader_params": params, "steps": steps})
    cfg = cfg.model_copy(update={"pipelines": pipelines})
    settings = ServiceSettings(
        _env_file=None,  # type: ignore[call-arg]
        chat_backend="fake",
        chat_fake_reply="A generated text for the test.",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )

    def make(engine: FakeEngine | None = None):
        connector = PdfDirConnector(
            kb=cfg.kb,
            path=raw,
            cache_dir=tmp_path / "pdfcache",
            engine=engine or FakeEngine(sample_pages()),
        )
        return connector, KbProcessor(cfg, settings=settings, tokenizer=words)

    return cfg, settings, cache, make


def test_pdf_report_counts_what_was_read_described_and_chunked(env) -> None:
    cfg, settings, cache, make = env
    connector, proc = make()
    report = pdf_quality_report(proc, connector)
    assert report.kb == "pdf_demo"
    assert report.conversion["files"] == 1 and report.conversion["pages"] == 3
    assert report.reader["pages"] == 3 and report.reader["tables"] == 1
    assert report.reader["images_seen"] == 4  # three figures in the PDF and the PNG file
    assert report.reader["images_skipped_small"] == 1
    model = report.reader["model_text"]
    assert model["images_described"] == 3 and model["image_cache_hits"] == 0
    assert model["tables_summarised"] == 1 and report.tables["summary_chunks"] == 1
    assert report.tables["tables"] == 1 and report.tables["table_chunks"] >= 1
    assert report.chunk_kinds["table_summary"] == 1 and report.chunk_kinds["table"] >= 1
    assert report.chunks_with_generated_text >= 2  # figure descriptions and the summary
    assert report.quality.docs == 4 and report.quality.over_cap == 0
    assert report.quality.doc_types == {"pdf": 3, "image": 1}
    # counts only: neither the page text nor a description is in the report
    text = report.model_dump_json()
    assert "Fast Training" not in text and "A generated text" not in text
    assert any("images described" in line for line in report.summary_lines())


def test_pdf_report_second_run_costs_no_model_call(env) -> None:
    cfg, settings, cache, make = env
    first = pdf_quality_report(*reversed(make()))  # type: ignore[arg-type]
    reset_shared_services()  # a new process: only the cache file is left
    connector, proc = make()
    second = pdf_quality_report(proc, connector)
    model = second.reader["model_text"]
    assert model["images_described"] == 0 and model["image_cache_hits"] == 3
    assert model["tables_summarised"] == 0 and model["table_cache_hits"] == 1
    assert model["spend_cny"] == 0.0
    assert second.chunk_kinds == first.chunk_kinds


def test_estimate_lists_what_a_run_would_describe_and_its_cost(env) -> None:
    cfg, settings, cache, make = env
    connector, _ = make()
    estimate = estimate_run(cfg, connector, settings, tokenizer=words)
    assert estimate.figures_total == 4 and estimate.figures_skipped_small == 1
    assert estimate.figures_to_describe == 3 and estimate.figures_cached == 0
    assert estimate.tables_total == 1 and estimate.tables_to_summarise == 1
    assert estimate.image_cost_cny > 0 and estimate.table_cost_cny > 0
    assert estimate.total_cny == pytest.approx(
        estimate.image_cost_cny + estimate.table_cost_cny + estimate.embed_cost_cny
    )
    assert estimate.fits_budget is True and estimate.budget_cny == 0.5
    # an estimate calls no model: no ledger line, no model service built
    assert not (Path(settings.cost_ledger_path)).exists()


def test_estimate_after_a_run_is_zero_for_everything_cached(env) -> None:
    cfg, settings, cache, make = env
    connector, proc = make()
    pdf_quality_report(proc, connector)
    reset_shared_services()
    estimate = estimate_run(cfg, make()[0], settings, tokenizer=words)
    assert estimate.figures_to_describe == 0 and estimate.figures_cached == 3
    assert estimate.tables_to_summarise == 0
    assert estimate.image_cost_cny == 0.0 and estimate.table_cost_cny == 0.0


def test_estimate_says_when_the_run_does_not_fit_the_budget(env) -> None:
    cfg, settings, cache, make = env
    pipelines = {
        ext: pipe.model_copy(update={"reader_params": {**pipe.reader_params, "budget_cny": 0.0001}})
        for ext, pipe in cfg.pipelines.items()
    }
    tight = cfg.model_copy(update={"pipelines": pipelines})
    estimate = estimate_run(tight, make()[0], settings, tokenizer=words)
    assert estimate.fits_budget is False


def test_estimate_refuses_a_model_without_a_price(env) -> None:
    from rag.cost import UnknownModelPriceError

    cfg, settings, cache, make = env
    pipelines = {
        ext: pipe.model_copy(
            update={"reader_params": {**pipe.reader_params, "vision_model": "no-such-model"}}
        )
        for ext, pipe in cfg.pipelines.items()
    }
    with pytest.raises(UnknownModelPriceError):
        estimate_run(
            cfg.model_copy(update={"pipelines": pipelines}), make()[0], settings, tokenizer=words
        )


def test_cache_file_is_the_one_the_config_names(env) -> None:
    cfg, settings, cache, make = env
    connector, proc = make()
    pdf_quality_report(proc, connector)
    assert ModelTextCache(cache).get("image", "x", "y", "z") is None
    assert cache.is_file()


def test_main_reports_usage_errors_in_one_line(
    env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, settings, cache, make = env
    kb_dir = tmp_path / "kb" / "private"
    kb_dir.mkdir(parents=True)
    data = json.loads(cfg.model_dump_json())
    data["source"] = {"connector": "pdf_dir", "path": str(tmp_path / "no-such-folder")}
    (kb_dir / "pdf_demo.yaml").write_text(json.dumps(data), "utf-8")  # JSON is YAML
    settings = settings.model_copy(update={"kb_dir": str(tmp_path / "kb")})
    assert main(["--kb", "nope", "--estimate"], settings=settings) == 2
    assert "unknown knowledge base" in capsys.readouterr().err
    assert main(["--kb", "pdf_demo", "--estimate"], settings=settings) == 1
    assert "does not exist" in capsys.readouterr().err
    assert main(["--kb", "pdf_demo", "--limit", "0"], settings=settings) == 2


def test_a_corrupt_file_is_one_counted_failure_and_the_others_are_reported(env) -> None:
    cfg, settings, cache, make = env
    connector, proc = make(FakeEngine(sample_pages(), fail_on=["b.pdf"]))
    folder = connector.root
    (folder / "b.pdf").write_bytes(b"corrupt")
    (folder / "c.pdf").write_bytes(b"%PDF c")
    report = pdf_quality_report(proc, connector)
    assert list(report.quality.read_errors) == ["pdf_demo:b.pdf"]
    assert "Docling cannot read b.pdf" in report.quality.read_errors["pdf_demo:b.pdf"]
    assert report.conversion["failed_files"] == 1
    assert report.conversion["files"] == 2 and report.conversion["pages"] == 6  # a.pdf, c.pdf
    assert any("failed files 1" in line for line in report.summary_lines())
    estimate = estimate_run(
        cfg, make(FakeEngine(sample_pages(), fail_on=["b.pdf"]))[0], settings, tokenizer=words
    )
    assert estimate.pages == 6 and estimate.files_failed == 1
