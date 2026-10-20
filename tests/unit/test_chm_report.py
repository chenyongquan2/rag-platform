"""Data-quality report of a CHM knowledge base (spec P4-01 AC11; design s6-3, s6-4).

The report holds counts and page ids, never page text: the manual is a private vendor document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llama_index.core.schema import MetadataMode

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.data.chm.connector import ChmDirConnector
from rag.ingest import KbProcessor
from rag.ingest import build as ingest_build
from rag.ingest.chm_report import ChmQualityReport, chm_quality_report, main
from rag.registry import connectors
from rag.testing import chm as t
from rag.testing.chm import (
    DEMO_WORDS,
    demo_en_volume,
    demo_kb_dict,
    demo_page,
    demo_zh_volume,
    hhc,
    write_volume,
)

SECRET_WORDS = DEMO_WORDS


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


def make_cfg(en: Path, zh: Path) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig.model_validate(demo_kb_dict({"en": en, "zh": zh}))


def page(title: str, *body: str, charset: str = "iso-8859-1") -> str:
    return demo_page(title, *body, charset=charset)


@pytest.fixture
def report(tmp_path: Path) -> ChmQualityReport:
    cfg = make_cfg(demo_en_volume(tmp_path / "en"), demo_zh_volume(tmp_path / "zh"))
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    assert isinstance(connector, ChmDirConnector)
    return chm_quality_report(proc, connector)


def test_quality_report_counts_pages_in_and_out_per_language(report: ChmQualityReport) -> None:
    en, zh = report.volumes["en"], report.volumes["zh"]
    assert (en.pages_in, en.pages_out) == (6, 4)  # the empty page and the copy of B are skipped
    assert en.skipped == {"empty": 1, "duplicate": 1}
    assert (zh.pages_in, zh.pages_out, zh.skipped) == (3, 3, {})


def test_quality_report_attributes_skipped_pages_to_their_own_language(tmp_path: Path) -> None:
    ok = demo_page("Fine", t.text("A sentence with enough words to be kept as content."))
    en = write_volume(
        tmp_path / "en",
        {"fine.htm": ok.encode("cp1252")},
        toc=("a.hhc", hhc([("Fine", "fine.htm", [])])),
    )
    zh = write_volume(
        tmp_path / "zh",
        {
            "fine.htm": ok.encode("cp1252"),
            "empty.htm": t.page_html(t.empty_paragraph()).encode("cp1252"),
        },
        toc=("a.hhc", hhc([("好", "fine.htm", []), ("空", "empty.htm", [])])),
        toc_encoding="gb18030",
    )
    cfg = make_cfg(en, zh)
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    rep = chm_quality_report(proc, connector)
    assert (rep.volumes["en"].pages_in, rep.volumes["en"].pages_out) == (1, 1)
    assert rep.volumes["en"].skipped == {}
    assert (rep.volumes["zh"].pages_in, rep.volumes["zh"].pages_out) == (2, 1)
    assert rep.volumes["zh"].skipped == {"empty": 1}


def test_quality_report_has_the_fidelity_numbers_and_names_lossy_pages(
    report: ChmQualityReport, tmp_path: Path
) -> None:
    assert report.reader["lost_chars"] == 0 and report.reader["lossy_pages"] == []
    assert report.reader["visible_chars"] > 100
    assert "lost 0 of" in "\n".join(report.summary_lines())
    lossy = page_html_with_hidden_text()
    en = write_volume(
        tmp_path / "en2",
        {"lossy.htm": lossy.encode("cp1252")},
        toc=("a.hhc", hhc([("Lossy", "lossy.htm", [])])),
    )
    cfg = make_cfg(en, demo_zh_volume(tmp_path / "zh2"))
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    rep = chm_quality_report(proc, connector)
    assert rep.reader["lossy_pages"] == ["mt5:en/lossy.htm"]
    assert rep.reader["lost_chars"] == len("hiddenwordsthatvanish")
    assert "lossy pages 1" in "\n".join(rep.summary_lines())


def page_html_with_hidden_text() -> str:
    # a <head> element inside the help text: the converter cannot carry it
    return t.page_html(
        t.h1("Lossy"),
        t.text("Visible text that is long enough."),
        "<head>hidden words that vanish</head>",
    )


def test_quality_report_pairs_pages_across_languages(report: ChmQualityReport) -> None:
    assert report.pairing == {
        "both": 2,  # a, b
        "only_en": 4,  # b_copy, empty, only_en, orphan
        "only_zh": 1,  # lie
    }


def test_quality_report_reports_toc_mismatches_per_language(report: ChmQualityReport) -> None:
    en = report.volumes["en"]
    assert en.not_in_toc == 1  # orphan.htm
    assert en.toc_without_page == 1  # gone.htm
    assert report.volumes["zh"].not_in_toc == 0


def test_quality_report_records_encodings_fallbacks_and_decode_errors(
    report: ChmQualityReport,
) -> None:
    assert report.volumes["zh"].encodings == {"gb18030": 3}
    assert report.volumes["en"].encodings == {"ascii": 6}
    assert report.reader["fallbacks"] == 1  # lie.htm says iso-8859-1 and is GBK
    assert report.reader["decode_errors"] == 0
    assert report.reader["declared_charsets"] == {"gb2312": 2, "iso-8859-1": 7}
    assert report.reader["not_in_toc"] == 1 and report.reader["no_title"] == 0


def test_quality_report_counts_decode_errors(tmp_path: Path) -> None:
    broken = (
        b"<html><body><div id='help'><p>ok \x81\x20 \x81\x81\x81 and more text here</p>"
        b"</div></body></html>"
    )
    en = write_volume(tmp_path / "en", {"x.htm": broken}, toc=("a.hhc", hhc([("X", "x.htm", [])])))
    zh = demo_zh_volume(tmp_path / "zh")
    cfg = make_cfg(en, zh)
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    rep = chm_quality_report(proc, connector)
    assert rep.reader["decode_errors"] == 1
    assert rep.reader["replacement_chars"] >= 3
    assert rep.quality.garbled_units == 1


def test_quality_report_has_read_noise_and_clean_rule_numbers(report: ChmQualityReport) -> None:
    read = {r.name: r for r in report.read_rules}
    assert read["nav_bar"].pages_hit == 9  # every page has a navigation bar
    assert read["nav_bar"].chars_removed > 0
    assert read["empty_paragraph"].pages_hit == 1
    clean = {r.name: r for r in report.quality.rules}
    assert clean["chm_title_echo"].units_hit == 2  # "Only English" and the orphan's own title
    assert clean["chm_title_echo"].units_seen == 9  # every page went through it
    assert clean["skip_empty"].units_hit == 1
    assert clean["dedupe"].units_hit == 1
    assert report.quality.units == 9 and report.quality.units_kept == 7


def test_quality_report_counts_chunks_and_tokens(report: ChmQualityReport) -> None:
    q = report.quality
    assert q.chunks >= 7
    assert q.over_cap == 0 and q.unclosed_fences == 0
    assert q.tokens.max <= 40


def test_quality_report_contains_no_page_text(report: ChmQualityReport) -> None:
    dumped = report.model_dump_json()
    for word in SECRET_WORDS:
        assert word not in dumped
    assert report.quality.samples == []  # the generic report's before/after diffs are off
    # page ids and counts are there
    assert "mt5:en/b_copy.htm" in dumped


def test_quality_report_marks_a_rule_that_removes_much(tmp_path: Path) -> None:
    pages = {"a.htm": page("A", t.text("tiny"))}
    tree = [("A", "a.htm", [])]
    en = write_volume(
        tmp_path / "en", {k: v.encode("cp1252") for k, v in pages.items()}, toc=("a.hhc", hhc(tree))
    )
    cfg = make_cfg(en, demo_zh_volume(tmp_path / "zh"))
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    rep = chm_quality_report(proc, connector, warn_ratio=0.0001)
    assert any(r.warn for r in rep.quality.rules)


def test_quality_report_summary_lines_name_the_numbers(report: ChmQualityReport) -> None:
    summary = "\n".join(report.summary_lines())
    assert "en" in summary and "zh" in summary
    assert "pages in 6" in summary and "out 4" in summary
    assert "paired 2" in summary


def test_quality_report_cli_writes_json_and_prints_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    en, zh = demo_en_volume(tmp_path / "en"), demo_zh_volume(tmp_path / "zh")
    kb_dir = tmp_path / "kb" / "private"
    kb_dir.mkdir(parents=True)
    (kb_dir / "mt5.yaml").write_text(
        "kb: mt5\n"
        "source:\n  connector: chm_dir\n  volumes:\n"
        f"    - {{language: zh, path: '{zh.as_posix()}'}}\n"
        f"    - {{language: en, path: '{en.as_posix()}'}}\n"
        "language: auto\n"
        "pipelines:\n"
        "  '.htm':\n    reader: chm\n"
        "    clean: [whitespace, chm_title_echo, skip_empty]\n"
        "    steps: [markdown_split, {token_split: {max_tokens: 40}}, merge_tiny,"
        " assign_chunk_ids]\n"
        "access: {callers: [local]}\n",
        "utf-8",
    )
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda model: words)
    settings = ServiceSettings(kb_dir=str(tmp_path / "kb"))
    out = tmp_path / "out" / "report.json"
    assert main(["--kb", "mt5", "--out", str(out)], settings=settings) == 0
    printed = capsys.readouterr().out
    assert "pages in 6" in printed and str(out) in printed
    data = json.loads(out.read_text("utf-8"))
    assert data["kb"] == "mt5" and data["volumes"]["en"]["pages_in"] == 6
    for word in SECRET_WORDS:
        assert word not in out.read_text("utf-8")
    assert main(["--kb", "mt5", "--limit", "1", "--out", str(out)], settings=settings) == 0
    assert json.loads(out.read_text("utf-8"))["volumes"]["en"]["pages_in"] == 1
    assert main(["--kb", "nope"], settings=settings) == 2


def test_quality_report_cli_reports_a_tripped_guard_without_page_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = {
        "big.htm": page("Big", t.text("Frobnicate the gizmo, and then do it once more later.")),
        "tiny.htm": page("Tiny", t.text("ok")),  # too short: skip_empty drops it
    }
    en = write_volume(tmp_path / "en", {k: v.encode("cp1252") for k, v in pages.items()})
    kb_dir = tmp_path / "kb" / "private"
    kb_dir.mkdir(parents=True)
    (kb_dir / "mt5.yaml").write_text(
        "kb: mt5\n"
        "source:\n  connector: chm_dir\n  volumes:\n"
        f"    - {{language: en, path: '{en.as_posix()}'}}\n"
        "pipelines:\n"
        "  '.htm':\n    reader: chm\n"
        "    clean:\n      - skip_empty: {max_corpus_share: 0.0001}\n"
        "    steps: [markdown_split, {token_split: {max_tokens: 40}}, assign_chunk_ids]\n"
        "access: {callers: [local]}\n",
        "utf-8",
    )
    monkeypatch.setattr(ingest_build, "load_tokenizer", lambda model: words)
    settings = ServiceSettings(kb_dir=str(tmp_path / "kb"))
    assert main(["--kb", "mt5", "--out", str(tmp_path / "r.json")], settings=settings) == 1
    err = capsys.readouterr().err
    assert "skip_empty" in err and "corpus" in err
    assert not (tmp_path / "r.json").exists()
    for word in SECRET_WORDS:
        assert word not in err


def test_documents_keep_title_path_language_and_page_key_through_cleaning(tmp_path: Path) -> None:
    cfg = make_cfg(demo_en_volume(tmp_path / "en"), demo_zh_volume(tmp_path / "zh"))
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    connector = connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), **cfg.source.model_extra
    )
    kept = proc.clean(proc.read(connector.documents()).documents)
    by_id = {d.node_id: d for d in kept}
    en_a, zh_a = by_id["mt5:en/a.htm"], by_id["mt5:zh/a.htm"]
    assert en_a.metadata["page_key"] == zh_a.metadata["page_key"] == "a"
    assert (en_a.metadata["language"], zh_a.metadata["language"]) == ("en", "zh")
    assert zh_a.metadata["title_path"] == "甲"
    assert "闪电" in zh_a.get_content(MetadataMode.NONE)
    assert "mt5:en/b_copy.htm" not in by_id and "mt5:en/b.htm" in by_id
