# P4-01 MT5 manual ingestion (CHM, zh + en, private knowledge base)

Status: review
Phase: P4
Depends on: P0-04, P0-06, P2-01
Design sections: s5-1, s5-2, s5-4, s6, s6-1..s6-4, s7-3, s7-4, s7-5, s8-4, s16-4, s1-3
Calls real models: yes (embeddings; estimate 2.0 CNY for both languages; limit 1.5 CNY, raised to 2.1 CNY after review; real spend 2.0686 CNY, see Handoff)

## Goal
The MetaTrader 5 API manual (two CHM files, Chinese and English) becomes the second, private knowledge base `mt5`:
CHM files are unpacked to a git-ignored folder, a `chm` reader turns each page into one Markdown Document with a title path
from the table of contents and a `page_key` shared by the zh and en version of the page, CHM-specific cleaning rules run with a
removal guard and a data-quality report, both languages are indexed into `kb_mt5_v1` (alias `kb_mt5`), and search can keep
one language version per `page_key` (`retrieval.dedupe_by_page_key`). Nothing extracted from the manual is committed.

## Interface
- `rag.data.chm`: `ChmSource` (protocol: `names()`, `read(name)`, `toc()`, `index()`, `info()`), `DirectoryChmSource(root)`,
  `decode_html(raw, *, hint=None) -> DecodedPage(text, encoding, declared, fallback, replaced)`,
  `parse_toc(text) -> list[TocEntry(name, local, path, depth)]`, `parse_index(text)`, `page_key_of(local)`,
  `unpack_chm(chm, out_dir, *, unpacker=None) -> UnpackResult`, `find_unpacker()` (7-Zip, `extract_chmLib`, `hh.exe`),
  connector `chm_dir` (`volumes: [{language, path, sections?}]`, `limit` per volume), CLI
  `python -m rag.data.chm unpack --chm F --out DIR`.
- `rag.data.SourceDocument.extra: dict` (hints from the connector: `language`, `page_key`, `toc_path`, `charset_hint`, `keywords`).
- `rag.ingest.chm`: reader `chm` (`ChmReader`, stats in `ChmReader.stats`), `html_to_markdown`; page metadata adds `page_key`, `links_to`.
- Cleaning rules (`rag.registry.steps`): `chm_title_echo`; every cleaning rule takes `max_page_share`,
  `max_corpus_share`, `guard_min_chars` (default off) and raises `RemovalGuardError`; the limits are not part of `pipeline_version`;
  `dedupe` takes `group_by: [metadata keys]`.
- `rag.ingest.chm_report`: `chm_quality_report(proc, connector)`; CLI `python -m rag.ingest.chm_report --kb mt5`
  (counters only, no page text, written to `data/reports/mt5/`). `IngestReport` gains `read_stats`, `clean_stats`;
  `KbProcessor.reader_stats()`.
- `rag.index.mapping`: keyword fields `page_key`, `links_to`; the retriever reads `page_key` from the index (`SOURCE_FIELDS`).
- `RetrievalConfig.dedupe_by_page_key: bool = False`; `rag.retrieve.page_groups.PageKeyDedupe`; wired in `make_search_pipeline`
  and `make_hybrid_search`. `search_config_version` of a KB with the flag off is unchanged.
- `kb/private/mt5.yaml` (`access.callers: [local]`, analyzer `es.analyzer` as a plain config key); tracked in git, `.gitignore`
  keeps other `kb/private/` files out.

## Acceptance criteria
Unit tests in `tests/unit/test_chm_*.py`, `test_retrieve_page_groups.py`, `test_kb_mt5.py`; integration in `tests/integration/test_chm_ingest_es.py`.
- [x] AC1 Encoding: ASCII, GBK page whose charset lies (declared iso-8859-1 or utf-8), cp1252 page declared iso-8859-1, UTF-8 page declared gb2312, garbage with replacement counted — `test_decode_*`
- [x] AC2 `.hhc` hierarchy gives title paths (nested lists, unclosed `<LI>`, Chinese names, anchors in `Local`); `.hhk` keywords — `test_parse_toc_*`, `test_parse_index_*`
- [x] AC3 `page_key_of` normalizes case, slashes, percent-encoding, anchors, extensions, Chinese names — `test_page_key_*`
- [x] AC4 `DirectoryChmSource` skips CHM internals, finds `.hhc`/`.hhk`, reads `#SYSTEM`; the unpack step runs the found tool and writes a manifest; no tool gives a clear error — `test_directory_source_*`, `test_unpack_*`, `test_find_unpacker_*`
- [x] AC5 Connector gives `doc_id` `mt5:<lang>/<file>`, language, page_key, toc_path, source_version, `limit`, `sections`; zh/en pages with one name share a `page_key`; a page missing from the toc is reported — `test_connector_*`
- [x] AC6 Reader: Document contract of s5-4 (fields, stable id and hash, no mtime, embed text = title path + body); headings, code blocks (`&nbsp;` indentation kept as spaces), tables, lists, images as marker plus `images`, links as text plus `links_to`, `&lt;` text stays — `test_reader_*`
- [x] AC7 HTML-stage noise: navigation bar, empty paragraphs, anchors, scripts are dropped and counted; look-alike body text stays — `test_html_noise_*`
- [x] AC8 Rules `chm_title_echo` and `dedupe(group_by)`: positive and negative tests — `test_title_echo_*`, `test_dedupe_group_by_*`
- [x] AC9 Whole-chain negative: prose with `<b>`, `a < 0`, `Vec<T>` survives reader plus clean chain — `test_chain_keeps_angle_brackets_in_prose`
- [x] AC10 Removal guard: page share and corpus share trip with a clear error without page text; below the limit passes; off by default; limits do not change `pipeline_version` — `test_guard_*`, `test_pipeline_version_ignores_the_guard_limits_but_not_other_parameters`
- [x] AC11 Quality report: counts per language, encodings, fallbacks, decode errors, no-title, duplicate page_keys, per-rule chars removed, unpaired pages; contains none of the page text — `test_quality_report_*`, `tests/unit/test_chm_ingest_job.py`
- [x] AC12 `dedupe_by_page_key`: both versions present keeps the question's language, other cases keep the best-ranked, several chunks of one version stay; off by default; config version unchanged when off; `page_key` is read from the index — `test_page_key_dedupe_*`, `test_search_reads_page_key_*`
- [x] AC13 KB config: `mt5.yaml` loads, builds a processor, has explicit `access.callers`, `local` sees it and another caller does not, a copy without `access` is refused; `es.analyzer` reaches the index body — `test_mt5_*`
- [x] AC14 Integration: tiny synthetic zh+en manual through the real pipeline into ES (unique names, deleted afterwards): chunks carry language, page_key, links_to; language filter works; page-key dedupe on search; analyzer `cjk` from the YAML reaches the index — `tests/integration/test_chm_ingest_es.py`

## Out of scope
Images (P6), table summaries, analyzer ablation and the question set (P4-02), cross-language experiments (P4-02), `get_context` for the other language version.

## Handoff (filled at the end)
- Files: `src/rag/data/chm/` (encoding, toc, source, unpack, connector, CLI), `src/rag/ingest/{chm,chm_html,clean_chm,chm_report}.py`, `src/rag/retrieve/page_groups.py`, changes in `data/base.py` (`extra`), `ingest/{clean,build,job,__init__}.py`, `retrieve/{pipeline,search,hybrid,query,__init__}.py`, `index/mapping.py`, `config/kb.py`, `src/rag/testing/chm.py` (invented demo manual), `kb/private/mt5.yaml`, `.gitignore`, `pyproject.toml` + `uv.lock` (beautifulsoup4 declared), README, `docs/decisions.md`, `docs/reports/P4-report.md`, tests `tests/unit/test_{chm_*,kb_mt5,retrieve_page_groups}.py`, `tests/integration/test_chm_ingest_es.py`; `test_index_mapping.py` got the two new field names (the spec adds them).
- Decisions (details in `docs/decisions.md`; `dedupe` grouped by language and title path, `also_matched_languages`, fidelity check): 7-Zip unpack behind a thin interface; encoding by UTF-8 first, declared charset through the browser aliases, CHM locale hint, mojibake check; `page_key` = path without extension, lower case (the zh pages are a subset of the en pages, same file names); DOM-based HTML to Markdown, no `md_markup` in the chain; only one text rule (`chm_title_echo`) because the pages have little noise; removal guard on every rule, limits kept out of `pipeline_version`; `dedupe` grouped by language; `dedupe_by_page_key` after the text dedupe and before the rerank, question language by Chinese characters; `mt5.yaml` tracked, `access.callers: [local]`; docstore and embedding cache in `C:/workspace/rag-platform-data/mt5/`.
- Real numbers: 12,599 pages (zh 6,273, en 6,326); index `kb_mt5_v1` (alias `kb_mt5`) holds 21,913 chunks (en 11,138, zh 10,775) written by the committed `mt5.yaml` (both languages in full) with the shared docstore and embedding cache; spend 2.0686 CNY in three stages (English 1.0375, 10 Chinese sections 0.3741, the rest and a re-process 0.6570); 35 pages with equal text under different title paths are all kept; 1,283 title echoes; fidelity: 0 of 8,305,764 visible letters, digits and Chinese characters lost on the real pages (8 pages lost text before the note-box fix). Smoke retrieval (15 title-based questions) on the final index: hit@1 11, hit@3 13, hit@10 14, one not found.
- Open issues: `tests/integration/test_otel_collector.py::test_collector_exports_to_jaeger_and_file` fails in this worktree (it reads `data/otel/traces.jsonl` of the checkout it runs in, the collector writes into the main checkout's folder), unrelated; tables cut by the token cap do not repeat their header (P2); `links_to` is stored but not returned by search; `also_matched_languages` reflects the candidates only; narrowing `sections` or using `--limit` against the real index deletes the left-out pages (documented, experiments only); the last full run needed `BUDGET_STOP_CNY=30` for that run because the conservative estimate (7.13 CNY, all pages re-processed, vectors cached) exceeded the stop line, with actual spend watched by a script (0.657 CNY); one equivalent mutation survives.
- Review rounds: 1 (CHANGES: manual text in early commits, note boxes lost text, `mt5.yaml` not reproducing the index, two test gaps, `other_languages` wording, equal-text pages with different titles; all addressed, history rewritten). Tests: unit 1,356 passed, 2 skipped; integration 50 passed (+1 unrelated failure above); about 60 mutation checks, none surviving except the one equivalent.
