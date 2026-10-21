# P6-01 PDF and image sources (Docling, table summary index, image descriptions)

Status: in progress
Phase: P6
Depends on: P0-04, P0-06, P4-01
Design sections: s5-2, s5-3, s5-4, s6, s6-4, s7-2, s14-2, s16-4
Calls real models: yes (vision model for figures, text model for table summaries, embeddings; budget 0.8 CNY in total, one end-to-end `ask_docs` run at most 0.1 CNY)

## Goal
PDF files and standalone images become a new private knowledge base `pdf_demo` through the existing pipeline:
a `pdf_dir` connector converts each PDF with Docling (once per file hash, cached), one Document per PDF page
(page number in the metadata, citations can say "page N"); figures are described by a vision model and the
description is written into the page text at the figure's position; every table is its own chunk (header
repeated when the token cap cuts it) and gets a model-written summary chunk linked by `table_id`; at search time a
summary hit is replaced by its table chunk. Retrieval, rerank, answer and agent code is not changed beyond one
config-switched postprocessor and extra returned fields.

## Interface
- Connector `pdf_dir` (`rag.data.pdf`): `source: {path, files?, limit?, docling: {ocr, table_structure, images_scale, num_threads}, cache_dir}`;
  files `.pdf` give one `SourceDocument` per page (`doc_id` `<kb>:<file>#p<N>`, content = page Markdown, `extra`:
  `page`, `pages`, `title`, `carried_title`, `figures`), files `.png/.jpg/.jpeg` one each.
  `rag.data.pdf.convert`: `convert_pdf(path, *, cache_dir, options, engine=None) -> ConvertedPdf` (cache key: file sha256,
  Docling version, our post-processing version, options), `normalise_headings`.
- Readers `pdf` and `image` (`rag.ingest.pdf`): page Markdown + figure descriptions (`<!-- image -->` replaced by
  `[Image description (written by a model, not part of the original document): ...]`), stats for the report.
  `PipelineConfig.reader_params` (new, default empty, not part of `pipeline_version` when empty).
- `rag.ingest.describe`: `ModelTextCache` (sqlite, key kind + model + prompt version + content sha256), `Describer` /
  `TableSummarizer` protocols, `GatewayVisionDescriber`, `GatewayTableSummarizer`, `SpendGuard(budget_cny)`, `estimate_*`.
- Chunk steps (`rag.ingest.tables`): `table_chunks` (each table its own chunk with `table_id`, `chunk_kind`; split by rows with the header
  repeated), `mark_generated_text`, `table_summary` (`mode: model|rule|off`). `token_split(repeat_table_header: bool = False)`.
  `merge_tiny` never merges a table chunk.
- Index fields `page` (integer), `table_id`, `chunk_kind` (keyword), returned by search with `has_generated_text`.
- `RetrievalConfig.replace_table_summaries: bool = False`; `rag.retrieve.table_groups.TableSummaryReplace` (wired like `PageKeyDedupe`).
- `kb/private/pdf_demo.yaml`; `python -m rag.ingest.pdf_report --kb pdf_demo [--estimate]`; `data/manifests/pdf_demo.json`
  (URLs + sha256, the PDFs themselves are not committed); questions in `evals/datasets/pdf_demo/`.

## Acceptance criteria
- [ ] AC1 Gateway check: which model reads images, latency and price estimate recorded — `docs/reports/P6-report.md` (real call, no test)
- [ ] AC2 Conversion is deterministic and cached by file hash + Docling version + options; second run calls the engine 0 times — `test_convert_*`
- [ ] AC3 Headings become a hierarchy (title `#`, `3.1` is level 3, unnumbered `##`); carried headings give each page its title path — `test_normalise_headings_*`, `test_carried_title_*`
- [ ] AC4 Connector: one `SourceDocument` per page with stable ids, page number, source_version = file hash; images as files; limit and files — `test_pdf_connector_*`
- [ ] AC5 Reader: Document contract (s5-4), page metadata, description written at the marker position, small images skipped, no description when off — `test_pdf_reader_*`
- [ ] AC6 Cache and cost: second read makes 0 model calls; estimate before running; budget stop before the call; stats count images described, tables summarised, cache hits, spend — `test_describe_cache_*`, `test_spend_guard_*`, `test_pdf_report_*`
- [ ] AC7 Tables: kept whole as one chunk when they fit, split by rows with repeated header when not, caption stays with the table, text around is not merged into it — `test_table_chunks_*`
- [ ] AC8 Header repeat in `token_split` is off by default and the default output for a fixed fixture is identical to before — `test_token_split_default_table_output_unchanged`, `test_token_split_repeats_table_header`
- [ ] AC9 Summary chunks: linked to the table by `table_id`, one per table, written by model / rule / off, model error falls back to the rule text, generated flag set — `test_table_summary_*`, `test_mark_generated_text_*`
- [ ] AC10 Search: a summary hit is replaced by its table chunk(s) at the summary's rank, duplicates removed, ACL kept, fetch error keeps the summary; off by default and config version unchanged when off — `test_table_replace_*`, `test_search_config_version_ignores_replace_table_summaries_when_off`
- [ ] AC11 Mapping and search return the new fields; existing KBs unchanged — `test_index_mapping.py`, `test_search_reads_pdf_fields_*`
- [ ] AC12 KB config: `pdf_demo.yaml` loads, builds a processor, private (`access.callers`), a copy without `access` refused — `test_pdf_demo_*`
- [ ] AC13 Other layers untouched: the same `make_search_pipeline` code path serves the new KB, only config differs — `test_pdf_demo_uses_the_same_search_chain`
- [ ] AC14 Integration: tiny PDF (generated in the test) through Docling and the real pipeline into ES with unique names, then deleted — `tests/integration/test_pdf_ingest_es.py`

## Out of scope
Scanned PDFs and OCR (switch exists, not evaluated), multi-column and formula-heavy papers (formulas stay `<!-- formula-not-decoded -->`),
tables that continue on the next page (two tables), multimodal vectors, Word/HTML/Confluence, answer-layer changes (the prompt shows `page` through `metadata.llm`).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
