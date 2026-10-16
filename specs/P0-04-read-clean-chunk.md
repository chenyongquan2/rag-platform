# P0-04 Reading, cleaning, chunking

Status: review
Phase: P0
Depends on: P0-01, P0-03
Design sections: s5-1, s5-2, s5-4, s6-1, s6-2, s6-3, s6-4, s7-1, s7-2, s7-4
Calls real models: no

## Goal
`rag.ingest` turns connector `SourceDocument`s into LlamaIndex `Document`s that meet the s5-4 contract,
cleans them with named, switchable rules before the IngestionPipeline (s7-4), and chunks them by heading,
then by a reranker-token cap, merges tiny chunks and renumbers ids over the whole page. A data quality
report (s6-4) runs over N docs from the CLI. Pure: no network, no ES.

## Interface
- `rag.ingest.readers`: `MarkdownReader(BaseReader)` registered as reader `markdown` (`.md`, `.mdx`;
  `hf_docs.yaml` maps the dataset's flattened `.ipynb` text to it); `reader_name_for(ext, pipeline)`.
  Document: `id_ = doc_id`; metadata `kb, doc_id, source_rel, unit, title_path, source_url, doc_type,
  language, acl_groups, pipeline_version` (no mtime/size/source_version/ingest_run_id/content_hash);
  excluded embed/llm keys = everything outside `cfg.metadata` whitelists (and empty values).
- `rag.ingest.clean`: `CleanRule(TransformComponent)` with `enabled`, `stats: RuleStats(units_seen,
  units_hit, chars_removed, chars_added, skipped{doc_id: reason})`. Rules (in `rag.registry.steps`):
  `hf_first_char`, `front_matter`, `md_markup`, `whitespace`, `skip_empty`, `dedupe`, `detect_language`,
  `suspicious_instruction` (also usable as a chunk step); preset `md_clean` expands to the generic ones.
- `rag.ingest.chunk` steps: `markdown_split` (MarkdownNodeParser subclass, `title_path`, `parent_id`),
  `token_split` (SentenceSplitter subclass: reranker tokenizer, `paragraph_separator="\n\n"`, overlap 0,
  fenced code kept whole when it fits), `merge_tiny`, `assign_chunk_ids` (`doc_id#<n>`, `chunk_index`,
  final whitelist).
- `rag.ingest.tokens.load_tokenizer(model) -> Callable[[str], list[int]]` (local HF cache only).
- `rag.ingest.build`: `pipeline_version(cfg, ext)`, `KbProcessor(cfg, *, settings, tokenizer)` with `read`,
  `clean`, `chunk`, `steps_for(ext)` (for P0-06's IngestionPipeline), `rule_stats`; `rag.ingest.report.quality_report(...)`; CLI
  `uv run python -m rag.ingest.report --kb hf_docs [--limit N | --all] [--out PATH]`.

## Acceptance criteria
Unit tests in `tests/unit/` (`test_ingest_readers.py`, `test_ingest_clean.py`, `test_ingest_chunk.py`,
`test_ingest_report.py`).
- [x] AC1 Readers are found by extension or by the pipeline's `reader`; a JSON notebook is refused; `hf_docs.yaml` builds for .md/.mdx/.ipynb — `test_reader_lookup_by_extension`, `test_markdown_reader_rejects_json_notebook`, `test_hf_docs_config_builds`
- [x] AC2 Document has every contract field and none of the forbidden ones — `test_reader_contract_fields`
- [x] AC3 Two reads give the same doc_id and hash; other files do not change it — `test_reader_stable_id_and_hash`
- [x] AC4 `pipeline_version` follows the clean/step config and changes the Document hash — `test_pipeline_version_tracks_config`
- [x] AC5 Embed text of a chunk is title path + text only; LLM text adds source_rel — `test_metadata_whitelist_embed_and_llm_text`
- [x] AC6 Each rule has positive and negative cases — `test_hf_first_char_restores_lost_first_character`, `test_front_matter_moves_yaml_to_metadata`, `test_md_markup_removes_comments_and_components`, `test_md_markup_keeps_code_tables_and_comparisons`, `test_whitespace_normalizes_outside_code`, `test_skip_empty_units`, `test_dedupe_keeps_one_copy_and_records_sources`, `test_detect_language_per_unit`, `test_suspicious_instruction_flags_en_and_zh`
- [x] AC7 Rule stats (chars removed, units hit, skips) and the `enabled` switch — `test_rule_stats_and_disable_switch`
- [x] AC8 Cleaning runs before hashing: noise-only differences give the same Document hash — `test_cleaned_hash_ignores_removed_noise`
- [x] AC9 Heading split gives title paths and ignores `#` inside code — `test_markdown_split_title_path_and_code_headings`
- [x] AC10 Chunk ids are `doc_id#0..n-1` in page order across headings, unique, with parent ids — `test_chunk_ids_unique_and_ordered_across_sections`
- [x] AC11 Token cap holds with the real reranker tokenizer (skipped if not cached) — `test_token_cap_with_reranker_tokenizer`
- [x] AC12 Code fences stay whole when they fit; oversized ones become fenced parts — `test_code_fence_kept_whole`, `test_oversized_code_fence_split_into_fenced_parts`
- [x] AC13 Tiny chunks merge into the next within the cap — `test_merge_tiny_merges_into_next_within_cap`
- [x] AC14 Suspicious flag is set per chunk — `test_suspicious_flag_per_chunk`
- [x] AC15 The steps run inside a LlamaIndex IngestionPipeline — `test_steps_run_in_ingestion_pipeline`
- [x] AC16 Missing tokenizer gives a clear error — `test_load_tokenizer_missing_model_raises`
- [x] AC17 Report: counts, skip reasons, rule hits, token distribution, title-only and cut-table chunks, removal warning — `test_quality_report_numbers`
- [x] AC18 Report CLI prints the summary and writes JSON — `test_report_cli`

Also (by running): report CLI on 50 docs and on the full corpus; numbers in the handoff.

## Out of scope
Writing to ES, docstore, IngestionPipeline run with UPSERTS_AND_DELETE, `ingest_run_id`/`content_hash`/
`source_version` on chunks, reading failure fallback (P0-06); CHM/PDF readers, encoding repair, structure
noise, sensitive data (later); table grouping/summaries (P2); full injection rule set and safety set (P1-03).

## Handoff (filled at the end)
- Files: `src/rag/ingest/{__init__,markdown,meta,tokens,language,readers,clean,chunk,build,report}.py`, `tests/unit/test_ingest_{readers,clean,chunk,report}.py` (29 tests; unit suite 89), `kb/public/hf_docs.yaml` (rules and steps for .md/.mdx/.ipynb), `pyproject.toml` + `uv.lock` (tokenizers, huggingface-hub declared), `.gitignore` (`data/reports/`), `docs/decisions.md`.
- Report numbers (pipeline `v1-3710a5d7eb`, cap 400): 50 docs -> 50 units kept, 438 chunks, tokens min/p50/p90/max 19/256/338/364, 0 over cap, 0 title-only, md_markup removed 8.9% (WARN, checked). Full corpus 2,647 docs -> 2,612 kept (skipped: 19 duplicate, 8 title_only, 4 toc_only, 4 too_short), 26,365 chunks, tokens 3/232/337/374 (mean 217), 0 over cap, 363 tiny, 58 title-only, 115 cut tables, 1 suspicious chunk, 5 garbled units, languages en 2,562 / zh 46 / hi 2 / de 1 / ko 1; read+clean+chunk ~2.3 min on this machine.
- Decisions (docs/decisions.md): `hf_first_char` repairs the dataset's lost first character (2,486 pages); .mdx and flattened .ipynb use the markdown reader; title_path is a " > " string without link targets; pipeline_version hashes the whole chain config; tokenizer via `tokenizers` (2 s, not 68 s); regex sentence split instead of NLTK; code fences/tables atomic, oversized fences re-fenced; heading-only chunks merge forward; parent_id = heading section `doc_id#s<k>`; md_markup scope (timm model-index comments are dropped); narrow injection patterns; script/stopword language detection; dedupe keeps the smallest doc_id.
- Open issues: tables cut by the token cap do not repeat their header (P2 table step); README_zh/ja/es/pt-br count as `en` (mostly English names); the 5% removal WARN always fires on this corpus (licence comments, image tags); uv.lock may conflict with P0-05 (httpx) on merge, re-run `uv lock`. P0-06: hand `KbProcessor.steps_for(ext)` to the IngestionPipeline and run `clean` before it.
- Review rounds:
