# P0-04 Reading, cleaning, chunking

Status: in progress
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
- `rag.ingest.build`: `pipeline_version(pipeline)`, `KbProcessor(cfg, settings)` with `read`, `clean`,
  `chunk`; `rag.ingest.report.quality_report(...)`; CLI
  `uv run python -m rag.ingest.report --kb hf_docs [--limit N | --all] [--out PATH]`.

## Acceptance criteria
Unit tests in `tests/unit/` (`test_ingest_readers.py`, `test_ingest_clean.py`, `test_ingest_chunk.py`,
`test_ingest_report.py`).
- [ ] AC1 Readers are found by extension or by the pipeline's `reader`; a JSON notebook is refused — `test_reader_lookup_by_extension`, `test_markdown_reader_rejects_json_notebook`
- [ ] AC2 Document has every contract field and none of the forbidden ones — `test_reader_contract_fields`
- [ ] AC3 Two reads give the same doc_id and hash; other files do not change it — `test_reader_stable_id_and_hash`
- [ ] AC4 `pipeline_version` follows the clean/step config and changes the Document hash — `test_pipeline_version_tracks_config`
- [ ] AC5 Embed text of a chunk is title path + text only; LLM text adds source_rel — `test_metadata_whitelist_embed_and_llm_text`
- [ ] AC6 Each rule has positive and negative cases — `test_hf_first_char_restores_lost_first_character`, `test_front_matter_moves_yaml_to_metadata`, `test_md_markup_removes_comments_and_components`, `test_md_markup_keeps_code_tables_and_comparisons`, `test_whitespace_normalizes_outside_code`, `test_skip_empty_units`, `test_dedupe_keeps_one_copy_and_records_sources`, `test_detect_language_per_unit`, `test_suspicious_instruction_flags_en_and_zh`
- [ ] AC7 Rule stats (chars removed, units hit, skips) and the `enabled` switch — `test_rule_stats_and_disable_switch`
- [ ] AC8 Cleaning runs before hashing: noise-only differences give the same Document hash — `test_cleaned_hash_ignores_removed_noise`
- [ ] AC9 Heading split gives title paths and ignores `#` inside code — `test_markdown_split_title_path_and_code_headings`
- [ ] AC10 Chunk ids are `doc_id#0..n-1` in page order across headings, unique, with parent ids — `test_chunk_ids_unique_and_ordered_across_sections`
- [ ] AC11 Token cap holds with the real reranker tokenizer (skipped if not cached) — `test_token_cap_with_reranker_tokenizer`
- [ ] AC12 Code fences stay whole when they fit; oversized ones become fenced parts — `test_code_fence_kept_whole`, `test_oversized_code_fence_split_into_fenced_parts`
- [ ] AC13 Tiny chunks merge into the next within the cap — `test_merge_tiny_merges_into_next_within_cap`
- [ ] AC14 Suspicious flag is set per chunk — `test_suspicious_flag_per_chunk`
- [ ] AC15 The steps run inside a LlamaIndex IngestionPipeline — `test_steps_run_in_ingestion_pipeline`
- [ ] AC16 Missing tokenizer gives a clear error — `test_load_tokenizer_missing_model_raises`
- [ ] AC17 Report: counts, skip reasons, rule hits, token distribution, title-only and cut-table chunks, removal warning — `test_quality_report_numbers`
- [ ] AC18 Report CLI prints the summary and writes JSON — `test_report_cli`

Also (by running): report CLI on 50 docs and on the full corpus; numbers in the handoff.

## Out of scope
Writing to ES, docstore, IngestionPipeline run with UPSERTS_AND_DELETE, `ingest_run_id`/`content_hash`/
`source_version` on chunks, reading failure fallback (P0-06); CHM/PDF readers, encoding repair, structure
noise, sensitive data (later); table grouping/summaries (P2); full injection rule set and safety set (P1-03).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
