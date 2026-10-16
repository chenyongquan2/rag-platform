# P0-03 Hugging Face data download

Status: in progress
Phase: P0
Depends on: P0-01
Design sections: s1-1, s14-2, s14-6 (also s5-1 connector registry, s5-4 doc_id, s16-2 repo layout)
Calls real models: no

## Goal
The HF documentation corpus (`m-ric/huggingface_doc`, 2,647 docs) and its public eval set
(`m-ric/huggingface_doc_qa_eval`, 65 questions) can be downloaded from a configurable mirror into a
git-ignored cache, are verified against a committed manifest (size, sha256, rows, license, source URL),
and are exposed as an `hf_dataset` connector with stable doc ids and an eval-set loader.

## Interface
- `ServiceSettings`: `hf_endpoint` (`HF_ENDPOINT`, default `https://hf-mirror.com`), `hf_data_dir`
  (`HF_DATA_DIR`, default `data/raw/hf`; point it at a shared directory to reuse one cache), `hf_timeout_s`.
- `rag.data.make_doc_id(kb, source_rel, unit=None) -> str` = `"<kb>:<source_rel>"` (+ `"#<unit>"`), s5-4.
- `rag.data.SourceDocument` (frozen): `doc_id, kb, source_rel, source_url, ext, content: bytes, source_version`.
- `rag.data.hf`: `Manifest`/`DatasetEntry`/`FileEntry`, `load_manifest(path)`, `save_manifest(m, path)`,
  `verify_file(path, file_entry)`, `ensure_dataset(entry, *, endpoint, data_dir, timeout_s, pin=False)`,
  `HfDataError`, `ManifestMismatchError`, `NotPinnedError`.
- Connector `hf_dataset` = `HfDatasetConnector(*, kb, dataset, manifest, limit=None, settings=None)`:
  `mode = "snapshot"`, `documents() -> Iterator[SourceDocument]`, `doc_ids() -> list[str]`.
  `limit` = first N rows of the pinned parquet file. `kb/public/hf_docs.yaml` gains `source.manifest`.
- `EvalQuestion(qid, question, answer, source_doc, doc_id, context)`,
  `load_eval_questions(manifest, *, kb, settings=None)`, `missing_sources(questions, doc_ids)`.
- CLI `uv run python -m rag.data.hf download [--pin] [--manifest P] [--endpoint URL] [--data-dir D]`
  and `... verify [--manifest P] [--data-dir D]`; exit code 1 on any mismatch or missing eval source.
- Committed manifest `data/manifests/hf_docs.json`; cache layout `<hf_data_dir>/<repo_id>/<revision>/<path>`.

## Acceptance criteria
All in `tests/unit/test_hf_data.py` (tiny parquet fixture written at test time, `file://` mirror, no network).
- [ ] AC1 doc_id is kb + normalized relative path (+ unit); empty or `..` paths are rejected — `test_make_doc_id_is_kb_plus_relative_path`
- [ ] AC2 Settings expose `hf_endpoint`, `hf_data_dir`, `hf_timeout_s` with defaults and env override — `test_hf_settings_defaults_and_env_override`
- [ ] AC3 Download fetches `<endpoint>/datasets/<repo>/resolve/<revision>/<path>`, caches under the data dir, does not re-fetch a verified file, and never leaves a bad file in the cache — `test_download_caches_and_verifies`
- [ ] AC4 Verification fails on wrong sha256, wrong size and wrong row count; a missing file says to run `download` — `test_verify_rejects_wrong_hash_size_and_rows`
- [ ] AC5 `--pin` fills unpinned size, sha256 and rows; download without pin refuses unpinned entries — `test_pin_fills_unpinned_manifest_entries`
- [ ] AC6 Doc ids are stable across loads and independent of row order — `test_doc_ids_stable_across_loads_and_row_order`
- [ ] AC7 `limit` returns the first N rows, the same on every load; invalid limits are rejected — `test_limit_takes_first_n_deterministically`
- [ ] AC8 `hf_dataset` is registered and created from the KB config; the YAML dataset must match the manifest — `test_hf_dataset_connector_registered_and_created_from_kb_config`
- [ ] AC9 Eval questions map `source_doc` to the corpus doc_id scheme; missing sources are reported — `test_eval_questions_map_to_corpus_doc_ids`, `test_missing_eval_sources_reported`
- [ ] AC10 CLI `download` and `verify` print counts; `verify` exits 1 when an eval source is missing — `test_cli_download_and_verify_print_counts`
- [ ] AC11 The committed manifest is fully pinned (2,647 docs, 65 questions, licenses, source URLs) and `data/raw/` is git-ignored — `test_committed_manifest_is_pinned`, `test_raw_data_dir_is_git_ignored`

Also (checked by running, not by a test): real `download --pin` then `verify` via hf-mirror report 2,647 docs,
65 questions, 0 missing eval sources.

## Out of scope
Markdown/ipynb readers, cleaning and chunking (P0-04); ingestion and incremental update (P0-06); turning the
65 questions into a committed eval dataset and metrics (P0-10); synthetic questions (P1-05).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
