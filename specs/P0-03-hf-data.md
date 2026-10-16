# P0-03 Hugging Face data download

Status: review
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
- [x] AC1 doc_id is kb + normalized relative path (+ unit); empty or `..` paths are rejected — `test_make_doc_id_is_kb_plus_relative_path`
- [x] AC2 Settings expose `hf_endpoint`, `hf_data_dir`, `hf_timeout_s` with defaults and env override — `test_hf_settings_defaults_and_env_override`
- [x] AC3 Download fetches `<endpoint>/datasets/<repo>/resolve/<revision>/<path>`, caches under the data dir, does not re-fetch a verified file, and never leaves a bad file in the cache — `test_download_caches_and_verifies`
- [x] AC4 Verification fails on wrong sha256, wrong size and wrong row count; a missing file says to run `download` — `test_verify_rejects_wrong_hash_size_and_rows`
- [x] AC5 `--pin` fills unpinned size, sha256 and rows; download without pin refuses unpinned entries — `test_pin_fills_unpinned_manifest_entries`
- [x] AC6 Doc ids are stable across loads and independent of row order; empty or duplicate paths are rejected — `test_doc_ids_stable_across_loads_and_row_order`, `test_duplicate_sources_are_rejected`
- [x] AC7 `limit` returns the first N rows, the same on every load; invalid limits are rejected — `test_limit_takes_first_n_deterministically`
- [x] AC8 `hf_dataset` is registered and created from the KB config; the YAML dataset must match the manifest — `test_hf_dataset_connector_registered_and_created_from_kb_config`
- [x] AC9 Eval questions map `source_doc` to the corpus doc_id scheme; missing sources are reported — `test_eval_questions_map_to_corpus_doc_ids`, `test_missing_eval_sources_reported`
- [x] AC10 CLI `download` and `verify` print counts; `verify` exits 1 when an eval source is missing — `test_cli_download_and_verify_print_counts`
- [x] AC11 The committed manifest is fully pinned (2,647 docs, 65 questions, licenses, source URLs) and `data/raw/` is git-ignored — `test_committed_manifest_is_pinned`, `test_raw_data_dir_is_git_ignored`

Also (checked by running, not by a test): real `download --pin` then `verify` via hf-mirror report 2,647 docs,
65 questions, 0 missing eval sources (done 2026-10-17; pinned sha256 values equal the Hub's LFS object ids).

## Out of scope
Markdown/ipynb readers, cleaning and chunking (P0-04); ingestion and incremental update (P0-06); turning the
65 questions into a committed eval dataset and metrics (P0-10); synthetic questions (P1-05).

## Handoff (filled at the end)
- Files: `pyproject.toml`, `uv.lock` (pyarrow 26.0.0), `.env.example`, `src/rag/config/settings.py`, `kb/public/hf_docs.yaml`, `data/manifests/hf_docs.json`, `src/rag/data/{__init__,base}.py`, `src/rag/data/hf/{__init__,__main__,cli,connector,download,evalset,manifest}.py`, `tests/unit/test_hf_data.py`, `docs/decisions.md`. Raw data cached at `C:\workspaceag-platform\dataaw\hf` (set `HF_DATA_DIR` there from other checkouts; the main checkout's default `data/raw/hf` already points at it).
- Decisions (docs/decisions.md): corpus = Hub auto-converted parquet on `refs/convert/parquet` pinned by commit, not the CSV on main; stdlib urllib download instead of `hf_hub_download`; add pyarrow; doc_id `<kb>:<source>` (+ `#<unit>`), `source_url` = `https://github.com/<source>`; `limit` = first N rows of the pinned file; manifest size/sha256/rows recorded by `download --pin`, then required.
- Open issues: `hf_docs.yaml` has only a `.md` pipeline, but the corpus also has 468 `.mdx` and 264 `.ipynb` docs (their text is already Markdown-like, not notebook JSON) — P0-04. 50 docs (`gradio-app/.../guides/cn/`) are Chinese in a KB with `language: en`; row 3 is one of them, so it is inside `limit: 50`. `uv.lock` may conflict with P0-02 on merge (re-run `uv lock`). The 65 questions are not yet exported to `evals/datasets/hf_docs` (P0-10).
- Review rounds: pending.
