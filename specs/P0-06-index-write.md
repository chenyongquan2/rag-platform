# P0-06 Writing to ES and incremental updates

Status: in progress
Phase: P0
Depends on: P0-01, P0-02, P0-04, P0-05
Design sections: s7-4, s7-5 (docstore per index version), s7-6, s5-4 (write-time fields), s16-8 (ingest errors)
Calls real models: yes, once at the end (50 docs, estimated < 0.05 CNY), only when the ledger shows enough budget

## Goal
`python -m rag.ingest.job --kb hf_docs` reads every document of a knowledge base, cleans it, and runs a LlamaIndex
`IngestionPipeline` (chunk steps + embedding, `UPSERTS_AND_DELETE`, JSON docstore) into our own `EsVectorStore`,
which writes new chunks first and deletes old ones only after the run (s7-4). Read failures keep the old version,
a mass-delete guard stops runs that read too little, the docstore is saved and the data generation raised only
after the whole run succeeded.

## Interface
- `rag.index.store`: `EsVectorStore(BasePydanticVectorStore)` (`stores_text=True`; `es`, `index_name`,
  `bulk_batch_size`): `begin_run(run_id, fields: Mapping[doc_id, WriteFields(source_version, content_hash)])`,
  `add(nodes) -> ids`, `delete(ref_doc_id)` (records only), `pending_deletes`, `written_docs`, `refresh()`,
  `finalize() -> FinalizeResult(updated_docs, removed_docs, deleted_chunks)`, `query(VectorStoreQuery)` (kNN only,
  hybrid search is P0-07); errors `BulkWriteError`, `FinalizeError`, `StoreFieldError`.
- `rag.index.mapping`: field `images` (keyword). `rag.index.manage`: `ensure_index_fields(es, index, body)` adds
  missing fields to an existing index; `index_embed_model(es, index)`, `set_index_embed_model(es, index, model_id)`
  (`_meta.embed_model`).
- `rag.factories.JsonFileDocumentStore.persist` writes a temporary file and renames it.
- `ServiceSettings`: `ingest_max_read_failure_ratio` (0.05), `ingest_min_input_ratio` (0.5),
  `ingest_bulk_batch_size` (500), `ingest_lock_timeout_s` (5), `ingest_report_dir` (`data/reports`).
- `rag.ingest.job`: `IngestJob(cfg, settings, *, es, redis, embed_model, connector, processor=None, names=None,
  index=None, allow_mass_delete=False, lock_provider=None, ledger=None)` with `run()` (lock + all phases) and the
  phases `prepare()`, `write()`, `commit()`; `IngestReport`; errors `IngestError`, `MassDeleteError`,
  `ReadFailureError`, `IndexModelMismatchError`; `KbChunker` (the KB's chunk steps as one TransformComponent).
- CLI `uv run python -m rag.ingest.job --kb hf_docs [--limit N | --all] [--fake-embed] [--index-prefix P]
  [--allow-mass-delete]`; `--fake-embed` writes to the `fake_` prefixed index unless a prefix is given.

## Acceptance criteria
Unit tests (`tests/unit/test_index_store.py`, `tests/unit/test_ingest_job.py`) use an in-memory fake ES and Redis;
integration tests (`tests/integration/test_ingest_es.py`) use the compose ES and Redis.
- [ ] AC1 `add` bulk-indexes one ES document per chunk: `_id` = `chunk_id` = `doc_id#n`, `content` = body text only, `title_path` and metadata fields separately, `ingest_run_id`/`content_hash`/`source_version` from the run's per-doc fields — `test_add_writes_chunk_fields`
- [ ] AC2 `add` refuses, before sending, chunks without run fields, without an embedding, or with metadata keys outside the mapping (also write-time keys in metadata) — `test_add_rejects_bad_chunks_before_sending`
- [ ] AC3 Bulk responses are checked item by item (HTTP 200 with `errors`); a failed item raises `BulkWriteError` with its id; requests carry at most `bulk_batch_size` chunks — `test_add_raises_on_partial_bulk_failure`, `test_add_sends_batches`
- [ ] AC4 `delete` only records the doc id; `finalize` deletes chunks of pending docs whose `ingest_run_id` is not this run (all chunks of a doc without new chunks), refreshes, reports counts and raises on failures — `test_delete_only_records_pending`, `test_finalize_deletes_leftovers_and_removed_docs`, `test_finalize_raises_on_failures`
- [ ] AC5 Mapping has `images`; missing fields are added to an existing index; the index records its embedding model — `test_index_mapping.py::test_mapping_matches_design`, `test_ingest_es.py::test_existing_index_gets_new_fields`
- [ ] AC6 Docstore persist is atomic (temp file + rename) — `test_factories.py::test_file_docstore_persist_is_atomic`
- [ ] AC7 The job cleans before the pipeline, counts added/updated/unchanged/deleted, saves the docstore after success, and writes a report (counts, failures, cost) — `test_job_counts_and_persists_docstore`
- [ ] AC8 A doc that fails to read is replaced by its docstore copy and counts as unchanged; more than 5% read failures abort before any change — `test_read_failure_uses_stored_copy`, `test_too_many_read_failures_abort`
- [ ] AC9 Input below 50% of the docstore aborts before `run()` (docstore file and ES untouched) unless `allow_mass_delete` — `test_mass_delete_guard_aborts_without_changes`
- [ ] AC10 A failed write leaves the docstore file and the data generation unchanged — `test_failed_run_does_not_persist_docstore`
- [ ] AC11 Real embeddings: an estimate over the budget stop aborts before any call; ledger lines carry the run id; the report has the run's cost — `test_budget_check_before_embedding`
- [ ] AC12 One job per KB at a time (file lock) — `test_ingest_lock_is_exclusive`
- [ ] AC13 An index built with one embedding model refuses a run with another; `--fake-embed` defaults to the `fake_` index — `test_index_refuses_other_embed_model`, `test_cli_fake_embed_uses_fake_prefix`
- [ ] AC14 CLI prints the summary, writes the report JSON, exit codes 0/1/2 — `test_cli_runs_and_writes_report`
- [ ] AC15 (integration) add, modify, delete, rename and revert give exactly the expected chunks — `test_five_change_cases`
- [ ] AC16 (integration) between writing new chunks and finalizing, a query still finds the modified doc — `test_query_between_write_and_finalize_finds_doc`
- [ ] AC17 (integration) a bulk partial failure leaves the docstore unsaved; the next run repairs — `test_bulk_partial_failure_then_next_run_repairs`
- [ ] AC18 (integration) changing one file leaves every other file's chunks untouched — `test_one_changed_file_leaves_others_intact`
- [ ] AC19 (integration) a read failure keeps that doc's chunks — `test_read_failure_keeps_chunks`
- [ ] AC20 (integration) the mass-delete guard aborts without touching docstore or ES — `test_mass_delete_guard_against_es`
- [ ] AC21 (integration) the data generation rises by one per successful run that changed the index, not on no-change or failed runs — `test_data_generation_once_per_changing_run`

Also (by running): the job on `--limit 50 --fake-embed`, then on 50 docs with real embeddings if the budget allows.

## Out of scope
Hybrid retrieval and filters (P0-07); the rebuild-and-switch flow and force merge (P2-01); `changes`-mode connectors
(later); Postgres docstore and multi-instance locks (later); full-corpus run (P0-12).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
