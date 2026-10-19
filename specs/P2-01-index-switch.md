# P2-01 New index version, gate, alias switch and rollback

Status: done
Phase: P2
Depends on: P0-06 (ingest job, store, docstore per index), P0-02 (alias, versions), P0-13 (retrieval settings)
Design sections: s7-4, s7-5, s7-6, s19 (P2 row)
Calls real models: no (FakeEmbedding only; the real gate, the retrieval eval, is P2-03)

## Goal
The whole "new index version" flow of s7-5 as one command, plus the proof that the five change cases keep the
index searchable: `rebuild` creates `v(N+1)` with its own docstore, writes everything, runs one more incremental sync,
force-merges, asks an injected gate, and only then moves the alias (data generation +1). The old index is kept,
`rollback` moves the alias back. Any failure before the switch leaves the alias where it was and says so.

## Interface
- `rag.index.rebuild.rebuild_kb(cfg, settings, *, es, redis, connector, names, embed_factory, gate, processor=None,
  version=None, allow_mass_delete=False, allow_unreadable_docs=False, lock_provider=None, after_step=None)
  -> RebuildReport`; steps in order: `create`, `full_write`, `sync`, `check`, `merge`, `gate`, `switch`.
- `Gate = Callable[[GateContext], GateResult | bool]`; `GateContext(es, names, new_index, old_index)`;
  `GateResult(passed, detail, metrics)`; `chunk_count_gate(min_ratio)` (a sanity floor, not a quality gate).
- `rag.index.rebuild.rollback_kb(settings, *, es, redis, names, to_version=None, lock_provider=None) -> RollbackResult`.
- Errors: `RebuildError(step, new_index, alias_target, alias_untouched, report)`, `GateFailedError(RebuildError)`,
  `RollbackError`, `GenerationNotRaisedError` (alias moved, Redis down).
- `rag.index.manage`: `force_merge_index` (closes and opens the index first, see decisions), `update_index_meta`, `index_meta`.
- `IngestJob(..., lock_name=None, raise_generation=True)`: the rebuild's jobs use their own lock name (the old
  index keeps taking incremental updates) and do not raise the generation (the switch does, once).
- CLI: `python -m rag.index.cli rebuild --kb K [--limit N | --all] [--fake-embed] [--version N] [--min-chunk-ratio R |
  --no-gate] [--allow-mass-delete] [--allow-unreadable-docs]`, `... rollback --kb K [--to-version N]`.
- Settings: `rebuild_force_merge_timeout_s`, `rebuild_min_chunk_ratio`.

## Acceptance criteria
Unit tests: `tests/unit/test_index_rebuild.py` (fake ES and Redis). Integration: `tests/integration/test_index_switch_es.py`.
- [x] AC1 `rebuild` writes `v(N+1)` with a new docstore bound to it, keeps `v(N)` and its docstore, switches the alias and reports both — `test_rebuild_creates_next_version_with_own_docstore_and_switches`
- [x] AC2 Source changes made during the full write (and incremental runs on `v(N)` meanwhile) end up in `v(N+1)` through the sync run, with no stale chunks; the incremental run on `v(N)` is not blocked — `test_sync_picks_up_changes_made_during_the_full_write`, `test_rebuild_does_not_block_incremental_runs_on_the_old_index` (an incremental run on the alias from inside a running phase)
- [x] AC3 The force merge comes after the last write and before the gate — `test_force_merge_after_last_write_before_gate`
- [x] AC4 A failing or raising gate leaves the alias, the generation and `v(N)` alone and the error says so — `test_gate_failure_leaves_alias_untouched`, `test_gate_exception_leaves_alias_untouched`
- [x] AC5 A failure at any step before the switch leaves the alias untouched and the message names the step — `test_failure_before_switch_leaves_alias_untouched` (start, create, full_write, sync, merge)
- [x] AC6 The data generation rises by exactly 1 per rebuild (the switch), not by the jobs writing `v(N+1)` — `test_switch_raises_generation_once`
- [x] AC7 An alias that moved during the rebuild makes the switch fail as a whole — `test_alias_moved_during_rebuild_fails_without_switching` (generation unchanged too)
- [x] AC8 A page that is unreadable now but present in the old index blocks the switch unless allowed — `test_unreadable_page_of_old_index_blocks_switch`
- [x] AC9 `rollback` points the alias back, deletes nothing, raises the generation (never restores a lower number, see docs/decisions.md), and refuses without a recorded previous index — `test_rollback_points_alias_back_without_deleting`, `test_rollback_refuses_without_switch_record`
- [x] AC10 One rebuild per knowledge base at a time; no alias or an aliased target version is refused; a leftover unaliased version can be resumed — `test_second_rebuild_is_refused`, `test_rebuild_needs_an_alias`, `test_resume_existing_unaliased_version`
- [x] AC11 `chunk_count_gate` compares chunk counts with the old index — `test_chunk_count_gate`
- [x] AC12 `IngestJob(raise_generation=False)` does not raise the generation, `lock_name` changes the lock — `test_ingest_job_without_generation_raise_and_own_lock`
- [x] AC13 CLI `rebuild` and `rollback` print a summary, write a report, exit 0/1/2 — `test_cli_rebuild_and_rollback`
- [x] AC14 (integration) edited file: other files' chunks (count, ids, run id) unchanged; the page is found at every moment of the run, and between the write of new chunks and the delete of old ones — `test_case_edit_one_file`
- [x] AC15 (integration) added file — `test_case_add_one_file`
- [x] AC16 (integration) deleted file — `test_case_delete_one_file`
- [x] AC17 (integration) unreadable file keeps its old chunks; in a rebuild it blocks the switch — `test_case_unreadable_file_keeps_old_version`, `test_unreadable_file_blocks_rebuild_switch`
- [x] AC18 (integration) chunking settings changed: new version flow, no chunk id keeps its text, old index untouched until the switch, alias switched — `test_case_chunking_change_goes_through_new_version`
- [x] AC19 (integration) queries through the alias in a loop across the switch: zero errors, zero empty results — `test_search_is_never_interrupted_during_the_switch`
- [x] AC20 (integration) rollback on real ES; gate failure keeps the alias; the merged new index has no deleted documents and one segment — `test_rollback_against_es`, `test_gate_failure_keeps_alias_against_es`, `test_force_merge_removes_deleted_documents`
- [x] AC21 A close or open that fails around the force merge leaves the index open when it can, and a resume opens a version left closed; the gate failure message says a changed config needs a new version — `test_force_merge_reopens_the_index_when_close_fails_after_applying`, `test_resume_opens_a_version_left_closed`, `test_resume_reopens_a_version_left_closed` (integration), `test_gate_failure_leaves_alias_untouched`
- [x] AC22 A switch or roll back that could not be recorded in `_meta` is shown (summary warning, `RollbackResult.recorded`); stale `rolled_back_*` keys are cleared; the rebuild summary names the index prefix — `test_unrecorded_switch_is_visible_and_rollback_still_works`, `test_rollback_reports_when_it_could_not_record`, `test_rollback_points_alias_back_without_deleting`, `test_cli_rebuild_prints_the_index_prefix`

## Out of scope
The real gate (retrieval eval on the dev set, P2-03), threshold and ablation scripts (P2-02, P2-03), deleting old
versions after a week (manual `delete_kb_index`), a changes-mode connector, multi-instance locks.

## Handoff (filled at the end)
- Files: `src/rag/index/rebuild.py` (new: `rebuild_kb`, `rollback_kb`, gates, report, errors), `src/rag/index/manage.py` (`force_merge_index`, `index_meta`, `update_index_meta`), `src/rag/index/cli.py` (`rebuild`, `rollback`, `status` shows the rollback target, injectable deps), `src/rag/ingest/job.py` (`lock_name`, `raise_generation`, `close_embedding`), `src/rag/config/settings.py` (`rebuild_*`), `src/rag/testing/services.py` (fake `forcemerge`, `close`, `open`, `segments`), `tests/unit/test_index_rebuild.py` (27 tests), `tests/integration/test_index_switch_es.py` (11 tests), `README.md`, `docs/decisions.md`, `docs/plan.md`.
- Decisions (docs/decisions.md): a plain force merge does not drop deleted chunks on a single-copy index (soft-delete history; N 30 vs 5 measured), so the new index is closed and opened first; rollback raises the data generation instead of restoring the old number; each phase of a rebuild has its own run id, own lock, and does not raise the generation; unreadable pages that the old index holds block the switch; the default gate is only a chunk-count floor (the retrieval eval is P2-03); leftover versions are kept and resumed with `--version`.
- Open issues: the live index accumulates soft-deleted history through incremental runs, so its BM25 IDF is slightly off until it is replaced by a rebuild (design s7-4 mentions a small bias; the close/open trick is not applied to a serving index); design s7-5 point 3 says "force merge cleans deleted chunks", true only with the close/open (not edited in the design copy); a failed close or open is retried once and a resume (`--version N`) opens the version again, but until then the new index may be closed; the CLI default gate checks counts only, so P2-03 must pass the eval gate for a real run; a full-corpus rebuild time is not measured here (P2-03 measures it).
- Review rounds: 2. Round 1 CHANGES (8 notes fixed: in-phase proof that the old index is not blocked, closed-index handling with resume, resume wording after a gate failure, more failing steps tested, unrecorded switch shown, plain-Chinese decisions, prefix in the summary). Round 2 APPROVE at 037f22c with two nits (AC order, stale test counts), fixed.
