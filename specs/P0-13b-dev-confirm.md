# P0-13b Confirm the P0-13 retrieval settings on the synthetic dev set

Status: done
Phase: P1
Depends on: P0-13 (settings and variants), P1-05 (synthetic dev and holdout sets)
Design sections: s14-2, s14-4, s9-3, s10-1
Calls real models: no (question vectors cached; local reranker)

## Goal
P0-13 chose its retrieval settings on the 65 public questions, which are not a tuning set. This spec
re-runs the variants that matter on the synthetic dev set only, decides with `compare` (exact McNemar,
paired permutation) whether the settings stand, and reports the holdout set once for the final settings.

## Interface
- `scripts/p0_13b/dev_runs.sh`: the dev runs (variant KB dirs in git-ignored `data/reports/p0-13b/kb_variants`).
- Records in git-ignored `data/reports/p0-13b/`; no code change unless a variant wins reliably.
- `RetrievalConfig.title_boost` default 2.0 -> 1.0 (P0-13 deferred this until the dev set confirms it; hf_docs already sets 1.0).

## Acceptance criteria
- [x] Dev runs, pre- and post-rerank: title_boost 2.0 (P0-12 baseline), 1.0 (chosen), 0.5; title 2.0 + vector
  weight 2; title 1.0 + vector weight 2; rerank_n 10 vs 20 for the chosen setting.
- [x] `compare` of each variant against the chosen setting on Hit@1 and MRR@8 (McNemar, permutation).
- [x] Decision recorded in `docs/decisions.md` (Chinese); config changed only on a reliable dev win.
- [x] Holdout run once with the final settings; section 「P0-13 在开发集上的确认」 in `docs/reports/P0-report.md`.
- [x] A run with `--questions` records that file in `kb_config.eval.questions` — `tests/unit/test_eval_cli_run.py::test_run_records_the_question_file_it_used`
- [x] `RetrievalConfig()` defaults to `title_boost` 1.0 — `tests/unit/test_retrieve_search.py::test_retrieval_config_defaults` (expected value changed by this spec).

## Out of scope
New variants beyond P0-13's list; threshold recalibration (P2); latency work.

## Handoff (filled at the end)
- Files: `scripts/p0_13b/dev_runs.sh`, `src/rag/config/kb.py` (`title_boost` default 1.0), `src/rag/eval/retrieval/cli.py` (`_with_question_file`), `tests/unit/test_eval_cli_run.py`, `tests/unit/test_retrieve_search.py` (expected default), `docs/decisions.md`, `docs/reports/P0-report.md` (section 12), `docs/progress.md`. Records in git-ignored `data/reports/p0-13b/` (11 dev runs, 9 compares, 1 holdout run).
- Results (99 dev questions): chosen setting (title 1.0, equal weights, rerank_n 10) Hit@1 0.788, Hit@5 0.919, MRR@8 0.841 after rerank, 0.616 / 0.747 before; title 2.0 before rerank MRR@8 0.656 (27 worse / 8 better, p 0.0013); title 1.0 + vector 2 after rerank Hit@5 0.848 (7/0, p 0.016), MRR@8 0.790 (p 0.027); title 0.5 identical; rerank_n 20 Hit@1 0.768 (3/1, p 0.63), P95 8.6 s. Holdout once: Hit@1 0.720, Hit@8 0.890, MRR@8 0.790.
- Decisions: P0-13 settings stand; code default `title_boost` 1.0.
- Open issues: run records written before this fix (P1-05 and P0-13b synthetic runs) carry `kb_config.eval.questions` = huggingface_doc_qa_eval.jsonl; `question_set.path` is right; dev/holdout label misses (3 and 2) lower every variant alike; analyzer and per-field variants that need a reindex not tried.
- Review rounds: 1 (APPROVE with two low notes, fixed: holdout wording says no sign of overfitting, recorded question file traced to `kb_config` and fixed with a test).
