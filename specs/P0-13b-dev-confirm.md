# P0-13b Confirm the P0-13 retrieval settings on the synthetic dev set

Status: in progress
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
- [ ] Dev runs, pre- and post-rerank: title_boost 2.0 (P0-12 baseline), 1.0 (chosen), 0.5; title 2.0 + vector
  weight 2; title 1.0 + vector weight 2; rerank_n 10 vs 20 for the chosen setting.
- [ ] `compare` of each variant against the chosen setting on Hit@1 and MRR@8 (McNemar, permutation).
- [ ] Decision recorded in `docs/decisions.md` (Chinese); config changed only on a reliable dev win.
- [ ] Holdout run once with the final settings; section 「P0-13 在开发集上的确认」 in `docs/reports/P0-report.md`.
- [ ] `RetrievalConfig()` defaults to `title_boost` 1.0 — `tests/unit/test_retrieve_search.py::test_retrieval_config_defaults` (expected value changed by this spec).

## Out of scope
New variants beyond P0-13's list; threshold recalibration (P2); latency work.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
