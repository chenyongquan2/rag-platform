# P7-01 Model selection (scaled down)

Status: review
Phase: P7
Depends on: P0-12 (ledger, rerank curve), P1-04 (judge), P1-06 (citation check), P2-01 (alias switch), P2-02 (threshold script), P2-03 (`ablate`), P3-07 (standard RAG baseline records)
Design sections: s19 (P7 row), s10-1, s10-2, s11-1, s14-3, s14-4
Calls real models: yes (cap 1.5 CNY: rerankers 0, generation at most 1.0, embedding at most 0.4; pilots first)

## Goal
One decision table per layer (metric, latency, cost, paired test against the current model) on the HF docs knowledge
base only, written to `docs/reports/P7-report.md`, with a recommendation per layer. Defaults stay unchanged; a switch
is the owner's decision.

## Interface
- `rag.eval.answer.runner.StoredContexts(run)`: a `ContextSource` that gives each question the chunks an earlier
  answer run gave it (same fragments, same order), so two chat models are compared on identical contexts.
- CLI `python -m rag.eval.answer run ... --contexts-from RUN` uses it instead of the retriever; with
  `--retry-errors` only RUN's gateway errors are answered again (`gateway_error_items`) and merged into a new run
  (`merge_retried`): the gateway timed out on connect for some calls, which says nothing about a model.
- `rag.eval.answer.pairing`: `PairedOutcome` (one question of one system: judge score, found, fully cited, citation
  recall, latency, CNY), `paired_answer_table(base, candidate)` (McNemar on score 3 and on fully cited, permutation
  on judge score and citation recall, medians and means) and `render_paired_answers(...)`.
- `scripts/p7_01/`: run scripts and adapters (P3-07 records and answer runs into `PairedOutcome`), the small-index
  embedding comparison. No config default changes.

## Acceptance criteria
Code (unit tests):
- [x] AC1 `StoredContexts` returns the stored chunks in order and the stored not-found flag; an unknown question id
  raises — `tests/unit/test_eval_answer_stored_contexts.py::test_stored_contexts_replay_chunks_and_not_found`, `::test_unknown_qid_raises`
- [x] AC2 `answer run --contexts-from` answers from the stored run and never builds a retriever — `tests/unit/test_eval_answer_stored_contexts.py::test_cli_contexts_from_skips_the_retriever`
- [x] AC2b `answer run --contexts-from RUN --retry-errors` answers again only RUN's gateway errors and writes the
  merged run; without `--contexts-from` it is a usage error — `tests/unit/test_eval_answer_stored_contexts.py::test_retry_errors_answers_only_the_failed_questions_and_merges`, `::test_retry_errors_needs_contexts_from`
- [x] AC3 The paired table counts better and worse per binary metric with the exact McNemar p and runs the permutation
  test on the continuous ones, on the common question ids only — `tests/unit/test_eval_answer_pairing.py::test_paired_table_counts_and_tests`
- [x] AC4 The rendered table names both systems, n, every metric with its p, latency and CNY per question — `tests/unit/test_eval_answer_pairing.py::test_render_has_metrics_p_latency_cost`

Experiments (`docs/reports/P7-report.md`):
- [x] AC5 Rerankers: current `BAAI/bge-reranker-base` against `cross-encoder/ms-marco-MiniLM-L-6-v2` and
  `BAAI/bge-reranker-v2-m3` on the 99 dev questions (same fused candidates, cached query vectors): Hit@1, Hit@8, MRR,
  McNemar and permutation p, rerank latency median and P95 for 10 candidates, CPU load; threshold re-derived (P2-02)
  for a candidate that wins
- [x] AC6 Generation: two cheap gateway models of other vendors than the judge, on the 65 public questions with the
  same contexts as the current model; score 3, citation precision and recall, format compliance, latency, CNY per
  question, paired against `deepseek-v4-flash` (P3-07 standard records)
- [x] AC7 Embedding: cost and time of switching (26,372 chunks, P0-12 ledger tokens), alias-switch procedure (P2-01),
  and a labelled small-index approximation with one local alternative model if it fits
- [x] AC8 Judge: kept; what comparing judges needs (the human sheets)
- [x] AC9 Spend per part within the caps; ledger before and after

## Out of scope
MT5 and PDF knowledge bases; holdout set; rebuilding the real index; changing any default; semantic-cache distance and
sufficiency thresholds (no model changed that feeds them); judge comparison with human labels (sheets unscored).

## Handoff (filled at the end)
- Files: `src/rag/eval/answer/runner.py` (`StoredContexts`, `gateway_error_items`, `merge_retried`), `cli.py`
  (`--contexts-from`, `--retry-errors`), new `pairing.py`; tests `test_eval_answer_stored_contexts.py` (5),
  `test_eval_answer_pairing.py` (3); `scripts/p7_01/`; `docs/reports/P7-report.md`; `docs/decisions.md`. Raw records
  in git-ignored `data/reports/p7-01/`.
- Decisions: no default changed. Rerankers: MiniLM-L-6 reliably worse on dev Hit@1 (0.788 to 0.707, p 0.039),
  bge-reranker-v2-m3 no reliable difference and about 3 times slower (9.5 s per 10 candidates), so the threshold was
  not re-set (the sweep shows why it would have to be). Generation: glm-5.3-flash equal to deepseek-v4-flash question
  by question (64/65 score 3), gemini-3.5-flash-lite 3 worse (p 0.25); baseline is the P3-07 standard records (63 of
  65 questions with identical pages). Embedding: the gateway has no second embedding model; small-index in-memory
  approximation with local bge-base-en-v1.5, no reliable difference. Judge kept. Spend 1.0422 CNY (generation part
  1.0413 against its 1.0 share, mostly two minimax timeouts).
- Open issues: glm citation check not run (about 0.2 CNY); v2-m3 threshold scores not collected; the query-side embedding
  model check exists but runs only when the retrieval pipeline is built (`hybrid_parts`, unit-tested), so a running
  process is not re-checked after an alias switch (untested; safe order: switch the alias, then restart); reranker
  latency noisy on the shared machine (three MiniLM runs started at 37%, 92%, 37% CPU); `tests/unit/test_cost.py` reads `.env`
  (stop line 33), so commits set `BUDGET_STOP_CNY=28` in the commit environment (P3-07 noted the same).
- Review rounds:
