# P7-01 Model selection (scaled down)

Status: in progress
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
- CLI `python -m rag.eval.answer run ... --contexts-from RUN` uses it instead of the retriever.
- `rag.eval.answer.pairing`: `PairedOutcome` (one question of one system: judge score, found, fully cited, citation
  recall, latency, CNY), `paired_answer_table(base, candidate)` (McNemar on score 3 and on fully cited, permutation
  on judge score and citation recall, medians and means) and `render_paired_answers(...)`.
- `scripts/p7_01/`: run scripts and adapters (P3-07 records and answer runs into `PairedOutcome`), the small-index
  embedding comparison. No config default changes.

## Acceptance criteria
Code (unit tests):
- [ ] AC1 `StoredContexts` returns the stored chunks in order and the stored not-found flag; an unknown question id
  raises — `tests/unit/test_eval_answer_stored_contexts.py::test_stored_contexts_replay_chunks_and_not_found`, `::test_unknown_qid_raises`
- [ ] AC2 `answer run --contexts-from` answers from the stored run and never builds a retriever — `tests/unit/test_eval_answer_stored_contexts.py::test_cli_contexts_from_skips_the_retriever`
- [ ] AC3 The paired table counts better and worse per binary metric with the exact McNemar p and runs the permutation
  test on the continuous ones, on the common question ids only — `tests/unit/test_eval_answer_pairing.py::test_paired_table_counts_and_tests`
- [ ] AC4 The rendered table names both systems, n, every metric with its p, latency and CNY per question — `tests/unit/test_eval_answer_pairing.py::test_render_has_metrics_p_latency_cost`

Experiments (`docs/reports/P7-report.md`):
- [ ] AC5 Rerankers: current `BAAI/bge-reranker-base` against `cross-encoder/ms-marco-MiniLM-L-6-v2` and
  `BAAI/bge-reranker-v2-m3` on the 99 dev questions (same fused candidates, cached query vectors): Hit@1, Hit@8, MRR,
  McNemar and permutation p, rerank latency median and P95 for 10 candidates, CPU load; threshold re-derived (P2-02)
  for a candidate that wins
- [ ] AC6 Generation: two cheap gateway models of other vendors than the judge, on the 65 public questions with the
  same contexts as the current model; score 3, citation precision and recall, format compliance, latency, CNY per
  question, paired against `deepseek-v4-flash` (P3-07 standard records)
- [ ] AC7 Embedding: cost and time of switching (26,372 chunks, P0-12 ledger tokens), alias-switch procedure (P2-01),
  and a labelled small-index approximation with one local alternative model if it fits
- [ ] AC8 Judge: kept; what comparing judges needs (the human sheets)
- [ ] AC9 Spend per part within the caps; ledger before and after

## Out of scope
MT5 and PDF knowledge bases; holdout set; rebuilding the real index; changing any default; semantic-cache distance and
sufficiency thresholds (no model changed that feeds them); judge comparison with human labels (sheets unscored).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
