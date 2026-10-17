# P0-10 Retrieval evaluation harness

Status: review
Phase: P0
Depends on: P0-01 (data classes, KB config), P0-03 (eval questions, doc_id scheme). Independent of the retriever (P0-07/08): tested with fakes.
Design sections: s14-1, s14-2, s14-4, s14-5, s14-6, s10-1 (also s9-1 per-route results, s5-4 doc_id = page, s16-2 evals/ layout)
Calls real models: no

## Goal
Any retriever that implements a small protocol can be evaluated on a labelled question set: Hit@k, MRR, nDCG@k
(ideal DCG capped at k) and Recall@k at page and chunk level, on the final list and on each route; every run writes
a reproducible record; two runs are compared with exact McNemar and paired permutation tests; a harness measures
rerank latency and nDCG for rerank_n 10, 20, 30.

## Interface
- `rag.types.RouteHit(chunk_id, doc_id, rank, score=None)`; `RetrievalResult.routes: dict[str, list[RouteHit]]`
  (route name, e.g. `bm25`/`vector` -> that route's own ranked list; for eval and trace, left out of the JSON sent to callers).
- `rag.eval.retrieval.Retriever` protocol: `retrieve(question: str, kb: str, top_k: int) -> RetrievalResult`.
  Final list = `result.chunks` in list order (ties must already be broken by the retriever); routes = `result.routes`.
- `rag.eval.questions`: `LabeledQuestion(qid, question, answer, relevant_pages, relevant_chunks=[])` (a page is a
  doc_id, s5-4), `QuestionSet(path, sha256, questions)`, `load_question_set(path)`, `save_question_set(qs, path)`,
  `labels_from_hf_eval(questions)`. Committed set `evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl` (65 questions).
- `rag.eval.retrieval.Reranker` protocol: `(question, candidates: Sequence[Chunk]) -> Sequence[Chunk]`;
  `rag.testing.FakeRetriever(results)` (scripted results or exceptions per question).
- `rag.eval.stats`: `mcnemar_exact(improved, worsened) -> float`, `paired_permutation_test(a, b, *, iterations=10000,
  seed=0) -> PermutationResult`, `wilson_interval(successes, n, *, confidence=0.95)`, `paired_changes(a, b) ->
  PairedChanges`, `min_detectable_difference(n, *, alpha=0.05, worsened=0) -> MinDetectable | None`.
- `rag.eval.retrieval.metrics`: `hit_at_k`, `mrr_at_k`, `ndcg_at_k`, `recall_at_k(ranked, relevant, k)`,
  `compute_metrics(ranked, relevant, ks) -> dict[str, float] | None`.
- `rag.eval.retrieval`: `run_retrieval_eval(retriever, question_set, *, kb_config, top_k=None, ks=(1, 3, 5, 10),
  label, provenance, clock, now) -> RunRecord`; `collect_provenance(*, settings, models, repo_root) -> Provenance`, `make_run_id(kb, label)`, `write_run(record, *, public,
  reports_root="evals/reports") -> Path` (`public/<run_id>/` or git-ignored `private/<run_id>/`: `results.json`,
  `report.md`); `load_run(path_or_id)`; `compare_runs(a, b, ...) -> Comparison`; `rerank_latency_curve(retriever,
  reranker, question_set, *, kb, rerank_ns=(10, 20, 30), final_k, ks, clock) -> RerankCurve`, `write_rerank_curve`.
- CLI `uv run python -m rag.eval.retrieval compare <run_a> <run_b> [--list final] [--level page] [--metric mrr@10]
  [--iterations N] [--seed S] [--alpha A] [--out FILE]` and `... export-hf [--manifest P] [--out P] [--no-download]`.

## Acceptance criteria
- [x] AC1 Per-route hits ride on `RetrievalResult` and stay out of its JSON — `tests/unit/test_types.py::test_retrieval_result_routes_kept_out_of_json`
- [x] AC2 Question sets round-trip through JSONL with a file sha256; duplicate qids, malformed lines and pages of another kb are rejected; HF eval questions become page labels — `tests/unit/test_eval_questions.py::test_question_set_roundtrip_and_hash`, `::test_question_set_rejects_bad_input`, `::test_labels_from_hf_eval_questions`, `::test_export_hf_cli_writes_question_set`
- [x] AC3 The committed HF set has 65 unique questions, each with one `hf_docs:` page label — `tests/unit/test_eval_questions.py::test_committed_hf_question_set`
- [x] AC4 Hit@k, MRR, Recall@k, nDCG@k match hand-computed values — `tests/unit/test_eval_metrics.py::test_metrics_hand_computed`
- [x] AC5 Ideal DCG is capped at k: with more relevant items than k a perfect list scores 1 (uncapped would not) — `::test_ndcg_ideal_dcg_capped_at_k`
- [x] AC6 A relevant page repeated in the list counts once; equal scores never reorder the list — `::test_repeated_page_counts_once_and_ties_keep_list_order`
- [x] AC7 Empty results score 0; a question without labels at a level has no metrics (excluded, n reported) — `::test_empty_results_and_unlabelled_questions`
- [x] AC8 Exact McNemar p: 2 better/0 worse = 0.5, 5/1 = 0.21875, 0/0 = 1, symmetric — `tests/unit/test_eval_stats.py::test_mcnemar_exact_known_tables`
- [x] AC9 Sign-flip permutation test: exact enumeration when 2^changed <= iterations, else seeded Monte Carlo (same seed, same p) — `::test_permutation_exact_for_few_changes`, `::test_permutation_monte_carlo_is_seeded`
- [x] AC10 Wilson interval on known values — `::test_wilson_interval_known_values`
- [x] AC11 Improved/worsened/unchanged counts; minimum detectable difference for 65 questions = 6 questions (9.2 points) — `::test_paired_changes_counts`, `::test_min_detectable_difference`
- [x] AC12 A run scores the final list and every route at page and chunk level, records per-hit route ranks and latency; a retriever error is recorded and scored 0 — `tests/unit/test_eval_run.py::test_run_scores_final_list_and_routes`, `::test_run_records_retriever_errors`
- [x] AC13 A run writes `results.json` and `report.md` with git commit, KB config snapshot and hash, index version, data generation, question-set hash, model ids, lock hash; private runs go to the git-ignored dir; an existing run dir is refused — `::test_write_run_record_and_provenance`, `::test_private_runs_go_to_ignored_dir`
- [x] AC14 Compare: McNemar for Hit@k, permutation for MRR/nDCG/Recall, changed questions with their top hits; CLI prints both and resolves run ids — `tests/unit/test_eval_compare.py::test_compare_runs_statistics`, `::test_compare_cli_prints_tests_and_changed_questions`
- [x] AC15 Rerank curve: the reranker gets the first n fused candidates for each n, latency and nDCG per n, Markdown table — `tests/unit/test_eval_rerank_curve.py::test_rerank_curve_measures_each_n`

## Out of scope
The real retriever and reranker (P0-07, P0-08); the full 65-question run and the real latency curve (P0-12); answer,
refusal and judge metrics, synthetic questions, dev/held-out splits (P1); CI smoke gate (s14-5, later).

Also (checked by running, not by a test): `export-hf` from the pinned eval parquet wrote the 65 questions
(sha256 `f895405b…`); `python -m rag.data.hf verify` reports 0 missing eval sources. 132 unit tests pass.

## Handoff (filled at the end)
- Files: `src/rag/types.py` (`RouteHit`, `RetrievalResult.routes`), `src/rag/eval/{__init__,stats,questions}.py`, `src/rag/eval/retrieval/{__init__,__main__,cli,protocol,metrics,record,runner,report,compare,rerank_curve}.py`, `src/rag/testing/{__init__,retrieval}.py`, `evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl`, `tests/unit/test_eval_{stats,metrics,questions,run,compare,rerank_curve}.py`, `tests/unit/test_types.py`, `README.md`, `docs/decisions.md`.
- Decisions (docs/decisions.md): code in `src/rag/eval/`, data and reports in `evals/`; per-route lists as a typed `RetrievalResult.routes` excluded from caller JSON; page = doc_id, k counts list positions, a page repeated in the list counts once, MRR cut at max(k); final list scored in list order (P0-07 must break score ties deterministically); permutation test exact when 2^changed <= iterations, else seeded; minimum detectable difference = fewest improved questions with exact McNemar p <= 0.05 (65 questions: 6, 9.2 points); reports under `evals/reports/public|private/<run_id>/`, `public` passed explicitly; retriever errors scored 0 and listed; HF set committed with stripped whitespace and without `context`; rerank curve interleaves n per question and times only the reranker call after one warm-up call.
- Open issues: no `run` CLI yet; P0-12 builds the retriever, calls `run_retrieval_eval` and `write_run(public=is_public_kb_file(...))`, and passes model versions (e.g. the reranker snapshot revision) via `collect_provenance(models=...)`. P0-07 must fill `RetrievalResult.routes` and break RRF ties deterministically. Rebase on main conflicts only in `docs/decisions.md` (both specs append at the end).
- Review rounds: pending.
