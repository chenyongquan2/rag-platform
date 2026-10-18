# P0-13 Retrieval tuning: fusion, reranker threads, threshold, label check

Status: review
Phase: P0
Depends on: P0-07 (RRF, BM25 query), P0-08 (reranker runner, threshold), P0-10 (compare, Wilson), P0-12 (full index, run records)
Design sections: s9-3, s10-1, s10-2, s14-2, s14-4, s15-3
Calls real models: query embeddings of the 10 new no-answer questions only (10 calls, 0.000081 CNY); everything else from the L0 cache

## Goal
Explain why fusion is worse than the vector route before rerank and fix it with a simple rule; pick the
reranker threads x slots default under 1-4 concurrent requests; calibrate the rerank threshold on answerable
and 10 hand-written no-answer questions; decide whether questions 015 and 017 are label errors. The 65 HF
questions are an external benchmark (s14-2): every variant tried is reported, the choice rule is stated,
and final confirmation waits for the P1-05 dev set.

## Interface
- `rag.retrieve.fusion.rrf_fuse(routes, *, k=60, weights=None)`: score = sum over routes w_route / (k + rank);
  weights are exact (decimal) fractions; a route with weight 0 adds nothing but keeps its trace list.
- Config: `RetrievalConfig.rrf_weights` (default `{bm25: 1.0, vector: 1.0}`; keys must be route names,
  weights >= 0, not all 0); `EsHybridRetriever` passes it to `rrf_fuse`.
- Config: `EvalConfig.questions` (file name of the default labelled set in `eval.dir`).
- `kb/public/hf_docs.yaml`: `title_boost` 1.0, `rrf_weights` equal, `threshold` 0.3 (calibrated), `eval.questions`.
- Service defaults: `reranker_torch_threads` = physical cores, so `rerank_max_concurrency` = 1 (from the concurrency test).
- Data: `evals/datasets/hf_docs/no_answer.jsonl` (10 questions, empty labels),
  `evals/datasets/hf_docs/label_overrides.jsonl` (label findings with justification; the public set is unchanged).
- Scratch scripts (not committed): fusion exploration, concurrency load, threshold calibration.

## Acceptance criteria
- [x] AC1 Weighted RRF matches hand-computed values; default weights equal plain RRF — `tests/unit/test_retrieve_fusion.py::test_weighted_rrf_hand_computed`
- [x] AC2 Weight 0 removes a route's contribution; negative weights and unknown routes are refused — `::test_weighted_rrf_zero_and_invalid_weights`
- [x] AC3 Config validates `rrf_weights` (route names, >= 0, not all 0) and the retriever fuses with them — `tests/unit/test_retrieve_hybrid.py::test_retriever_uses_rrf_weights`, `tests/unit/test_config.py::test_rrf_weights_validation`
- [x] AC4 The no-answer set loads (10 questions, no labels) and override file entries name existing qids — `tests/unit/test_eval_questions.py::test_no_answer_set_and_label_overrides`
- [x] AC5 With several *.jsonl in eval.dir, `eval.questions` names the default set; without it `run` is a usage error — `tests/unit/test_eval_cli_run.py::test_run_uses_the_configured_question_file`

Also (by running): fusion variants pre/post rerank with `compare`; concurrency P50/P95; threshold sweep with
Wilson intervals; final eval with the chosen settings; report section 「P0-13 调优」.

## Out of scope
Dev set (P1-05); stop-word analyzer or reindexing; ONNX int8; editing the public question set.

## Handoff (filled at the end)
- Files: `src/rag/retrieve/{fusion,hybrid}.py` (weighted RRF), `src/rag/config/{kb,settings}.py` (`rrf_weights`, `RRF_ROUTES`, `EvalConfig.questions`, reranker thread default), `src/rag/eval/retrieval/cli.py` (`eval.questions`), `kb/public/hf_docs.yaml`, `evals/datasets/hf_docs/{no_answer,label_overrides}.jsonl`, `tests/unit/test_{retrieve_fusion,retrieve_hybrid,config,eval_cli_run,eval_questions,retrieve_rerank}.py`, `README.md`, `docs/{decisions,plan}.md`, `docs/reports/P0-report.md` (section 11). Raw records in git-ignored `data/reports/p0-13/` (eval runs, compare outputs, load test, threshold scores); scratch scripts not committed.
- Results (65 questions, page level): fusion was worse than vector because `title_path^2` with `best_fields` let short section titles matching only question words win. title_boost 1.0: no rerank Hit@1/MRR@8 0.508/0.645 -> 0.723/0.815 (Hit@1 18 better 4 worse, p 0.004); rerank 10 0.769/0.813 -> 0.831/0.896, Hit@5 0.877 -> 0.985 (7/0, p 0.0156), Hit@1 and MRR no reliable difference. Vector weights 1.5-3, BM25 depth caps, most_fields and stop-word removal tried, not better once the title fix is in (all in docs/decisions.md and the report). Load test (rerank_n 10, two repeats, 32-58% CPU used by other processes): P95 at 1 request 4x2 5.8-6.2 s, 8x1 3.4-4.3 s, 8x2 3.4-3.7 s; at 2 requests all 5.8-7.1 s; at 4 all hit the 10 s deadline (timeouts 19-23 / 12-13 / 18 of 40). Threshold: 0/65 false refusals for any threshold <= 0.88, no-answer refusal 4/10 at 0.3 [17%, 69%], 7/10 at 0.5; kept 0.3. Final run: Hit@1 0.831, Hit@5 0.985, MRR@8 0.896, nDCG@8 0.919, P95 4.0 s, 0 timeouts; 64 questions without 015: Hit@5 1.000.
- Decisions (docs/decisions.md): title_boost 1.0 for hf_docs (code default stays 2.0 until P1-05); equal RRF weights; 8 x 1 reranker default; threshold 0.3 by "no false refusal, no correct page dropped, 0.2 below the weakest correct page (0.519)"; 015 label error (licence header shared by 742 pages, page skipped as title_only), 017 retrieval failure fixed by the title boost; overrides in a file, public set unchanged; `eval.questions`.
- Open issues: every choice waits for confirmation on the P1-05 dev set; search_docs P95 3 s target not met (about 3.4 s at one request, 6.5-7.1 s at two); 10 no-answer questions give wide intervals and near-miss ones score high (answer step must refuse); `label_overrides.jsonl` is not read by the eval code; load numbers taken on a busy CPU.
- Review rounds: 0 (waiting for review).
