# P0-13 Retrieval tuning: fusion, reranker threads, threshold, label check

Status: in progress
Phase: P0
Depends on: P0-07 (RRF, BM25 query), P0-08 (reranker runner, threshold), P0-10 (compare, Wilson), P0-12 (full index, run records)
Design sections: s9-3, s10-1, s10-2, s14-2, s14-4, s15-3
Calls real models: no (question vectors are in the L0 cache; ledger checked before and after)

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
- `kb/public/hf_docs.yaml`: `title_boost`, `rrf_weights`, `threshold` set from this spec.
- Service defaults for `reranker_torch_threads` / `rerank_max_concurrency` from the concurrency test.
- Data: `evals/datasets/hf_docs/no_answer.jsonl` (10 questions, empty labels),
  `evals/datasets/hf_docs/label_overrides.jsonl` (label findings with justification; the public set is unchanged).
- Scratch scripts (not committed): fusion exploration, concurrency load, threshold calibration.

## Acceptance criteria
- [ ] AC1 Weighted RRF matches hand-computed values; default weights equal plain RRF — `tests/unit/test_retrieve_fusion.py::test_weighted_rrf_hand_computed`
- [ ] AC2 Weight 0 removes a route's contribution; negative weights and unknown routes are refused — `::test_weighted_rrf_zero_and_invalid_weights`
- [ ] AC3 Config validates `rrf_weights` (route names, >= 0, not all 0) and the retriever fuses with them — `tests/unit/test_retrieve_hybrid.py::test_retriever_uses_rrf_weights`, `tests/unit/test_config.py::test_rrf_weights_validation`
- [ ] AC4 The no-answer set loads (10 questions, no labels) and override file entries name existing qids — `tests/unit/test_eval_questions.py::test_no_answer_set_and_label_overrides`

Also (by running): fusion variants pre/post rerank with `compare`; concurrency P50/P95; threshold sweep with
Wilson intervals; final eval with the chosen settings; report section 「P0-13 调优」.

## Out of scope
Dev set (P1-05); stop-word analyzer or reindexing; ONNX int8; editing the public question set.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
