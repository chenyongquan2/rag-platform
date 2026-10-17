# P0-08 Rerank, threshold, context expansion

Status: in progress
Phase: P0
Depends on: P0-07 (`EsHybridRetriever`, `HybridSearch`, `HybridQuery`, `Deadline`, `TextDedupe`), P0-10 (`Retriever` and `Reranker` protocols), P0-04 (reranker tokenizer), P0-11 (`ThresholdDecision` of the business record)
Design sections: s10-1, s10-2, s10-3, s9-1, s16-8 (rerank row), s7-6 (get_context after an update); also s7-2 (parent/neighbour chunks), s8-3 (one reranker for all knowledge bases)
Calls real models: no (the reranker runs locally; integration tests use the fake-vector index, no query embeddings)

## Goal
`SearchPipeline` runs the whole retrieval chain - `EsHybridRetriever` -> `TextDedupe` -> `BatchedTimeoutRerank`
(local bge-reranker-base, batches of 8, stops by the request deadline, process-wide concurrency limit) ->
`RerankThreshold` (not_found plus the best 3) -> top_k cut -> optional `ContextExpander` - and returns our
`RetrievalResult` with rerank scores kept apart from fusion scores; `get_context` returns a chunk with its
context, or the page's current content flagged `updated` when the chunk id is gone.

## Interface
- `rag.types`: `Chunk.rerank_score: float | None` (sigmoid, [0, 1]; `score` stays the fusion score), `Chunk.context: list[ContextPiece]`;
  `ContextPiece(chunk_id, chunk_index, text)`; `ThresholdDecision` (moved from `rag.observe.records`, re-exported there; adds `unscored`);
  `RetrievalResult.not_found`, `.below_threshold: list[Chunk]` (best 3 when not_found), `.rerank: list[RouteHit]` (the reranker's own list,
  score = rerank score; dropped at the search_docs boundary like `routes`), `.threshold: ThresholdDecision | None`, `.config_version`;
  `ContextResult(chunk_id, doc_id, mode, found, updated, truncated, chunks)`.
- `rag.retrieve.rerank`: `PairScorer` protocol `predict(pairs) -> Sequence[float]`; `CrossEncoderScorer.load(model, *, device, max_length)`;
  `cross_encoder_kwargs(settings)`; `load_cross_encoder(settings)` and `get_rerank_runner(settings)` (one per process);
  `RerankRunner(scorer, *, batch_size, max_concurrency)` with `rerank(question, texts, deadline=None) -> RerankRun(scores, timed_out, queue_ms, score_ms, batches)`;
  `rerank_text(title_path, body)`; `BatchedTimeoutRerank(BaseNodePostprocessor)(runner, top_n)`; `ChunkReranker(runner)` (P0-10 `Reranker`); `RerankScaleError`.
- `rag.retrieve.threshold.RerankThreshold(BaseNodePostprocessor)(threshold, keep_best=3)`.
- `rag.retrieve.context`: `ContextReader(es, *, kb, index, count_tokens)` with `get_context(chunk_id, mode, *, acl_groups, window, max_tokens, deadline)`;
  `ContextExpander(BaseNodePostprocessor)(reader, mode, window, max_tokens)`; modes `none | neighbors | parent` (expander), `chunk | neighbors | parent | page` (get_context).
- `rag.retrieve.pipeline`: `SearchPipeline(HybridSearch)` (`retrieve(question, kb, top_k, ...)` = P0-10 `Retriever`; `get_context(chunk_id, mode=None)`);
  `make_search_pipeline(cfg, settings, *, acl_groups, index_prefix="", es=None, embed_model=None, redis=None, runner=None, count_tokens=None)`;
  `search_config_version(cfg, settings, embed_model_id)`. `HybridSearch` gains `expander` (runs after the top_k cut) and `config_version`.
- Config: `RetrievalConfig.threshold` in [0, 1], `.expand` (`none`), `.expand_window` (1), `.expand_max_tokens` (4000);
  `ServiceSettings.reranker_device` (`cpu`), `.reranker_max_length` (512), `.rerank_batch_size` (8), `.rerank_max_concurrency` (logical CPUs / 2).
- `rag.testing`: `FakeCrossEncoder` (scripted scores, optional fake clock and per-pair time); `FakeElasticsearch.search` resolves aliases.

## Acceptance criteria
Unit tests in `tests/unit/test_retrieve_{rerank,threshold,context,pipeline}.py` (fake cross-encoder, scripted ES); integration in `tests/integration/test_rerank_model.py`.
- [ ] AC1 The cross-encoder gets only (question, title path + body), never other metadata — `test_retrieve_rerank.py::test_rerank_text_is_question_title_path_and_body`
- [ ] AC2 Candidates go in batches of 8 (only the first `rerank_n`); output sorted by rerank score, ties in fusion order; fusion score unchanged, rerank score separate — `::test_rerank_batches_of_eight_and_sorts_by_rerank_score`
- [ ] AC3 The deadline is checked after each batch (a batch that would not fit is not started); on timeout the scored ones come first by rerank score, the rest in fusion order, `rerank_timeout` set; an expired deadline scores nothing — `::test_rerank_stops_at_deadline_and_appends_unscored_in_fusion_order`
- [ ] AC4 A semaphore limits concurrent reranks; time waiting for it counts against the deadline; no slot before the deadline means nothing scored and `rerank_timeout` — `::test_rerank_semaphore_limits_concurrency`, `::test_rerank_queue_time_counts_against_deadline`
- [ ] AC5 Scores outside [0, 1] (not sigmoid) are refused — `::test_scores_outside_unit_interval_are_refused`
- [ ] AC6 The model is loaded once per process, offline, with explicit device, max length and sigmoid; one runner per process; settings defaults — `::test_cross_encoder_loaded_once_per_process`, `::test_rerank_settings_defaults`
- [ ] AC7 `ChunkReranker` implements the P0-10 `Reranker` protocol and runs in `rerank_latency_curve` — `::test_chunk_reranker_follows_eval_protocol`
- [ ] AC8 Threshold judges only chunks with a rerank score (score >= threshold passes) and records its decision; after a rerank timeout unscored chunks are kept and nothing is reported not found — `test_retrieve_threshold.py::test_threshold_drops_scored_below`, `::test_threshold_after_timeout_keeps_unscored`
- [ ] AC9 All below the threshold: empty list, `not_found`, the best 3 in a separate field — `::test_threshold_all_below_returns_not_found_with_top3`
- [ ] AC10 Expansion adds neighbours (doc_id + chunk_index) or the parent section from ES with the caller's ACL filter, deduped with the selected chunks and each other, within the token budget; an ES error or no time left skips it — `test_retrieve_context.py::test_expander_adds_neighbours_deduped_with_selected`, `::test_expander_parent_mode_adds_section`, `::test_expander_respects_token_budget`, `::test_expander_skips_on_es_error_or_deadline`
- [ ] AC11 `get_context` returns the chunk with its context per mode; a chunk id that no longer exists returns the page's current chunks with `updated`; an unknown page is `found=False` — `::test_get_context_modes`, `::test_get_context_falls_back_to_page_when_chunk_is_gone`
- [ ] AC12 `SearchPipeline` runs retriever -> dedupe -> rerank -> threshold -> cut -> expander and returns `RetrievalResult` with fusion and rerank scores, rerank list, threshold decision, flags, timings, index version and config version; it runs in the P0-10 eval harness — `test_retrieve_pipeline.py::test_pipeline_runs_full_chain`, `::test_pipeline_in_eval_harness`
- [ ] AC13 Not found and rerank timeout reach the result (flags, error categories, best 3) — `::test_pipeline_not_found_result`, `::test_pipeline_rerank_timeout_result`
- [ ] AC14 The factory builds the chain from the KB config (rerank_n, threshold, expansion), shares the process runner, and the config version changes with the config — `::test_make_search_pipeline_from_config`, `::test_retrieval_config_rerank_and_expand_defaults`
- [ ] AC15 (integration) Real bge-reranker-base scores equal sigmoid(logits) and lie in [0, 1]; a known relevant pair scores high, an unrelated pair low; load time and per-candidate latency for 8/20/30 candidates measured — `tests/integration/test_rerank_model.py::test_real_scores_are_sigmoid_in_unit_interval`, `::test_known_relevant_pair_high_unrelated_low`, `::test_real_model_latency`
- [ ] AC16 (integration) The pipeline with the real reranker on the 50-doc fake-vector index puts question 025's page first; expansion and the get_context fallback work on real ES — `::test_pipeline_ranks_known_page_first`, `::test_context_on_real_index`

## Out of scope
search_docs / get_context tools, response shaping and business records (P0-09); per-sub-question context budget for answers (P1-02/P1); threshold calibration and the real rerank_n curve (P0-12); multi-KB merge by rerank score (s8-3, later); ONNX int8 (evaluated or deferred, see handoff); rerank metrics and spans beyond OpenInference's automatic postprocessor spans (P0-09).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
