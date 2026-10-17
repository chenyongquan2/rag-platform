# P0-07 Hybrid retrieval: one _msearch, RRF, text dedupe

Status: review
Phase: P0
Depends on: P0-02 (index mapping, alias), P0-05 (query embedding profile), P0-06 (chunks in ES), P0-10 (`Retriever` protocol, `RetrievalResult.routes`, tie-break rule)
Design sections: s9-1, s9-2, s9-3, s16-8 (retrieval errors), s16-1 (filter whitelist, ACL, deadline_ms); also s16-2 (deadline passed down)
Calls real models: yes, query embeddings only (one `llm` test and the 65-question smoke; spent 0.0006 CNY)

## Goal
`EsHybridRetriever` sends a BM25 and a kNN query with the same filter (knowledge base, caller's ACL groups,
whitelisted caller filters) in one `_msearch`, survives one failed route (`degraded`), fuses with RRF (k=60,
deterministic ties), and `TextDedupe` drops chunks with the same body; `HybridSearch` turns this into our
`RetrievalResult` (per-route lists, flags, timings) for the P0-10 harness and later search_docs.

## Interface
- `rag.retrieve.filters`: `CALLER_FILTER_KEYS = ("doc_type", "language", "source_rel_prefix")`;
  `build_es_filter(*, kb, acl_groups, filters=None) -> list[dict]`; `FilterError(ValueError)` (`invalid_filter`).
- `rag.retrieve.deadline.Deadline`: `after_ms(ms, clock=time.monotonic)`, `remaining_s()`, `remaining_ms()`, `expired`.
- `rag.retrieve.fusion`: `FusedHit(chunk_id, doc_id, score, route_ranks, route_scores)`; `rrf_fuse(routes: Mapping[str, Sequence[RouteHit]], *, k=60) -> list[FusedHit]`
  (ties: higher exact score, then better best single-route rank, then chunk_id ascending).
- `rag.retrieve.dedupe`: `dedupe_key(text, drop_lines=())`; `TextDedupe(BaseNodePostprocessor)` (`path_keys`).
- `rag.retrieve.hybrid`: `EsHybridRetriever(BaseRetriever)(es, embed_model, *, kb, index, config: RetrievalConfig, es_reserve_ms, es_margin_ms, embed_threads=4)`;
  `HybridQuery(QueryBundle)(query_str, acl_groups, filters, deadline, trace)`; `RetrievalTrace`; `RetrievalError(category, route_errors)`; route names `BM25`, `VECTOR`.
- `rag.retrieve.search`: `HybridSearch(retriever, *, acl_groups, deadline_ms, postprocessors=None, generation=None)` with
  `retrieve(question, kb, top_k, *, filters=None, deadline_ms=None, request_id=None) -> RetrievalResult` (P0-10 `Retriever`);
  `make_hybrid_search(cfg, settings, *, acl_groups, index_prefix="", es=None, embed_model=None, redis=None)`; `IndexEmbedModelError`; `node_to_chunk`.
- Config: `RetrievalConfig.num_candidates` (100), `.rrf_k` (60), `.title_boost` (2.0); `ServiceSettings.search_deadline_ms` (10000), `.search_es_reserve_ms` (1000), `.search_es_margin_ms` (200).
- `ErrorCategory`: `retrieval_failed`, `retrieval_timeout`, `invalid_filter`. `rag.testing.FakeMsearchClient` (scripted `_msearch`).

## Acceptance criteria
Unit tests in `tests/unit/test_retrieve_*.py` (scripted `_msearch` responses, no network); integration in `tests/integration/test_retrieve_es.py`; `llm` in `tests/llm/test_retrieve_gateway.py`.
- [x] AC1 Filter builder always adds kb and the caller's ACL groups, maps whitelisted keys, rejects unknown keys (also `acl_groups`, `kb`) and empty ACL groups — `test_retrieve_filters.py::test_filter_always_has_kb_and_acl`, `::test_filter_rejects_unknown_keys_and_cannot_drop_acl`, `::test_filter_requires_acl_groups_and_kb`
- [x] AC2 RRF scores match hand-computed values (k=60, ranks from 1); per-route ranks and scores kept — `test_retrieve_fusion.py::test_rrf_hand_computed`
- [x] AC3 Score ties break by best single-route rank, then chunk_id, independent of input order — `::test_rrf_ties_are_deterministic`
- [x] AC4 TextDedupe keeps the first of chunks whose normalized body is equal, ignoring page-path lines; removed ids go into the trace — `test_retrieve_dedupe.py::test_dedupe_keeps_first_equal_body`, `::test_dedupe_ignores_page_path_lines`
- [x] AC5 One `_msearch`: BM25 `multi_match` over content and boosted title_path, kNN with the filter inside knn, k/size = top_k_per_route, num_candidates from config, same filter on both; query vector from the embed model — `test_retrieve_hybrid.py::test_one_msearch_with_both_routes_and_same_filter`
- [x] AC6 Fused nodes carry fusion score and per-route ranks/scores; the trace holds each route's own list and the index version — `::test_retrieve_fuses_and_keeps_route_lists`
- [x] AC7 A route error or time-out (in the response or as `timed_out`) uses the other route and sets degraded — `::test_one_route_error_degrades`, `::test_one_route_timed_out_degrades`
- [x] AC8 Query embedding fails or exceeds the deadline: BM25 only, degraded — `::test_embedding_failure_runs_bm25_only`, `::test_slow_embedding_runs_bm25_only_within_deadline`
- [x] AC9 Both routes fail, transport error or time-out, or deadline already passed: `RetrievalError` with `retrieval_failed` / `retrieval_timeout` — `::test_both_routes_failed_raises`, `::test_transport_errors_raise_categorized`, `::test_expired_deadline_raises_timeout`
- [x] AC10 The remaining deadline is passed to ES (request timeout and per-search timeout); a plain `QueryBundle` without caller access is refused — `::test_deadline_passed_to_es`, `::test_plain_query_bundle_is_refused`
- [x] AC11 `HybridSearch` returns `RetrievalResult` (chunks with ids, doc ids, fusion scores, route ranks/scores, routes, degraded, timings, index version, data generation), deduped before the top_k cut, and runs in the P0-10 harness with per-route metrics — `test_retrieve_search.py::test_hybrid_search_result`, `::test_hybrid_search_in_eval_harness`
- [x] AC12 Config defaults (num_candidates 100, rrf_k 60, title_boost 2; num_candidates >= top_k_per_route; search settings); the factory refuses an index written with another embedding model — `test_retrieve_search.py::test_retrieval_config_defaults`, `::test_factory_refuses_other_embed_model`
- [x] AC13 (integration) A filter restricts both routes; ACL groups and kb are enforced on both; a malformed kNN in real ES degrades — `test_retrieve_es.py::test_filter_restricts_both_routes`, `::test_acl_and_kb_enforced_on_both_routes`, `::test_real_route_error_degrades`
- [x] AC14 (integration) Question 025 finds its page in top-10 on the fake-vector index; (llm) the same with a real query embedding after the budget check, also in the vector route — `test_retrieve_es.py::test_known_question_fake_index`, `tests/llm/test_retrieve_gateway.py::test_known_question_real_embedding`

Also (by running): the 65 questions through `run_retrieval_eval` on the 50-doc index (real query
embeddings, then the fake-vector index); numbers in the handoff. Not committed: the reports went to a scratch directory.

## Out of scope
Rerank, threshold, context expansion (P0-08); search_docs/ask_docs, auth and caller identity (P0-09); full-corpus run and `run` CLI (P0-12); per-route RRF weights (equal weights, s9-3); MMR; caches.

## Handoff (filled at the end)
- Files: `src/rag/retrieve/{__init__,deadline,filters,fusion,query,dedupe,hybrid,search}.py`, `src/rag/config/{kb,settings}.py` (`num_candidates`, `title_boost`, `rrf_k`; `search_*`), `src/rag/types.py` (3 error categories), `src/rag/testing/retrieval.py` (`FakeMsearchClient`), `kb/public/hf_docs.yaml`, `tests/unit/test_retrieve_{filters,fusion,dedupe,hybrid,search}.py` (34 tests; 232 unit in total), `tests/integration/test_retrieve_es.py` (4), `tests/llm/test_retrieve_gateway.py` (1), `README.md`, `docs/decisions.md`.
- Smoke (65 HF questions, 50-doc index, top_k 8): real vectors - 0 errors, 0 degraded, page Hit@1 = Hit@8 = 1/65 for final, BM25 and vector (only question 025 has its page among the 50 docs; rank 1 on every list), latency P50 832 ms (query embedding P50 810 ms, `_msearch` 17 ms, fuse 1.6 ms, dedupe 2.4 ms); fake-vector index - 0 errors, BM25 Hit@1 1/65, vector 0, final Hit@3 1/65 (q025 at rank 3), P50 21 ms. Spent 0.0006 CNY (65 query embeddings, one cached); ledger total 0.0504 CNY.
- Decisions (docs/decisions.md): request context in `HybridQuery` (a `QueryBundle`) and `RetrievalTrace`; caller filter whitelist with kb and ACL always first, empty ACL refused; ties by exact fraction score, best route rank, chunk_id; an `error` or `timed_out` search is a failed route, partial hits unused; deadline: embedding waits until 1 s before it in a thread pool, `_msearch` HTTP time-out = remaining, per-search ES time-out 200 ms earlier, no ES retries, default 10 s; dedupe compares `content` (no path line in it) minus lines equal to the chunk's source_rel/source_url, NFKC + casefold + whitespace; `best_fields` with title boost 2; factory checks `_meta.embed_model`; `Chunk.metadata` carries route ranks/scores, never `acl_groups`.
- Open issues: candidate counts and removed duplicates are only in `RetrievalTrace`, not in `RetrievalResult` (P0-11 should put them on the retrieval span). Abandoned query-embedding calls keep a pool thread busy until the query profile's own deadline (7 s); with the gateway down, at most `embed_max_concurrency` such calls run and later questions go BM25-only. Title boost, `num_candidates` and `best_fields` are untuned (P0-12 on the full corpus). No `run` CLI yet (P0-12 uses `make_hybrid_search` + `run_retrieval_eval`).
- Review rounds: none yet (independent review pending).
