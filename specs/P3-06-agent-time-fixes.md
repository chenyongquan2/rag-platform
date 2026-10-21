# P3-06 Time and evidence fixes found by the P3-05 evaluation

Status: review
Phase: P3
Depends on: P3-05 (findings), P3-03 (judge, rounds), P3-02 (rewrite), P0-07 / P0-08 (retrieval chain)
Design sections: s13-4, s13-3, s10-1, s10-2, s16-8, s1-2 (15 s P95 target)
Calls real models: yes (budget for this spec: at most 1.5 CNY in total; estimate below)

## Goal
Fix the five graph problems P3-05 found (each re-verified against the code below), then re-measure
the graph on a targeted subset of the P3-05 questions with real models.

## Verified against the code
(a) `retrieve_node` (nodes.py) gives a round >= 2 search `min(search_deadline_ms, remaining - 12 s
reserve)`, often a few hundred ms; `EsHybridRetriever._embed` then waits for the query vector for
`remaining - es_reserve` (200 ms), leaving nothing for rerank; the runner refuses a batch that
does not fit, so the search ends `rerank_timeout` with no scores. `RerankThreshold` keeps every
chunk after a partial rerank (s16-8), and `judge.above_threshold` counts an unscored chunk as above
the threshold, so the judge says "enough" by the rules: P3-05 records: 54 round-two searches with
chunks and no score, 0 in round one; 45 of 50 no-answer questions reached the answer model.
(b) `retrieve_node` has no `try/except`: a `RetrievalError` ("deadline passed before the search was
sent") in round two failed the request: 5 no-answer runs at 18.0 s. The guard only checks
`remaining > reserve`, so the search may get a 1 ms deadline.
(c) `plan_next` needs `remaining > reserve (12 s) + agent_round_min_ms (8 s)`: round one must end
by 10 s; it averages 10.1 s (rewrite 2.7 + retrieve 4.4 + judge 2.7); 8 of 11 answerable questions
that needed another round were cut off. Answer step in the P3-05 records: median 2.6 s, P95 4.3 s,
max 6.0 s, so 12 s of reserve is twice what it needs.
(d) Rewrite (declarative) and judge each cost ~2.65 s and ~0.0036 CNY on every question; 63 of 63
single-hop one-round runs have exactly these two calls; gold page in round one 63 against 64
for the original question (no gain).
(e) 15 s P95 cannot be met by round one (10 s) plus the answer (3 s) without cutting (d).

## Changes (as of review round 1)
1. `agent_search_min_ms` (service setting, default 6000; measured: a whole search 4.5 s on a busy machine): a search of round >= 2 is made only when `remaining - reserve >= agent_search_min_ms`; the searches that do not fit are dropped (queries taken round-robin over the sub-questions). No search possible: `stop_reason: deadline`, straight to the answer (conditional edge), no judge call.
2. After a rerank timeout the graph keeps scored chunks only in round >= 2 (`RetrievalRound.from_result(scored_only=)`); scored chunks at or above the threshold are the evidence, none left: `not_found`. Round one keeps the unscored fused chunks for the answer (as `search_docs`); the judge never counts unscored chunks as grounds; `rerank_timeout` stays set.
3. The vector wait leaves `rerank_reserve_ms` for the rerank: a field of `HybridQuery`, an argument of `HybridSearch.retrieve` (default 0), passed by `retrieve_node` for round >= 2 only (`agent_rerank_reserve_ms`, 3000; at most half of the search's limit). `search_docs` and the retrieval evals do not pass it. `search_config_version` is unchanged.
4. `retrieve_node`: `RetrievalError` in round >= 2 is caught: `fallbacks` gets `retrieve:<category>`, `AskResult.retrieval_error` (also in `AskResponse` and `error_categories`), `degraded`, `stop_reason: retrieval_error`, the run answers from earlier rounds. Round one stays fatal.
5. `agent_answer_reserve_ms` 7000 (answer step P99 6.0 s, 6.8-7.0 s busy); `agent_answer_max_tokens` 1536 for the graph's answer model only; `plan_next` needs `remaining > reserve + max(agent_round_min_ms, 0.9 x previous round ms)`.
6. `hf_docs`: `agent.declarative: false`; `agent.judge_skip_score: 0.98` (not for questions the route flagged two_hop or multi_hop maybe/yes; not for a sub-question whose only sure chunk is also another's; must not be below `retrieval.threshold`). Old behaviour by config.
7. `agent.round2_on_not_enough` (default true = design 13.4): false sends a plain `not_enough` to "not found"; only `partly_enough` gets another round. Not tested with real models.

Not done: "merge candidates and rerank once" under time pressure (touches the retrieval chain).

## Acceptance criteria
Unit tests in `tests/unit/test_agent_time_fixes.py` unless named.
- [x] AC1 `plan_next` table: enough, max rounds, no new query, deadline with the fixed floor and with the adaptive minimum, token cap — `test_plan_next_table`, `test_plan_next_without_a_model_before_the_answer_and_at_the_token_cap`
- [x] AC2 Round two without the minimum search time is not searched, stops `deadline`, no judge call, still answers from round one — `test_round_two_without_search_time_stops_with_deadline`
- [x] AC3 Searches that do not fit are dropped, one phrasing per sub-question first — `test_searches_that_do_not_fit_are_dropped_round_robin`
- [x] AC4 A round-two search that comes back `rerank_timeout` with no scores is no evidence: round `not_found`, no answer-model call — `test_unscored_round_two_is_not_found_and_not_answered`
- [x] AC5 Partial rerank keeps scored chunks at or above the threshold only; the judge does not count unscored chunks — `test_a_partly_scored_round_two_keeps_what_the_reranker_judged`, `test_judge_does_not_count_unscored_chunks` (test_agent_judge.py)
- [x] AC6 `RetrievalError` in round two: answers with `coverage_unconfirmed`, `degraded`, `retrieval_timeout` category; in round one it still raises — `test_retrieval_error_in_round_two_still_answers`, `test_retrieval_error_in_round_one_raises`
- [x] AC7 The round minimum follows the previous round — `test_round_minimum_follows_previous_round`
- [x] AC8 Judge skip: sub-questions with a chunk >= `judge_skip_score` need no model call; the rest are asked; off by default — `test_judge_skip_score` (test_agent_judge.py)
- [x] AC9 `declarative: false` makes no rewrite call for a plain question (existing parametrized case in `test_agent_rewrite.py`, unchanged) and `hf_docs` carries it — `test_hf_docs_sets_agent_keys` (test_agent_config.py)
- [x] AC10 `search_docs` output is identical with and without the new retrieval parameter when time is ample; the vector wait stops early when time is short — `test_rerank_reserve_changes_nothing_with_ample_time`, `test_embedding_wait_leaves_time_for_rerank`, `test_rerank_reserve_is_at_most_half_of_the_time_limit` (test_retrieve_hybrid.py), `test_search_docs_output_is_identical_with_and_without_the_rerank_reserve` (test_retrieve_pipeline.py)
- [x] AC11 New settings have explicit defaults and checked ranges — `test_time_fix_settings_are_named_keys_with_checked_ranges`, `test_agent_settings_defaults_and_validation` (test_agent_config.py)
- [x] AC12 The `compare` command reports a new run beside the P3-05 records by group — `tests/unit/test_eval_agentic_subset.py`

- [x] AC13 Review round 1: round one answers after a total rerank timeout; the plain search is unchanged by the reserve (1.5 s embedding, 3 s limit); only round >= 2 passes the reserve; skip guards (flagged question, shared sure chunk, threshold check); `retrieval_error` stop reason and response field; answer token cap; `round2_on_not_enough` — `test_a_completely_unscored_round_one_is_still_answered_and_flagged`, `test_a_plain_query_waits_as_before_for_a_slow_gateway`, `test_search_docs_with_a_slow_gateway_is_the_same_as_before_p3_06`, `test_only_searches_of_round_two_and_later_ask_for_the_rerank_reserve`, `test_judge_skip_is_off_for_a_question_that_may_need_two_pages`, `test_judge_skip_needs_a_sure_chunk_that_is_not_another_sub_questions_too`, `test_judge_skip_score_must_not_be_below_the_rerank_threshold`, `test_the_response_exposes_the_retrieval_error`, `test_the_graph_caps_the_answer_call_and_other_calls_keep_their_profile`, `test_round2_on_not_enough_is_on_by_default_and_plan_next_follows_it`, plus the boundary tests for the surviving mutations

Existing tests changed because the spec changed: `test_agent_settings_defaults_and_validation` (answer reserve 6 s) and `test_hf_docs_sets_agent_keys` (the new hf_docs keys). No other existing test needed a change; the helper `_pipeline` / `_retriever` in two retrieval test files got an optional argument.

## Re-measure (real models, subset of the P3-05 questions)
Estimate: 5 crashed + 11 needing another round + 12 no-answer + 12 single-hop = 40 questions at
0.01 to 0.03 CNY = 0.5 to 1.0 CNY, plus the judge (two passes, ~28 answers x 0.0028) 0.08 CNY.
Stop at 1.5 CNY total or a ledger total of 28 CNY. The full 150-question rerun (about 2.4 CNY for
the old graph alone) does not fit and is a follow-up.

## Out of scope
The colloquial:05 retrieval difference (cause unknown), per-question embedding ledger lines, a
hold-out set, changing the judge prompt, streaming.

## Handoff
- Files: `src/rag/agent/{nodes,context,judge,state,graph,result,decisions,runtime}.py`, `src/rag/retrieve/{hybrid,search,query}.py`, `src/rag/llm/factory.py` (`max_tokens`), `src/rag/config/{settings,kb}.py`, `src/rag/types.py`, `src/rag/serve/{schema,mcp_server,service,devfake}.py`, `src/rag/testing/{agent,serve}.py`, `src/rag/observe/attributes.py`, `src/rag/eval/agentic/{subset,cli,runner}.py` (`compare`), `kb/public/hf_docs.yaml`, `README.md`, `.env.example`, tests `test_agent_time_fixes.py`, `test_eval_agentic_subset.py` and additions to `test_agent_judge.py`, `test_agent_config.py`, `test_llm.py`, `test_retrieve_{hybrid,pipeline,search}.py`; `docs/decisions.md`, `docs/reports/P3-report.md` section 4. Raw records: `data/reports/p3-06{,b}/` (copy in `C:/workspace/rag-platform-data/p3-06/`).
- Decisions: see `docs/decisions.md` (P3-06). Review round 1 corrected two over-reaches: the rerank reserve changed `search_docs` (now per call, round >= 2 only) and a rerank timeout in round one gave a false "not found" (now only round >= 2 is scored-only).
- Re-measure (40 questions picked from the P3-05 records, graph run twice, 0.98 CNY of 1.5): run with the code of commit `cf936c7`, i.e. BEFORE the review-round-1 changes; those changes were not re-measured. Crashes 5 -> 0 of 10 attempts (not seen again); no-answer questions reaching the answer model 12/12 -> 7/12 (standard 5/12); round-two searches without a rerank score 21/40 -> 5/21 and 1/38; single-hop cost 0.0121 -> 0.0057 CNY, median 12.1 -> 6.4/7.5 s; the six changes were made together, so no gain is attributable to one; before ran on a quieter machine than after. P95 of the subset got longer (20.7 -> 27 s); 15 s target missed.
- Open issues: the review-round changes need a re-measure on a quiet machine; `round2_on_not_enough` untested with models; the skip rule still falsely skips 5 of 100 answerable questions at 0.98 and unflagged two-hop questions are exposed; judge fallback to rules says "enough" for any chunk above the threshold when the model has no time; the second run had gateway trouble.
- Not verified: headline accuracy; quiet-machine latency; run-to-run variation beyond two runs.
- Review rounds: 1 (CHANGES: the two over-reaches above, S1-S5, the `round2_on_not_enough` switch, nits; all addressed).
