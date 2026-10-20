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

## Changes
1. `agent_search_min_ms` (service setting, default 4000): a search of round >= 2 is made only when
   `remaining - reserve >= agent_search_min_ms`; the searches that do not fit are dropped (the
   queries are taken round-robin over the sub-questions, so a shortage keeps one phrasing of each).
   No search possible: the node returns `stop_reason: deadline`, and the graph goes straight to the
   answer (conditional edge), no judge call.
2. A retrieval with `rerank_timeout` becomes a round of its scored chunks only
   (`RetrievalRound.from_result`): scored chunks at or above the threshold are the evidence;
   unscored ones are dropped from the round (documented: the graph has the judge and further
   rounds, `search_docs` keeps its s16-8 behaviour); none left: the round is `not_found`.
   `judge.above_threshold` no longer counts a chunk without a score.
3. `EsHybridRetriever` takes `rerank_reserve_ms` (default 0 = unchanged): the query-vector wait
   stops that much earlier (at most half of the total time limit), so rerank still has time.
   Service setting `search_rerank_reserve_ms` (default 2000) feeds it. `search_config_version` is
   unchanged (it does not include service settings).
4. `retrieve_node`: `RetrievalError` in round >= 2 (rounds exist) is caught: `fallbacks` gets
   `retrieve:<category>`, `retrieval_error` (new state and `AskResult` field) carries the category
   (`retrieval_timeout` or `retrieval_failed`) into `error_categories`, `stop_reason` is
   `deadline`, the run answers from what exists. Round one stays fatal.
5. `agent_answer_reserve_ms` default 6000 (was 12000). `plan_next` needs
   `remaining > reserve + max(agent_round_min_ms, 0.9 x previous round ms)`; the judge node
   measures the round (set by the rewrite node) and passes it in.
6. Cost of rewrite and judge, chosen by the P3-05 records (see the report, section 4):
   `agent.declarative: false` for `hf_docs` (no rewrite call for a plain question; the rewrite still
   runs for a follow-up or a two-hop question); new `agent.judge_skip_score` (None = off, `hf_docs`:
   0.98): a sub-question that has a chunk with rerank score >= it is grounded without asking the
   model; no sub-question left to ask: no model call. The old behaviour stays available by config.

Not done: "merge candidates and rerank once" under time pressure (touches the retrieval chain; the
dropping of searches by time in 1 gives the same bound on round two).

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
- Files: `src/rag/agent/{nodes,context,judge,state,graph,result,decisions}.py`, `src/rag/retrieve/{hybrid,search}.py`, `src/rag/config/{settings,kb}.py`, `src/rag/types.py` (`AskResult.retrieval_error`), `src/rag/observe/attributes.py`, `src/rag/eval/agentic/{subset,cli,runner}.py` (new `compare` command), `kb/public/hf_docs.yaml`, `README.md`, `.env.example`, tests `test_agent_time_fixes.py`, `test_eval_agentic_subset.py` and additions to `test_agent_judge.py`, `test_agent_config.py`, `test_retrieve_{hybrid,pipeline,search}.py`; `docs/decisions.md`, `docs/reports/P3-report.md` section 4. Raw records: `data/reports/p3-06{,b}/` (copy in `C:/workspace/rag-platform-data/p3-06/`).
- Decisions (all in `docs/decisions.md`): search minimum 4 s; after a rerank timeout only scored chunks count (every round, `search_docs` unchanged); the query-vector wait leaves `search_rerank_reserve_ms` (2 s, at most half the limit; 0 = old behaviour) for rerank; a later round's `RetrievalError` is noted and answered from earlier rounds (round one still raises); answer reserve 6 s and the round minimum max(8 s, 0.9 x last round); `hf_docs`: `declarative: false`, `judge_skip_score: 0.98` (chosen on the P3-05 records, not on held-out questions). Not done: merging candidates and reranking once; run-in-parallel searches.
- Re-measure (40 questions picked from the P3-05 records, graph run twice, 0.98 CNY of the 1.5 CNY budget): crashes 5 -> 0 in both runs; no-answer questions reaching the answer model 12/12 -> 7/12 (standard 5/12); round-two searches without any rerank score 21 -> 5 and 1; single-hop cost 0.0121 -> 0.0057 CNY, median 12.1 -> 6.4/7.5 s; the 11 "needs another round" questions cut by the deadline 8 -> 2 and 0; judge-3 answers on the 23 answerable ones 16 -> 19 and 18 (standard 18). P95 of the whole subset got longer (20.7 -> 27 s) and the 15 s target is still missed.
- Open issues: the machine was busy (CPU 42-64% before, median 98%/85% during the runs; rerank batch of 8 took 6.5-9.8 s), so timings are directions only; the second run had gateway trouble (3 answer-model and 14 small-model timeouts, which broke three two-hop questions); the judge fallback to rules still says "enough" for any chunk above the threshold when the model has no time (no-answer:03, :18, :19 reach the answer model that way); the `retrieval_error` path was only exercised by unit tests; the full 150-question rerun on a quiet machine is a follow-up (about 1.5 CNY plus judge).
- Not verified: headline accuracy (the subset was picked from the questions that went wrong, the skip threshold was chosen on the same records); run-to-run variation beyond two runs; quiet-machine latency.
- Review rounds: 0 (author only).
