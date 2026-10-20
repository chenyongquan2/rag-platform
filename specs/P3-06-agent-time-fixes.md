# P3-06 Time and evidence fixes found by the P3-05 evaluation

Status: in progress
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
- [ ] AC1 `plan_next` table: enough, max rounds, no new query, deadline with the fixed and with the adaptive minimum, token cap — `test_plan_next_table`
- [ ] AC2 Round two without the minimum search time is not searched, stops `deadline`, no judge call, still answers from round one — `test_round_two_without_search_time_stops_with_deadline`
- [ ] AC3 Searches that do not fit are dropped, one phrasing per sub-question first — `test_searches_that_do_not_fit_are_dropped_round_robin`
- [ ] AC4 A round-two search that comes back `rerank_timeout` with no scores is no evidence: round `not_found`, no answer-model call for a no-answer question — `test_unscored_round_two_is_not_found_and_not_answered`
- [ ] AC5 Partial rerank keeps scored chunks at or above the threshold only; the judge does not count unscored chunks — `test_partial_rerank_keeps_scored_chunks_only`, `test_judge_does_not_count_unscored_chunks` (test_agent_judge.py)
- [ ] AC6 `RetrievalError` in round two: answers with `coverage_unconfirmed`, `degraded`, `retrieval_timeout` category; in round one it still raises — `test_retrieval_error_in_round_two_still_answers`, `test_retrieval_error_in_round_one_raises`
- [ ] AC7 The round minimum follows the previous round — `test_round_minimum_follows_previous_round`
- [ ] AC8 Judge skip: sub-questions with a chunk >= `judge_skip_score` need no model call; the rest are asked; off by default — `test_judge_skip_score` (test_agent_judge.py)
- [ ] AC9 `declarative: false` makes no rewrite call for a plain question — `test_declarative_off_makes_no_rewrite_call`
- [ ] AC10 `search_docs` output is identical with and without the new retrieval parameter when time is ample; the vector wait stops early when time is short — `test_rerank_reserve_changes_nothing_with_ample_time`, `test_embedding_wait_leaves_time_for_rerank` (test_retrieve_hybrid.py)
- [ ] AC11 `hf_docs` config has the new values; defaults are explicit — `test_hf_docs_agent_config`, `test_new_settings_defaults`

## Re-measure (real models, subset of the P3-05 questions)
Estimate: 5 crashed + 11 needing another round + 12 no-answer + 12 single-hop = 40 questions at
0.01 to 0.03 CNY = 0.5 to 1.0 CNY, plus the judge (two passes, ~28 answers x 0.0028) 0.08 CNY.
Stop at 1.5 CNY total or a ledger total of 28 CNY. The full 150-question rerun (about 2.4 CNY for
the old graph alone) does not fit and is a follow-up.

## Out of scope
The colloquial:05 retrieval difference (cause unknown), per-question embedding ledger lines, a
hold-out set, changing the judge prompt, streaming.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
