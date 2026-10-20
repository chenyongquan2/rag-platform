# P3-05 Agentic RAG against standard RAG

Status: review
Phase: P3
Depends on: P3-01, P3-02, P3-03 (the `ask_docs` graph), P1-04 (answer eval, judge), P2-02 (no-answer sets)
Design sections: s13-5, s13-4, s14-1, s14-3, s14-4, s15-1, s15-2
Calls real models: yes (budget for this spec: at most 5.0 CNY; estimate below, pilot of 10 questions first)

## Goal
An `agentic_eval` CLI that runs the same questions through (A) standard RAG (retrieval chain, then
answer, no agent: the P1-04 setup) and (B) the real `ask_docs` graph with real retrieval (ES, local
reranker) and real models, scores both with the P1-04 judge, and reports the paired comparison per
question type, the cost and latency of both, and what the judge-and-correct loop did inside the graph.

## Question sets (all `hf_docs`, 150 questions per system)
| type | file | n | scored by |
|---|---|---|---|
| single_hop | `huggingface_doc_qa_eval.jsonl` (the 65 public questions) | 65 | judge, 1-3 rubric |
| two_hop | `two_hop.jsonl` | 20 | judge, 1-3 rubric |
| colloquial | `colloquial.jsonl` (new: 15 hand-written casual rewrites of public questions) | 15 | judge, 1-3 rubric |
| no_answer | `no_answer.jsonl` (10), `no_answer_extra.jsonl` (20), `no_answer_hard.jsonl` (20) | 50 | refusal |

`colloquial.jsonl`: every 4th public question (row index 0, 4, ..., 56), rewritten by hand in casual
wording (lower case, pronouns inside the question, typos, abbreviations, three mixed Chinese/English).
Gold pages and reference answers are copied unchanged from the source question (`meta.source_qid`); no
new facts. No question was chosen or dropped by looking at either system's result.

## Metrics fixed before the run (do not change after seeing numbers)
- Judge: P1-04's, unchanged: `qwen/qwen3.6-flash`, prompt version `p1-04-hf-v1`, temperature 0, thinking
  off, the 1-3 scale (HF agent_rag). Two passes over the same answers for stability; the first pass is the
  primary score. A question with no answer (not found, parse failure, error, or a clarification instead of
  an answer) gets 1 by rule, as in P1-04.
- Binary metric for answerable questions: **correct = judge score 3** (completely correct). Secondary:
  adequate = score >= 2. Mean score is reported as the HF percentage `(score - 1) / 2`.
- Binary metric for no-answer questions: **refused = no answer given** (`found` false). A
  `need_clarification` result counts as a refusal to answer (nothing was answered) and is reported
  separately; the graph is never resumed in this harness.
- Paired tests on the same questions: exact McNemar on the binary metric, paired sign-flip permutation
  test on the score, Wilson 95% intervals on every rate, the smallest difference `n` can show
  (`min_detectable_difference`). A difference that is not significant is called "direction only".
- Pass criteria (design s13-5): single-hop B must not be worse than A; two-hop and colloquial should be
  better; no-answer refusal rate must not drop. "Worse" and "better" are judged by the paired test and
  the counts, not by the mean alone.
- Retrieval labels (no model): gold page in the chunks given to the answer step (A), in round one of B
  (after the rewrite), in all rounds of B, in B's answer context. Two-hop needs both pages.

## Cost estimate (before any real run)
Ledger so far: 17.08 CNY. P1-04: 0.0047 CNY per standard answer, judge 0.0014 per call. `ask_docs` makes
3 to 8 model calls per question (route fallback, rewrite, judge per round, correction rounds, answer);
P3-03 measured 0.004 CNY per judge round and P3-02 0.015 CNY per rewrite check.
- A: 150 x 0.0047 = 0.7 CNY at most (no-answer questions mostly stop at the threshold and cost less).
- B: guess 0.02 to 0.04 CNY per question, 150 questions: 3.0 to 6.0 CNY. **This may exceed 5.0**: the
  pilot decides; if the projection is above the budget the no-answer sets are cut (all 50 are kept only if
  the projection fits).
- Judge: 100 answerable questions x 2 systems x 2 passes x 0.0014 = 0.56 CNY.
Rule: stop when this spec's own spend reaches `--max-cny` (default 5.0) or the ledger total would pass 28.

## Interface
- `rag.eval.agentic.sets`: `EvalItem`, `load_items(kb, sets=None) -> list[EvalItem]`, `SET_FILES`.
- `rag.eval.agentic.records`: `SystemRecord` (one question, one system), `JudgeLine`, `append_record`,
  `read_records`, `done_qids` (resume).
- `rag.eval.agentic.runner`: `run_standard(items, ...)`, `run_agentic(items, ...)` (async, one question at
  a time, append after each, skip finished ones, stop at the spend limit), `TraceHooks` (records the
  rewrite, judge and route outcomes of each round).
- `rag.eval.agentic.analysis`: pure functions over records: `paired_binary`, `paired_scores`,
  `per_type_summary`, `loop_report`, `rewrite_effect`, `cost_latency`, `min_detectable_table`.
- `rag.eval.agentic.report`: `render_report(records, judge_lines) -> str`, `export_human_sheet`.
- CLI `uv run python -m rag.eval.agentic run|judge|report|human-sheet|project`;
  records under git-ignored `data/reports/p3-05/{standard,agentic}/records.jsonl`.

## Acceptance criteria
Unit tests in `tests/unit/test_eval_agentic_*.py`.
- [x] AC1 The colloquial file: 15 questions, each derived from a public question (same gold pages and reference), differs from it, every gold page exists in the corpus, file marked as hand-written — `test_colloquial_questions_derive_from_public_ones`, `test_colloquial_gold_pages_exist_in_corpus`
- [x] AC2 Loader: the four types, counts, `requires=all` for two-hop, no-answer items have no gold page, qids unique across sets, `--types` and `--take` (first N of a type) select — `test_load_items_types_and_counts`, `test_load_items_selection`
- [x] AC3 Standard run: records answer text, found, context pages, tokens and CNY from the ledger rows of that question, latency; a retrieval error is recorded and the run goes on — `test_run_standard_records`, `test_run_standard_retrieval_error`
- [x] AC4 Agentic run: records result status/found/text, rounds, stop reason, fallbacks, per-round pages, per-round judge verdicts and rewrite outcomes (hooks), model calls per profile — `test_run_agentic_records_rounds_and_hooks`
- [x] AC5 `need_clarification` is recorded as clarification and counted as no answer; the graph is not resumed — `test_agentic_clarification_is_refusal`
- [x] AC6 Resume: a second run over the same file skips finished questions and appends only the rest; the spend limit stops before the next question and records the rest as pending — `test_resume_skips_finished`, `test_spend_limit_stops_run`
- [x] AC7 Judge step: scores the answerable records of both systems twice (1-3 scale), rule score 1 without a call for no-answer records, resumes — `test_judge_lines_and_rule_scores`, `test_judge_resume`
- [x] AC8 Paired comparison per type: improved/worsened/unchanged counts, exact McNemar p, permutation p of the scores, Wilson intervals, min detectable difference — `test_paired_binary_counts_and_p`, `test_paired_scores_permutation`, `test_per_type_summary`
- [x] AC9 No-answer: refusal rate with Wilson, need_clarification counted as refusal and listed apart — `test_refusal_rate_and_clarification_split`
- [x] AC10 Loop report: rounds distribution, first-round verdicts, rounds two/three share, gold hit round one vs all rounds (rescued and lost counts), false-clarification rate on the public set — `test_loop_report_rescue_counts`, `test_false_clarification_rate`
- [x] AC11 Rewrite effect: gold page in A's context vs B's round one, paired — `test_rewrite_effect_paired`
- [x] AC12 Cost and latency: mean calls, tokens, CNY per question and type, P50/P95, deadline/token-cap/coverage-unconfirmed counts with the score of those answers — `test_cost_latency_tables`, `test_deadline_cases_listed`
- [x] AC13 Human sheet: 30 rows, answers of A and B in a hidden, seeded order, no system or judge columns, key apart, scores blank — `test_human_sheet_blind_pair`
- [x] AC14 CLI with fakes: run (both systems), judge, report, project write their files — `test_cli_with_fakes`
- [x] AC15 One real-gateway test, under 0.05 CNY: one question through both paths — `tests/llm/test_agentic_eval_gateway.py::test_one_question_both_systems`

Also (by running, `llm`): pilot of 10 questions, full run of both systems, judge twice, a probe of 8 questions with a 60 s deadline, report in `docs/reports/P3-report.md` section 3.

Actual spend: 3.80 CNY (see the Handoff).

## Out of scope
Changing the agent code (bugs found are listed with evidence for a separate spec); human scores (the owner
fills the sheet); judge prompt changes; streaming and first-text latency (P3-04); a hold-out set (this
run uses the public and hand-written sets, nothing here was tuned on them in this spec).

## Handoff (filled at the end)
- Files: `src/rag/eval/agentic/{__init__,__main__,sets,records,runner,judging,analysis,report,human,cli}.py`, `src/rag/testing/ledger_chat.py` (fake chat models that write ledger lines), `evals/datasets/hf_docs/colloquial.jsonl` and its NOTICE entry, `tests/unit/test_eval_agentic_{sets,run,judge,analysis,cli}.py`, `tests/llm/test_agentic_eval_gateway.py`, `README.md`, `docs/decisions.md`, `docs/reports/P3-report.md` (section 3). Raw records in git-ignored `data/reports/p3-05/` (copy in `C:/workspace/rag-platform-data/p3-05/`).
- Decisions: correct = judge score 3 on the 1-3 scale (adequate = 2 or 3); refusal = no answer given, a clarification counts as a refusal and is listed apart; added "clean refusal" (without runs that ended in an error) after the full run because 5 graph runs and 1 standard run crashed; clarifications are never resumed; checkpoints and history in memory for the eval; one question at a time; judge unchanged from P1-04 (two passes).
- Headline (150 questions each): single-hop 64/65 vs 64/65 correct (0 better, 0 worse); two-hop 15/20 vs 16/20 (1 better, p = 1.0, direction only); colloquial 15/15 vs 13/15 (0 better, 2 worse, p = 0.5, direction only, the opposite of the design's hope); no-answer refusal 50/50 vs 50/50 by definition, clean refusal 49/50 vs 45/50 (p = 0.22, direction only). Cost 0.0040 vs 0.0162 CNY per question, P50 5.6 vs 13.4 s, P95 7.8 vs 20.9 s (target 15 s missed; 63 of 150 graph runs over 15 s, none over 30 s). In-graph judge accuracy 94.5% (137/145). The correction loop almost never ran: 3 of 11 answerable questions that needed another round got one, 8 were cut by the deadline.
- Open issues (for the orchestrator to schedule, nothing fixed here): (1) later rounds get almost no time, the reranker times out, unscored chunks all pass the threshold and the judge falls back to "enough": 45 of 50 no-answer questions reach the answer model (standard: 25), 10x cost; (2) a later round's `RetrievalError: deadline passed before the search was sent` fails the whole request (5 no-answer questions: no_answer:09, :19, no_answer_hard:05, :06, :16) instead of answering from round one; (3) the round-two gate (remaining time > 12 s reserve + 8 s `agent_round_min_ms`) fails whenever round one takes over 10 s, a third of the questions (mean 10.1 s): 18 runs end with stop reason `deadline` at only 12-19 s; (4) the rewrite model call on 96% of questions costs 2.7 s and ~0.004 CNY with no gain in gold-page retrieval on single-hop (63 vs 64), P95 20.9 s; (5) a rewrite changed retrieval for the worse in one case (colloquial:05: the gold page albert.md is retrieved from the raw question, not from the rewrite, which kept 'ALBERT'; cause unknown; the 60 s probe answered it correctly); (6) per-question usage has no embedding ledger lines at all (not investigated).
- Not verified: human scores (sheet exported, 30 rows, not filled); every question was run once, graph run-to-run variation is not measured (the 60 s probe of 8 questions shows it is not small); no genuinely vague questions, so missed clarifications are unmeasured; the colloquial and two-hop sets are hand-written and not reviewed by the owner; first-text latency not captured; no mutation check.
- TDD: tests written first; the pre-commit hook runs the whole unit suite, so red commits are impossible and each module was committed with its tests. Red was seen before the code for sets, judge, analysis, CLI and the two later additions; for the runner tests it was shown afterwards by moving `runner.py` away (import error).
- Spend: 3.80 CNY of the 5.0 CNY budget (standard 0.595, graph 2.431, judge two passes 0.551, 60 s probe 0.216, llm test 0.010); the pilot projected 3.03 for the run plus judge, the real figure was 3.58 (+18%, cause not investigated).
- Review rounds: 1 (CHANGES, minor: report numbers and wording, tests and two small tool options; see the last section).

### Review round 1 (report, tests, tool; no new paid run)
- Judge accuracy in the graph on the 100 answerable questions is 92% (86 correct stops, 3 wrongly stopped, 5 spurious "need more", 6 correct "need more"); the 94.5% over 145 included 45 trivial no-answer rows. Only 6 of 9 first-round misses were caught; only 6 of 11 "need more" verdicts were right.
- The colloquial set had a ceiling (standard 15/15) and so did not test the "colloquial is harder for standard RAG" hypothesis.
- Standard RAG's query embeddings were warm in the cache (14 query-embedding ledger rows against 259 for the graph; new colloquial text retrieved in 4.16 s against 3.1 s): the latency ratio is about 2.0, not 2.4.
- The 60 s probe ran while the MT5 ingest ran (steps 1.2-1.5x slower); two rounds is about 31-33 s, three about 34-42 s.
- Per-question "embedding calls" is 0 because `hybrid.py` submits the query embedding to a `ThreadPoolExecutor`, which does not copy the context variable `collect_ledger_rows` uses (about 290 rows, 0.002 CNY; retrieval code not changed).
- The pilot's +18% projection gap: re-applying the pilot's per-type means (4, 2, 2, 2 questions) reproduces 3.03 CNY exactly; actual per-question cost was higher (0.0049 against 0.0032 single-hop, 0.0022 against 0.0008 no-answer for standard RAG).
- Tool changes: `--retry-errors`; the human sheet always includes the questions where the two systems' judge scores differ (4: two_hop:02, :08, colloquial:05, :07); tests for `gold_hit(requires="all")`, strict 15 s and 30 s boundaries, inclusive spend limit and ledger stop line, the embedding branch of `usage_by_profile`. Mutation check: `all` to `any` in `gold_hit` is now caught.

### Findings for P3-06 (concrete fixes proposed by the reviewer; the orchestrator will make P3-06 from this)
(a) Minimum useful search budget for rounds 2 and later: in `retrieve_node` (nodes.py ~226) skip a round-2+ search when remaining minus reserve is below a new setting `search_min_ms` (about 4-5 s, or derived from rerank per-pair time times `rerank_n`), with stop reason `deadline`.
(b) A round-2+ search that returns `rerank_timeout` with zero scores is not evidence: do not add its chunks (or mark it `not_found`); in judge.py ~89 an unscored chunk must not count as grounds (threshold.py ~64 currently keeps every chunk with `not_found=False` on a partial rerank).
(c) Cap the query-embedding wait in hybrid.py ~242 `_embed` (for example min(remaining - es_reserve - rerank_reserve, about 2 s)) or run BM25 in parallel.
(d) Finding 2: `retrieve_node` has no try/except. When `round_no > 1` and rounds already exist, catch `RetrievalError`, mark degraded with category `retrieval_timeout`, stop reason `deadline`, and still answer; round 1 stays fatal. The 18.00 s pattern comes from the guard at nodes.py ~238 checking only remaining > reserve.
(e) Finding 3: the gate at nodes.py ~301 (remaining <= reserve 12 s + `agent_round_min_ms` 8 s) needs round one done by 10 s while round one averages 10.1 s and the answer step is median 2.6 s, P95 4.3 s, max 6.0 s. Set `agent_answer_reserve_ms` to about 6000, make the round minimum adaptive (max(`agent_round_min_ms`, 0.9 x the previous round's duration)), and under time pressure cut `multi_queries` to 1-2 or merge candidates and rerank once (round 2 is about rewrite 3 s + up to 3 sequential searches 3-4 s each + judge 2.7 s, about 15 s with a one-slot reranker).
(f) Finding 4: rewrite and judge each cost about 2.65 s and 0.0036 CNY per call; 63 of 63 single-hop one-round runs have exactly these 2 calls and no measured gain (gold hit 63 against 64). Options: `declarative: false` for hf_docs, rewrite only on a retry, skip the judge model when the top rerank scores are clearly high.
(g) The 15 s P95 target cannot be met by round one (about 10 s) plus the answer (about 3 s) without (f) and (c).
Tests to require: table tests for `plan_next`; a fake retriever returning `rerank_timeout` without scores in round 2 ends `not_found` without an answer-model call; a `RetrievalError` in round 2 still answers with `coverage_unconfirmed`, and in round 1 raises as before.
