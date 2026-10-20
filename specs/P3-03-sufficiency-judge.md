# P3-03 Sufficiency judge, correction loop, judge-accuracy script, hard questions

Status: in progress
Phase: P3
Depends on: P3-01 (graph, hooks, limits), P3-02 (rewrite module, `next_round`, helper model), P2-02 (threshold, no-answer sets), P1-05 (synthetic sets)
Design sections: s13-4 (main), s13-3, s8-2, s10-2, s14-2, s14-4, s16-8, s11-1
Calls real models: yes, the `cheap` model in the judge-accuracy run (budget 2.0 CNY, pilot of 10 questions first); one `llm` test (under 0.02 CNY)

## Goal
The `judge` hook of the `ask_docs` graph decides after every round whether what was found is enough (three
tiers, rules first, cheap model second), and the graph corrects a partly or not enough result: round 2 asks
the missing sub-questions again with several phrasings (multi-query), round 3 with a hypothetical answer
passage (HyDE); three rounds without anything above the threshold end as a real `not_found` with the closest
chunks attached. A script measures how often the judge agrees with a label made from the question sets, and
40 hand-written hard questions (20 two-hop, 20 no-answer) are added to the eval data.

## Interface
- `rag.agent.judge`: `SufficiencyJudgement`, `GroundsVerdict` (structured output), `SufficiencyJudge` (the default
  `judge` hook), `judge_rounds(ctx, rounds, *, open_owners=None) -> JudgeOutcome` (also used by the script),
  `above_threshold(chunks, threshold)`.
- `rag.agent.decisions`: `JudgeOutcome(verdict: enough | partly_enough | not_enough, missing, fallback, used_model)`;
  `RewriteOutcome` gains `queries` (tuple of `SearchQuery`) and `exhausted`. A hook may still return the P3-01
  strings `sufficient` / `insufficient` (read as enough / partly enough with every sub-question missing).
- `rag.agent.state`: `SearchQuery(text, owner, kind)` (kind original | multi_query | hyde); `RetrievalRound` gains
  `query` and `below_threshold`; `AgentState` gains `search_plan`, `sufficiency`.
- `rag.agent.correction`: `corrective_queries(state, ctx, missing) -> RewriteOutcome` (behind
  `QueryRewriter.next_round`): `MultiQueryPlan`, `HydePlan`.
- `rag.agent.helper.ask_helper(ctx, schema, messages, *, timeout_ms=None, model=None)`.
- `AgentConfig`: `judge` (default false), `multi_queries` (3), `hyde` (true). Settings: `agent_judge_profile`
  (`cheap`), `agent_judge_timeout_ms`, `agent_judge_context_tokens`, `agent_round_min_ms`.
- `ask_docs(..., judge_model=None)`; `AgentContext.judge_model`.
- `AskResult`: `sufficiency`, `missing`, `nearest` (closest chunks of a not-found result); `StopReason` gains
  `no_new_query`.
- `python -m rag.eval.judge_accuracy run|report` (`rag.eval.judge_accuracy.{records,metrics,runner,cli}`).
- Data: `evals/datasets/hf_docs/two_hop.jsonl` (20), `no_answer_hard.jsonl` (20).

## Acceptance criteria
Unit tests in `tests/unit/test_agent_judge.py` (J), `test_agent_correction.py` (C), `test_agent_config.py` (G),
`test_eval_judge_accuracy.py` (E), `test_hard_questions_data.py` (D). Fakes only.
Test names below are relative to the file named in the first criterion of each group.
- [x] AC1 Three tiers from rerank scores and the model's per-sub-question answer: enough, partly enough (the missing sub-question named, by the model or by the rules), not enough; a score equal to the threshold passes — `test_agent_judge.py::test_three_tiers_from_scores_and_scripted_judge`
- [x] AC2 Every chunk below the threshold: zero model calls — `test_agent_judge.py::test_everything_below_threshold_makes_no_model_call`
- [x] AC3 Chunks of all rounds count for the same sub-question; only the open ones are judged — `test_agent_judge.py::test_later_rounds_add_to_the_same_sub_question`
- [x] AC4 Model error, timeout, garbage, a missing field, an answer that skips a sub-question, no model: the rules decide, `fallback` names why, nothing fails — `test_agent_judge.py::test_judge_failure_falls_back_to_the_rules_with_a_flag`, `::test_a_judge_without_a_model_falls_back_too`; through the graph (`judge:model_error` in `fallbacks`, degraded, answered) — `test_agent_correction.py::test_a_failing_judge_model_falls_back_to_the_rules_and_the_request_is_answered`
- [x] AC5 The context sent to the judge is capped in estimated tokens (a setting), shared between the sub-questions — `test_agent_judge.py::test_the_judge_context_is_capped_in_tokens`
- [x] AC6 Questions and fragments are untrusted blocks whose tags cannot be forged — `test_agent_judge.py::test_the_judge_prompt_treats_questions_and_fragments_as_untrusted`
- [x] AC7 `agent.judge` off keeps the P3-01 behaviour (no call, no verdict); the judge uses its own model, else the helper model; its call counts against the token cap — `test_agent_judge.py::test_judge_hook_is_off_without_the_switch_and_uses_the_judge_model_only`, `::test_judge_model_falls_back_to_the_helper_model_and_is_guarded_by_the_token_cap`, `test_agent_correction.py::test_the_judge_without_the_switch_keeps_the_p3_01_behaviour`
- [x] AC8 P3-01 string verdicts (`sufficient` / `insufficient`) are still read — `test_agent_correction.py::test_p3_01_string_verdicts_are_still_read`
- [x] AC9 Partly enough: only the missing sub-question is searched again (other phrasings); results accumulate, deduplicated by chunk id — `test_agent_correction.py::test_partly_enough_requeries_only_the_missing_sub_question_and_accumulates`
- [x] AC10 Not enough: round 2 multi-query, round 3 HyDE, in that order; three rounds below the threshold end as `not_found` with the closest chunks (`nearest`), no stop reason, no answer-model call — `test_agent_correction.py::test_not_enough_goes_multi_query_then_hyde_and_ends_as_not_found_with_nearest_chunks`
- [x] AC11 A phrasing that finds the page ends the loop; `max_rounds` is the KB's; HyDE can be switched off — `test_agent_correction.py::test_a_phrasing_that_finds_the_page_in_round_two_ends_the_loop`, `::test_max_rounds_is_the_knowledge_bases_setting`, `::test_hyde_can_be_switched_off_and_round_three_asks_for_phrasings_again`
- [x] AC12 Partly enough after the last round: answered from what was found, `stop_reason max_rounds`, the missing sub-questions listed — `test_agent_correction.py::test_partly_enough_after_the_last_round_answers_and_says_what_was_missing`
- [x] AC13 Phrasings keep identifiers whole, stay in the language, drop instruction-like or off-topic text and repeats; when none is left the run stops (`no_new_query`, `multi_query:<reason>`, degraded) — `test_agent_correction.py::test_multi_query_keeps_identifiers_and_drops_hijacked_or_translated_phrasings`, `::test_when_every_phrasing_is_refused_the_loop_stops_and_says_so`, `::test_a_phrasing_that_repeats_a_query_already_searched_is_not_searched_again`
- [x] AC14 The correction prompt marks the question untrusted — `test_agent_correction.py::test_the_correction_prompt_marks_the_question_untrusted`
- [x] AC15 A failing correction model (error, garbage, empty) stops the loop with the reason; the request is still answered — `test_agent_correction.py::test_a_failing_correction_model_stops_the_loop_and_the_request_is_still_answered`
- [x] AC16 Judge on and rewrite off: no new phrasing can be made; the run says `no_new_query`, no fallback — `test_agent_correction.py::test_corrections_need_the_rewrite_switch`
- [x] AC17 A round that cannot finish (the answer's time plus `agent_round_min_ms` not left) is not started: `deadline`, coverage unconfirmed — `test_agent_correction.py::test_a_round_that_cannot_finish_is_not_started`
- [x] AC18 The token cap stops further rounds — `test_agent_correction.py::test_the_token_cap_stops_further_rounds`
- [x] AC19 HyDE text is a search query only: not in the answer prompt, the judge prompt or the result; an off-topic passage is not searched — `test_agent_correction.py::test_hyde_text_is_a_query_only_and_never_reaches_the_answer`, `::test_a_hyde_passage_that_is_off_topic_or_an_instruction_is_not_searched`
- [x] AC20 Spans carry verdict, missing, model use, tokens, technique and queries — `test_agent_correction.py::test_spans_carry_the_verdict_the_missing_part_and_the_cost_of_the_judge`, `::test_the_round_two_rewrite_span_names_the_technique_and_the_queries`
- [x] AC21 The P3-01 fakes still pass the judge's rules — `test_agent_correction.py::test_pages_result_helper_default_still_passes_the_judge_rules`
- [x] AC22 Settings and KB keys are explicit and checked; `hf_docs.yaml` sets them — `test_agent_config.py::test_judge_and_correction_config`, `::test_hf_docs_sets_agent_keys`
- [x] AC23 Response, MCP schema and business record carry `sufficiency`, `missing`, `nearest` — `test_serve_ask_service.py::test_the_judge_result_reaches_the_response_and_the_business_record`
- [x] AC24 Label: the gold page is among the chunks round one returned (alternatives unless `requires: all`; a no-answer question never) — `test_eval_judge_accuracy.py::test_label_enough_means_the_gold_page_is_among_the_returned_chunks`, `::test_collect_labels_every_question_from_the_sets_and_records_what_the_judge_said`
- [x] AC25 Confusion matrix, accuracy, precision and recall of enough with Wilson intervals — `test_eval_judge_accuracy.py::test_confusion_accuracy_precision_recall_and_wilson_intervals`, `::test_wilson_interval_is_the_one_of_the_stats_module`
- [x] AC26 Three tiers against the label, split by question type — `test_eval_judge_accuracy.py::test_the_three_tiers_against_the_label_and_the_split_by_question_type`
- [x] AC27 Comparison with the rules alone and with fixed two rounds (wasted rounds, missed corrections, rounds run) — `test_eval_judge_accuracy.py::test_the_judge_is_compared_with_the_rules_alone_and_with_fixed_two_rounds`
- [x] AC28 `--rescue`: rounds two and three for questions that were not enough, written by the production correction code — `test_eval_judge_accuracy.py::test_the_rescue_run_asks_round_two_and_three_for_questions_that_were_not_enough`
- [x] AC29 The spend limit stops the run before the next question (partial file) — `test_eval_judge_accuracy.py::test_the_run_stops_when_the_spend_limit_is_reached`
- [x] AC30 CLI: `run` writes records and report, `report` recomputes without a model; bad arguments exit 2 with one stderr line; a set that mixes answerable and no-answer questions is refused — `test_eval_judge_accuracy.py::test_the_command_line_runs_writes_the_records_and_reports_again`, `::test_bad_arguments_exit_with_code_2_and_one_line_on_stderr`, `::test_a_set_file_that_mixes_answerable_and_no_answer_questions_is_refused`
- [x] AC31 The 20 two-hop questions load, name two different pages each (both required) and carry the evidence excerpt of each fact — `test_hard_questions_data.py::test_two_hop_set_loads_and_every_question_needs_two_different_pages`
- [x] AC32 The 20 no-answer questions load unlabelled, typed (14 near miss, 6 off topic), each says what was checked; none repeats another set — `test_hard_questions_data.py::test_no_answer_hard_set_loads_unlabelled_typed_and_says_what_was_checked`, `::test_the_hard_sets_repeat_nothing_of_the_other_sets`, `::test_unverified_premises_stay_marked`
- [x] AC33 Every gold page exists in the corpus and every evidence excerpt is in its page text (skipped without the downloaded corpus) — `test_hard_questions_data.py::test_every_gold_page_exists_and_its_evidence_is_in_its_text`
- [x] AC34 (by running) judge accuracy on the synthetic dev set, the public 65 questions and the no-answer sets with the real search chain and the real cheap model, spend recorded; section 「P3-03 够不够判断」 of `docs/reports/P3-report.md`
- [x] AC35 One `llm` test, never run by default: the real cheap model on three hand-made cases, under 0.02 CNY — `tests/llm/test_sufficiency_gateway.py::test_real_cheap_model_judges_three_cases`

## Out of scope
Switching knowledge base after a failed round (the graph serves one KB per call; documented extension point in
`correction.py`); step-back prompts (s8-2); per-caller budgets (P8); the end-to-end Agentic RAG eval (P3-05).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
