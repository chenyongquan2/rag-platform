# EVAL-4 Regression question set, one-command comparison, pre-commit reminder

Status: in progress
Phase: eval upgrade (step 4 of `rag-platform-eval-upgrade-plan.md`, sections 10, 12, 13)
Depends on: EVAL-0 (deadline-cut classification), EVAL-2 (hard and ambiguous sets), EVAL-3 (three passes, stable questions)
Design sections: s14-4, s13-5
Calls real models: yes (estimated cost: about 0.45 CNY per regression run; the acceptance check runs it twice, about 1 CNY; the unit tests use fakes)

## Goal
A fixed set of 40 questions that `ask_docs` answered fully correctly in all three step-3 passes, stored with the
passes it was chosen from, and one command that runs the set on the current code, judges it and compares with
that baseline, so that "did my change make answers worse" has a repeatable answer. A pre-commit hook reminds
(never blocks) when a commit touches the code or configuration the set guards.

## Why the rule is the lenient one (plan section 10 versus target 9.1)
Target 9.1 says every question that goes from right to wrong must be looked at. The plan allows one accident
(a flagged question is run twice more and is a regression only when fewer than 2 of 3 passes are fully correct)
and says to return to the strict rule when few questions flip between passes. **That condition is not met**
(`docs/reports/eval-upgrade-report.md` section 3.6, 75 sampled questions, 3 passes): `ask_docs` changed result on
7 questions between pass 1 and 2 and on 6 between pass 2 and 3 (hard questions: 5 and 4 of 25); standard RAG on 2
and 0. A held-out check on the same records: of 147 (question, pass) cases where `ask_docs` was fully correct in
the other two passes, 6 (4.1%) were not correct in the held-out pass, i.e. a strict rule would flag about 1.6 of
40 questions on unchanged code each run. The lenient rule stays.

## Selection
- Source: the three step-3 passes of the 75 sampled questions (pass 1 stored, passes 2 and 3 run), read through the
  same functions as `repeat-analyze`. A question is *stable for a system* when it is correct in all 3 passes
  (judge score 3; ambiguous: asked back). Stable for `ask_docs`: single_hop 19, two_hop 8, colloquial 6, hard 12,
  ambiguous 2 (47). Stable for standard RAG: 19, 9, 7, 10, 0.
- The guarded system is `ask_docs`: every question of the set is stable for it. 40 questions: single_hop 12 (drawn
  from the 19 by `sha256(seed:qid)` rank, seed 20261023, date 2026-10-23: the single-hop questions are the ones that
  never move, so they are capped to leave room for the types that do), two_hop 8, colloquial 6, hard 12 (all of
  each), ambiguous 2 (all there are; ambiguous has too few stable questions for a share of its own).
- Each question records `stable_for` (`agentic`, and `standard` when it is also stable for standard RAG).
- Files: `evals/datasets/hf_docs/regression_v1.json` (the set: version, seed, date, source passes, counts, per
  question type, `stable_for`, a hash of question, reference and gold pages so a changed dataset is noticed) and
  `regression_v1.baseline.json` (the three passes of each question for both systems: score, correct, status, stop
  reason, time, CNY). A new version is a new file; a set is never edited after it was used.

## The comparison rule (per question, per guarded system)
1. Run the set once (attempt 1). Correct (judge 3 / asked back) = *pass*.
2. Attempts that are **run problems** carry no information and are run again, never counted as wrong: a run that
   crashed (`status` error; its message is printed, because a bug in the code looks the same), a gateway
   `generation_error` (`llm_timeout`, `llm_error`, `budget_exceeded`), a deadline cut as classified by EVAL-0, a rerank
   timeout, a gateway or helper-model timeout/error in a record that is not correct, a retrieval route that failed,
   no record (spend limit), an answered record without a judge score. A record that is not correct and was stopped by the total deadline after the answer step ran is also a run
   problem (latency, set after the first real check: unchanged code reported `hard:01`, which takes 22 to 24 s of the
   30 s, as a regression because of two deadline stops). **Not** run problems, so wrong: a reply that does not
   parse (`structured_output_failed`, `answer_parse_failed`) and `internal_error`. Limit: a change that only makes the agent slower shows up as deadline
   cuts, i.e. as run problems (exit 3, never a pass), not as a regression; the acceptance check therefore uses the
   retrieval threshold, not the answer reserve time.
3. The first clean attempt wrong: run again (at most 5 attempts per question). Among the first three clean attempts,
   2 correct = *recovered* (passes, listed as flaky); 2 wrong = *regression* (decided at once, no third attempt).
4. A question still undecided after the attempts is a *run problem*. The command prints PASS only when every question
   is decided and none is a regression; exit codes 0 pass, 1 regression, 2 refused or usage error (nothing
   decided), 3 inconclusive (run problems left).
5. A stage (one attempt of the questions still undecided) of 3 or more questions that are all run problems stops the
   command as inconclusive: the environment is broken, trying again would only repeat it. Before anything runs the
   command also refuses (exit 2) a baseline that is not the set's, a knowledge base other than the set's, fewer than
   3 `--max-attempts`, and an `--out` directory that already holds records (the runner would reuse them).
   Known gap: when the spend limit stops a stage between its two systems, that stage is not judged and its cost
   is missing from the printed cost; its questions end as run problems.
6. Before anything runs it prints the estimate (baseline CNY per question, 20% margin, and the upper bound if
   everything is rerun) and the ledger, and refuses (exit 2) when spent + the first pass would pass the stop line;
   every rerun stage checks the line again and stops as inconclusive instead of spending past it. A retrieval-only
   warm-up runs first; persistent search failures stop the command (Docker or Elasticsearch not ready is not a
   regression).

## Interface
- `src/rag/eval/agentic/regress/suite.py`: `RegressionSet`, `Baseline`, `build_suite`, `read_suite`, `write_suite`,
  `check_items`.
- `src/rag/eval/agentic/regress/verdict.py`: `Attempt`, `classify_attempt`, `decide`.
- `src/rag/eval/agentic/regress/cli.py`: commands added to `python -m rag.eval.agentic`:
  `regress-build --base DIR [--seed] [--date] [--max-single-hop 12] [--out] [--force]` and
  `regress [--set FILE] [--out DIR] [--with-standard] [--estimate-only] [--max-attempts 5] [--max-cny X]
  [--kb-override SECTION.KEY=VALUE]...` (the override is for checking the tool; the output labels such a run).
- `scripts/regression_reminder.py` and a `regression-reminder` local hook in `.pre-commit-config.yaml`
  (`verbose: true`, `pass_filenames: false`, runs only when staged files match `src/rag/agent/`, `src/rag/answer/`,
  `src/rag/retrieve/`, `src/rag/config/` or `kb/**.yaml`; always exits 0, calls no model).

## Acceptance criteria
Tests in `tests/unit/test_eval_regress_{suite,verdict,cli}.py` and `tests/unit/test_regression_reminder.py`.
- [ ] AC1 Only questions correct in every pass enter the set; a type with more stable questions than its cap is drawn by
  hash rank (same seed, same list, whatever the order; another seed, another list), a scarcer type is taken whole;
  `stable_for` marks standard RAG — `suite::test_only_stable_questions_enter`, `::test_cap_draw_is_deterministic`,
  `::test_stable_for_marks_standard`
- [ ] AC2 The baseline keeps all passes of both systems with score, correct flag, status and cost; the set rejects a
  dataset whose question text changed — `suite::test_baseline_keeps_every_pass`, `::test_changed_question_is_noticed`
- [ ] AC3 The shipped files are consistent: 40 questions, counts per type, every question in the datasets and
  matching its hash, three correct `ask_docs` passes each, `stable_for` agrees with the stored standard passes —
  `suite::test_shipped_set_is_consistent`
- [ ] AC4 Verdict: first attempt correct passes; wrong then two correct is recovered; wrong twice is a regression with no
  third attempt; wrong, correct, wrong is a regression — `verdict::test_decide_pass_recovered_regression`
- [ ] AC5 Run problems (deadline cut, model timeout, run error, rerank timeout, helper timeout in a wrong record,
  failed route, missing record, not judged) are never wrong; a correct record with a helper timeout is still
  correct — `verdict::test_run_problems_are_not_wrong`, `::test_correct_with_timeouts_is_correct`
- [ ] AC6 Problem attempts are replaced; problem then correct passes; wrong then only problems ends as a run
  problem, not as a regression — `verdict::test_problem_attempts_are_replaced`, `::test_undecided_is_a_run_problem`
- [ ] AC7 The estimate is printed first from the baseline cost, with margin and the upper bound; over the stop line the
  command refuses with exit 2 and runs nothing (no warm-up, no model); `--estimate-only` runs nothing —
  `cli::test_estimate_first_and_refusal_over_the_stop_line`, `::test_estimate_only_runs_nothing`
- [ ] AC8 All correct: one attempt, exit 0, PASS — `cli::test_all_correct_passes_in_one_attempt`
- [ ] AC9 A question that is wrong in attempts 1 and 2 is a regression (exit 1, named, attempt 3 only for the questions
  still undecided); a question wrong once and right twice is recovered (exit 0, listed as flaky) —
  `cli::test_regression_is_reported_and_only_flagged_questions_rerun`, `::test_a_single_miss_is_recovered`
- [ ] AC10 Deadline cuts and model errors never give a regression: the command ends as inconclusive (exit 3), names the
  questions, and still reports a regression found on other questions — `cli::test_run_problems_are_inconclusive_not_regression`
- [ ] AC11 A rerun stage that would pass the stop line is not run; the question stays undecided (exit 3) —
  `cli::test_rerun_stage_respects_the_stop_line`
- [ ] AC12 `--with-standard` also runs standard RAG on the questions stable for it and compares with its baseline —
  `cli::test_with_standard_guards_the_stable_standard_questions`
- [ ] AC13 `result.json` and `result.md` are written with the set version, the code revision, the overrides (labelled)
  and the cost — `cli::test_result_files_label_overrides`
- [ ] AC14 Code failures (unparsable reply, `internal_error`) are wrong, gateway errors are problems, a crashed run keeps
  its message — `verdict::test_code_failures_are_wrong_not_run_problems`
- [ ] AC15 Refused before anything runs: fewer than 3 attempts, another knowledge base, a baseline of another set or with
  missing passes, an `--out` with earlier records — `cli::test_refusals_before_anything_runs`
- [ ] AC16 `--max-cny` refuses an estimate above it and stops a rerun stage that would pass it; a warm-up that keeps
  failing stops the run as not decided (exit 2) — `cli::test_spend_limit_of_the_run`,
  `::test_warm_up_failure_stops_the_run_as_not_decided`
- [ ] AC17 `regress-build` writes the set and baseline from passes on disk, leaves out a question with a miss, and does
  not rewrite existing files without `--force`; the cap draw takes the smallest hash ranks —
  `cli::test_regress_build_writes_the_files_once`, `suite::test_cap_draw_takes_the_smallest_hash_ranks`
- [ ] AC18 The reminder prints "please run the regression command" with the command, exits 0 for any input; the hook
  pattern matches the guarded paths and not tests or docs; the hook is verbose and takes no file arguments —
  `reminder::test_reminder_text_and_exit_code`, `::test_hook_definition_matches_guarded_paths`

## Out of scope
The online sampling schedule (plan section 10, optional); making the hook blocking; a retrieval-only smoke set
(plan section 2); changing agent code; guarding the no-answer questions (not repeated in step 3).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
