# EVAL-3 Repeated runs: question sample, pass orchestration, pass^3 analysis, report section

Status: done
Phase: eval upgrade (step 3 of `rag-platform-eval-upgrade-plan.md`, sections 9, 12, 13)
Depends on: EVAL-0 (deadline-cut classification, retry), EVAL-1 (judge), EVAL-2 (`hard.jsonl`, `asked_back`), P3-05 (runner, records), P3-07 (stored run = pass 1)
Design sections: s14-4, s13-5
Calls real models: no (tooling only, cost 0; unit tests use fakes, no `-m llm`). The commands it adds do call models when the owner runs them.

## Goal
Tooling so the owner can run the step-3 protocol pass by pass and read the result: a fixed-seed list of 75
questions, a command that runs one pass (cost estimate first, refusal at the stop line, warm-up, alternating
system order), an analysis (pass^k, pass@k, McNemar, adjacent-pass differences, deadline cuts, stable questions)
and a plain-Chinese section for `docs/reports/eval-upgrade-report.md`.

## Protocol (plan option B)
- Questions: 40 old answerable (single_hop 20, two_hop 12, colloquial 8, drawn by sha256 rank of `seed:qid`, so the
  draw does not depend on file order or the Python version) + all 25 hard + all 10 ambiguous. No-answer questions
  are not repeated. Committed as `evals/datasets/hf_docs/repeat_step3.json` with the seed and the date.
- Pass 1 is stored: old questions from the P3-07 run (cache cold; several `--pass1-old` dirs, a later dir wins per
  system and question, for the retry dir), new questions from the EVAL-2 step-2 run (`--pass1-new`, path only).
- Passes 2 and 3 are run: each starts with a retrieval-only warm-up (no chat model); pass 2 runs standard RAG
  first, pass 3 `ask_docs` first (even pass: standard first). Each pass is a records directory `BASE/passN`.
- Correct: judge score 3 (answerable types) or `asked_back` (ambiguous). Deadline cuts count as not correct and are
  listed apart (never as refusals).

## Interface (`src/rag/eval/agentic/repeat/`, commands added to `python -m rag.eval.agentic`)
- `sample.py`: `QuestionList`, `draw_sample`, `build_question_list`, `read_question_list`, `write_question_list`.
- `passes.py`: `system_order`, `PassSource`, `load_pass`, `estimate_pass`, `check_budget`, `warm_up`, manifest.
- `analysis.py`: `analyze` -> `RepeatAnalysis` (per system and group: pass^k, pass@k with Wilson; McNemar via
  `paired_binary`; adjacent-pass flips; gap against noise; cut/failed list; stable questions).
- `report.py`: `render_section`, `write_section` (marker-delimited section; creates the file when missing).
- `repeat-sample [--seed] [--date] [--out] [--check]`; `repeat-run --base DIR --pass N [--pass1-old DIR]...
  [--pass1-new DIR] [--estimate-only] [--no-judge] [--retry-errors] [--max-cny]`;
  `repeat-analyze --base DIR [--passes 3] [--report FILE] [--stable-out FILE]`.

## Acceptance criteria
Tests in `tests/unit/test_eval_repeat_{sample,passes,analysis,report,cli}.py`.
- [x] AC1 Sampler: same seed and same items give the same list whatever the item order; another seed gives another list; counts per type; hard and ambiguous all included; the shipped file matches a fresh draw — `sample::test_draw_is_deterministic_and_order_independent`, `::test_counts_and_seed_change`, `::test_shipped_list_matches_the_sets`
- [x] AC2 pass^k is all passes correct, pass@k is any pass correct, both with Wilson 95% intervals, per system and group; ambiguous uses `asked_back`; no-answer questions are left out — `analysis::test_pass_all_and_any_arithmetic`, `::test_ambiguous_uses_asked_back`
- [x] AC3 Paired McNemar between the systems on pass^k (a = standard, b = agentic) — `analysis::test_mcnemar_on_pass_all`
- [x] AC4 Adjacent-pass differences per system, split into to-wrong and to-right, and the same count without questions that had a cut or failed record; gap against the largest flip count gives the verdict — `analysis::test_adjacent_pass_flips`, `::test_gap_against_noise_verdict`
- [x] AC5 Deadline-cut and failed records are listed apart, are not correct, and are not refusals — `analysis::test_deadline_cut_is_listed_and_not_correct`
- [x] AC6 Questions missing a pass or a judge score are listed as incomplete and left out of the numbers; stable questions (all passes correct) are written per system, per type and for both — `analysis::test_incomplete_questions_are_left_out`, `::test_stable_questions_file`
- [x] AC7 Pass 1 is assembled from several directories (later wins) restricted to the sampled questions — `passes::test_load_pass_overlay_and_restriction`
- [x] AC8 Order: pass 2 standard first, pass 3 agentic first — `passes::test_system_order_alternates`; warm-up searches every question, repeats once when rerank timed out, refuses when timeouts remain — `passes::test_warm_up_runs_retrieval_only`, `::test_warm_up_refuses_persistent_timeouts`
- [x] AC9 Estimate: per-type mean cost of the earlier records, two-hop mean as the fallback for a type without records, judge calls, 20% margin, remaining questions only — `passes::test_estimate_uses_means_fallback_and_margin`
- [x] AC10 `repeat-run` prints the estimate before anything runs, refuses (exit 2, nothing run, no warm-up) when spent + estimate passes the stop line, `--estimate-only` runs nothing — `cli::test_run_prints_estimate_first_and_refuses_over_the_stop_line`, `::test_estimate_only_runs_nothing`
- [x] AC11 `repeat-run` with fakes: warm-up before the first system, systems in the pass's order, records and judge in `BASE/passN`, the pass number in `runs.jsonl`; a second call does nothing; a pass the spend limit stopped is reported incomplete (exit 1, no judging) — `cli::test_run_pass_with_fakes_orders_systems_and_warms_up`, `::test_run_reports_an_incomplete_pass_and_does_not_judge`
- [x] AC12 `repeat-analyze` with fake records on disk writes the report section and the stable file; the section is replaced on a second run, other sections are untouched, a missing file is created — `report::test_section_is_appended_replaced_and_created`, `cli::test_analyze_writes_report_and_stable_file`
- [x] AC13 The Chinese section states pass sources, run order, tables with intervals, the noise answer and what cannot be said — `report::test_section_content`

- [x] AC14 The report writer replaces the step-3 placeholder of the existing report (and marks its overview row), numbers the sub-headings like the other sections, keeps CRLF files CRLF, and never replaces a written section that has no marker pair (appends instead) — `report::test_section_replaces_the_placeholder_left_for_step_3`, `::test_a_written_step_3_section_without_markers_is_not_replaced`, `::test_section_keeps_the_line_endings_of_a_crlf_report`
- [x] AC15 A spend limit (`--max-cny`) cannot take the pass past the stop line; a non-positive one is refused; `repeat-sample` does not overwrite a different list without `--force`; `repeat-analyze` refuses when no question is complete — `cli::test_max_cny_cannot_take_the_pass_past_the_stop_line`, `passes::test_check_budget_refuses_a_spend_limit_that_is_already_used_up`, `cli::test_sample_does_not_overwrite_a_different_list_without_force`, `::test_analyze_refuses_when_nothing_is_complete`

## Out of scope
Running any pass (owner), the regression set and its compare rule (step 4), the 60-second rerun, changing agent code.

## Handoff
- Files: `src/rag/eval/agentic/repeat/{sample,passes,analysis,report,cli}.py`, `src/rag/testing/repeat.py` (fake pass builders), `src/rag/eval/agentic/cli.py` (registers the three commands, writes `pass` into `runs.jsonl`), `evals/datasets/hf_docs/repeat_step3.json` (seed 20261023, 2026-10-23), `tests/unit/test_eval_repeat_{sample,passes,analysis,report,cli}.py` (50 tests).
- Decisions: pass directories are `BASE/passN` in the normal records layout, so `report` and `compare-runs` work on them; pass 1 is read by path (several `--pass1-old` dirs, later wins, so the P3-07 retry dir can follow the main one) and the paths are saved in `BASE/repeat.json`; the draw ranks questions by sha256 of `seed:qid`; only questions with a decidable result in every pass for both systems enter the numbers; the noise yardstick is the plan's (gap in pass^k questions against the largest adjacent-pass flip count of either system), shown with the McNemar p value because it is strict and can disagree with it; deadline cuts, model timeouts and run errors are not correct and are listed apart (`retry_reason`), and the flip count is also given without questions that had such a record; `repeat-run` refuses before the warm-up when spent + (spend limit or estimate with 20% margin) + judge estimate passes the stop line, and never retries failed records unless `--retry-errors` is given (a failure is part of what is measured); the report section replaces the step-3 placeholder ("还没做") of `docs/reports/eval-upgrade-report.md` when there is no marker pair yet, else appends; hard and ambiguous questions are priced from their step-2 run, a type with no records at the two-hop mean.
- Open issues: nothing was run on real models (only fake records and the stored P3-07 and step-2 records, read-only, for the estimate); the three commands call private functions of `rag.eval.agentic.cli` (`_run`, `_judge`, `_classify`, `_spent_in`), covered by end-to-end tests with fakes; pass 1 comes from earlier runs without this protocol's warm-up, so its conditions differ from passes 2 and 3; the gap-against-noise rule is the plan's wording and is strict.
- Review rounds: 2 (round 1 CHANGES: `--max-cny` could pass the stop line, slash-compressed numbers and loose wording in the Chinese text, missing guards and tests; round 2 APPROVE). After the approval one change was added: the report writer replaces the placeholder section for step 3 that the step-2 report left in the file, and marks its row in the overview table; it has its own tests.
