# EVAL-0 Deadline-truncated answers and retry of failed questions (evaluation tool only)

Status: done
Phase: P3 (evaluation upgrade, step 0)
Depends on: P3-05 (harness), P3-07 (records that show both defects)
Design sections: s13-5, s14-4
Calls real models: no (cost 0; no `-m llm` test)

## Goal
The agentic evaluation tool tells apart "the model said it found nothing" from "the answer was cut off by
the 30 s total deadline", and `--retry-errors` re-runs every question that failed for a reason that is
not the question's own (an error, a model timeout, a deadline cut). Agent code (`compose_result` in
`rag.agent`) is **not** changed, so new records stay comparable with P3-05 and P3-07; old records are
classified by the same rule at read time (nothing is re-run to get the new numbers).

Why (P3-07): `two_hop:04` and `two_hop:14` hit the 30 s deadline before the answer step ran; the stored
record is `status = not_found` with no `generation_error`, so the tool counted them as clean refusals.
Six questions failed with `generation_error = llm_timeout` (4 standard, 2 agentic) and `--retry-errors`
did not re-run them because their status is not `error`.

Option B (make `compose_result` set `generation_error` itself) is a product fix and out of scope here.

## Rule (read time, in `rag.eval.agentic.records`)
- `is_deadline_truncated(record)`: system is `agentic`, `stop_reason == "deadline"`, `status == "not_found"`,
  and `"answer" not in steps_ms` (the answer step never ran). A `deadline` stop with an `answer` step is
  a real refusal and stays one; an answered, clarification or error record is never truncated.
- `retry_reason(record)`: `"error"` (status `error`), else `"generation_error"` (non-empty
  `generation_error`), else `"deadline_truncated"`, else `None`. `is_retryable` is `retry_reason is not None`.

## Interface
- `rag.eval.agentic.records.is_deadline_truncated(record) -> bool`, `retry_reason(record) -> str | None`,
  `is_retryable(record) -> bool`.
- `--retry-errors` of `run` re-runs the records with `is_retryable` (help text updated).
- `rag.eval.agentic.analysis`: a truncated record is not a clean refusal (`clean_refusal` in
  `per_type_summary` and in `runs.compare_runs`); `TypeSummary.truncated: dict[str, int]` per system, for
  every question type; the no-answer `refusal` (no answer given) is unchanged; an answerable truncated
  question keeps its judge score 1 (no answer) and is counted separately.
- `rag.eval.agentic.report`: new section "Answers cut by the deadline" listing every truncated question
  (qid, type, seconds, rounds); the no-answer section prints the truncated counts.
- `rag.eval.agentic.runs`: `RunCostRow.truncated`; compare-runs prints truncated and retryable counts.
- `rag.eval.agentic.judging`: a judge line written by rule (`no_answer`) for a record that has since been
  answered (re-run by `--retry-errors`) is judged again; otherwise the stale score 1 would stay.
- CLI `python -m rag.eval.agentic classify --out DIR`: no model; prints, per system, the retryable
  questions by reason and the deadline-truncated ones.

## Acceptance criteria
- [x] AC1 deadline-truncated rule: truncated only without an `answer` step, only agentic, only `not_found`
  (a `deadline` stop with an answer step, an answered record, a clarification are not) — `tests/unit/test_eval_agentic_truncation.py::test_is_deadline_truncated_rule`
- [x] AC2 `retry_reason` / `is_retryable` for error, generation error, truncated, clean refusal, answered —
  `test_retry_reason_and_is_retryable`
- [x] AC3 a truncated no-answer record is a refusal but not a clean refusal; counted in `truncated` and not in
  `errors` — `test_truncated_is_not_a_clean_refusal`
- [x] AC4 a truncated answerable question stays unanswered (score 1) and is counted in `truncated`, and
  the pooled `answerable` row has it too — `test_truncated_answerable_is_counted_separately`
- [x] AC5 report lists truncated questions in their own section — `test_report_lists_deadline_truncated_separately`
- [x] AC6 compare-runs: truncated counts and clean refusal use the new rule — `test_compare_runs_truncated_not_clean`
- [x] AC7 `--retry-errors` re-runs a record with a non-empty `generation_error` (standard) and a
  deadline-truncated record (agentic) and leaves a clean refusal and an answered record alone; without the
  flag nothing is re-run — `tests/unit/test_eval_agentic_truncation.py::test_retry_errors_reruns_generation_errors`,
  `test_retry_errors_reruns_deadline_truncated`
- [x] AC8 the judge re-judges a rule-scored line of a record that was answered by the re-run —
  `test_judge_rejudges_stale_rule_line`
- [x] AC9 `classify` command output — `tests/unit/test_eval_agentic_truncation.py::test_classify_lists_retryable_by_reason`
- [x] AC10 acceptance on the stored P3-07 records (git-ignored, read-only; not a unit test): `two_hop:04`,
  `two_hop:14` truncated, 6 questions retryable for `llm_timeout` — numbers in the handoff note.

## Out of scope
- `compose_result` / any agent code (option B; separate spec). Re-running the P3-07 questions (steps 3 and
  later of `docs/…/rag-platform-eval-upgrade-plan`). Retrying `answer_parse_failed` records.

## Handoff
- Files: `src/rag/eval/agentic/{records,analysis,runs,report,runner,judging,cli}.py`; new `tests/unit/test_eval_agentic_truncation.py` (10 tests); this spec; `docs/progress.md`.
- Decisions: option A of the plan (evaluation tool only, rule applied at read time, old records need no re-run); a truncated record is still a refusal in the "no answer given" row, not in the clean-refusal row, and a truncated answerable question keeps judge score 1; `--retry-errors` drops and re-runs error, `generation_error` and truncated records; the judge re-judges a by-rule `no_answer` line when the question has been answered since (the later line wins in `judge_scores`); extra `classify` command to show the read-time result without a model.
- P3-07 records (read-only, `C:/workspace/rag-platform-data/p3-07/p3-07`): `two_hop:04` and `two_hop:14` are `deadline_truncated`; 6 `generation_error` (standard 4: huggingface_doc_qa_eval 010, 012, 017, 018; agentic 2: colloquial:05, no_answer_hard:12), so 8 retryable in all. The 30 other `deadline` + `not_found` records (29 no-answer, `two_hop:08`) have an `answer` step and stay real refusals.
- Open issues: a judge call that fails on a re-judged stale line leaves a score-less line, so `judge_scores` falls back to the old 1 and `judge_records` treats it as done; `subset.py` (P3-06 table) still counts truncated records as refused; the product fix in `compose_result` (option B) is a separate spec.
- Review rounds: 1 (APPROVE).
