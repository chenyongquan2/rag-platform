# EVAL-1 Judge calibration against human scores

Status: review
Phase: EVAL (eval upgrade plan, step 1)
Depends on: P1-04 (answer judge, 50-row sheet), P3-05 (ask_docs comparison, 30-row sheet)
Design sections: s14-3, s14-4
Calls real models: no (0 CNY; the blind pre-score is done by a separate AI session, not by the platform's judge)

## Goal
Measure the judge against human scores as the plan asks: the share of answers a human scored 3
that the judge also scored 3 (TPR, "keeps good"), the share a human scored 1 or 2 that the judge
also scored 1 or 2 (TNR, "catches bad"), kappa, Wilson 95% intervals, the list of disagreements,
and a check against the gate (TPR >= 0.85, TNR >= 0.85, kappa >= 0.7). Produce the blind 30-row
sheet of the P3-05 two-system answers (topped up to at least 10 answers the judge scored 1 or 2)
and a blind pre-score for it. This spec never marks the calibration as passed: the pre-score is an
AI judgement and the owner still has to confirm the rows.

## Interface
- `rag.eval.calibration`: `CalibrationItem(label, human, judge)`, `BinaryAgreement` (n, human_full,
  human_not_full, `tpr`/`tnr`/`exact` as `Proportion`, `kappa`, `weighted_kappa`, `disagreements`),
  `calibrate(items)`, `GateCheck`, `check_gate(report, ...)`, `render_calibration(report, gate, source)`.
- `rag.eval.agentic.human`: `export_human_sheet(..., min_non_full=0, judge_run=1)` tops the sheet up with rows
  whose judge score (first pass) is 1 or 2 until the sheet holds that many such answers; the key
  gets an `origin` column (`differ`, `random`, `top_up`). `agentic_agreement(sheet, key, lines, run=1)`,
  `prescore_items(sheet)` and `fill_prescores(sheet, scores, out)`.
- CLI `python -m rag.eval.agentic human-sheet [--min-non-full 10] [--out-dir D]` and
  `human-agreement --out DIR --sheet CSV [--key CSV] [--run 1] [--source human|ai-prescore] [--md FILE]`.
- `rag.eval.answer` `human-agreement` also prints the binary agreement (`AgreementReport.binary`);
  CLI `rejudge-sheet` and `rejudge-agreement`. `Proportion` moved to `rag.eval.stats` (re-imported by `rag.eval.answer.record`) to avoid an import cycle.
- `rag.eval.answer.rejudge`: `export_rejudge_sheet(sheet, key, judge, n=15, seed=0, out_dir)` (blind
  re-score sheet: no AI comments, no earlier scores; rows where the earlier human score differs
  from the judge always included) and `rejudge_agreement(rejudge_sheet, rejudge_key, confirmed_sheet)`.

## Acceptance criteria
- [x] AC1 TPR, TNR with Wilson intervals, kappa and the disagreement list from (judge, human) pairs; TNR is "not estimable" with no non-full human score — `tests/unit/test_eval_calibration.py::test_calibrate_rates_kappa_and_disagreements`
- [x] AC2 Gate: pass needs all three; any miss is fail; fewer than 10 non-full (or full) human answers makes that rate "not estimable" and the verdict "not estimable" unless something already failed — `test_gate_pass_fail_not_estimable`
- [x] AC3 The 30-row export tops up to `min_non_full` judge-scored 1/2 answers, stays blind (no system, no score), records `origin` in the key, is deterministic, and says so when it cannot reach the number — `tests/unit/test_eval_agentic_human.py::test_human_sheet_tops_up_non_full_answers`
- [x] AC4 Agreement of a filled two-answer sheet with the judge lines (first pass), blank rows skipped, report names the source (human or AI pre-score) — `test_agentic_agreement_from_filled_sheet`
- [x] AC5 Pre-score helpers: the items hold only question, reference and one answer under opaque ids; filled scores come back into the sheet and comments start with `AI：` — `test_prescore_items_are_blind_and_fill_round_trips`
- [x] AC6 CLI `human-agreement` prints TPR/TNR/kappa and the gate and writes the Markdown — `test_cli_human_agreement`
- [x] AC7 P1-04 `human-agreement` reports the binary agreement too — `tests/unit/test_eval_answer_agreement.py::test_human_agreement_reports_binary_rates`
- [x] AC8 Re-score export: 15 rows, blind, includes every row where human and judge differed, the key maps new ids to old ids; the comparison counts changed scores — `tests/unit/test_eval_answer_rejudge.py`

## Out of scope
Step 1b (judge prompt v2), the 100-row version of the calibration, the 40-row citation sheet's TPR/TNR,
any real-model call.

## Handoff (filled at the end)
- Files: `src/rag/eval/calibration.py` (new), `src/rag/eval/agentic/human.py`, `src/rag/eval/agentic/cli.py`, `src/rag/eval/answer/{agreement,rejudge,cli,record}.py`, `src/rag/eval/stats.py`; tests `test_eval_calibration.py`, `test_eval_agentic_human.py`, `test_eval_answer_rejudge.py`, `test_eval_answer_agreement.py`; report `docs/reports/human-agreement.md` section 5.
- Decisions: the sheet is the P3-05 one (judge pass 1), topped up to 13 judge-scored 1/2 answers (all that P3-05 had); selection by the judge's own scores biases TNR upward, so the report also splits by `origin` (only `random` rows are free of it); gate needs at least 10 answers per class, else "not estimable"; the pre-score is an AI's, never counted as passed.
- Open issues: owner has not scored the 32-row sheet; no 100-row version; the 40-row citation sheet has no TPR/TNR.
- Review rounds:
