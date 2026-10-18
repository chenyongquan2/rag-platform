# P1-04 Answer evaluation with an LLM judge

Status: in progress
Phase: P1
Depends on: P0-08/P0-13 (`SearchPipeline` settings), P0-10 (eval layout, stats), P1-01 (chat factory, ledger), P1-02 (answer generation), P1-03 (security set and scorer)
Design sections: s14-1, s14-3, s14-4, s13-1, s13-5, s11-2, s20-1
Calls real models: yes (estimate below; pilot of 5 questions first)

## Goal
The 65 HF questions and the 30 security questions are answered by the standard (non-agentic) path
(`SearchPipeline` -> `build_context` -> `generate_answer`), with per-question records (answer, citations,
citation check, parse outcome, latency, tokens, CNY); format compliance with Wilson intervals; an HF-style judge
on a 1–5 and a 1–3 scale, run twice for stability; a blind sheet and a script for the owner's human scores.

## Cost estimate (written before any real run)
Prices are the configured estimates (CNY per million tokens, input/output): `deepseek-v4-flash` 2/8, judge
`qwen/qwen3.6-flash` 2/8 (added, `estimated: true`; the gateway shows no prices). Ledger so far: 2.652 CNY.
Ledger evidence: P1-02 answer call 1,351 prompt / 90 completion (46 reasoning) with 3 fragments; judge probe with
thinking on spent 600 reasoning tokens on "2+2", with `enable_thinking: false` 54 / 21 tokens.
- (a) Answers: 8 fragments, 4,000 estimated tokens (estimate is ~38% high) -> ~3,000 prompt; completion ~600
  (reasoning ~400 + answer ~200), up to 1,200; +10% parse retries. 65 x 1.1 x (3,000 x 2 + 600 x 8) / 1e6 = **0.77 CNY**
  (high case 1,200 completion: 1.12).
- (b) Judge: ~700 prompt, ~150 output per call; 65 answers x 2 scales x 2 runs = 260 calls -> **0.68 CNY** (high 1.0).
- (c) Security: 30 answers, ~1,800 prompt, ~600 completion, +10% -> **0.28 CNY**; judging them (1–5 once) 0.07.
- Total expected **1.8 CNY**, high case **2.6 CNY**; projected ledger 4.5–5.3 CNY, far below the 28 CNY stop.
Pilot rule: run 5 questions (answer + both scales once), project to 65 from the pilot's real tokens; go on only if
the projected total stays below 10 CNY.

## Interface
- `rag.eval.answer`: `AnswerItem`, `AnswerQuestionRecord`, `AnswerRunRecord`, `Usage`;
  `run_answer_eval(items, contexts, *, kb_config, model, label, provenance) -> AnswerRunRecord` (async);
  `RetrieverContexts(retriever, kb, top_k)`, `PlantedContexts(sset, kb_config)`;
  `format_compliance(records) -> FormatCompliance`; `project_cost(run, questions) -> CostProjection`.
- `rag.eval.answer.judge`: `JUDGE_PROMPT_VERSION`, `JudgeConfig`, `build_judge_prompt(scale, ...)`,
  `parse_judge_output(text, scale)`, `judge_answers(run, *, model, config, scales) -> JudgeRun` (async),
  `judge_stability(a, b, scale) -> Stability`.
- `rag.eval.answer.agreement`: `weighted_kappa(a, b, *, low, high)`, `spearman(a, b)`, `export_human_sheet(run, n, seed, out_dir)`,
  `human_agreement(sheet, key, judge) -> AgreementReport`.
- CLI `uv run python -m rag.eval.answer run|judge|stability|human-sheet|human-agreement|security|project`;
  records under git-ignored `data/reports/answer-eval/`.

## Acceptance criteria
Unit tests in `tests/unit/test_eval_answer_{run,judge,agreement,cli}.py`.
- [ ] AC1 Runner: question -> contexts -> answer; record has text, cited chunk ids, citation check counts, parse outcome, latencies, per-question ledger tokens and CNY — `test_runner_records_answer_citations_usage`
- [ ] AC2 Retrieval `not_found` gives found=false with no model call; a retriever exception is recorded and the run goes on — `test_runner_not_found_and_retrieval_error`
- [ ] AC3 Format compliance: fully cited share, unknown numbers, unsupported sentences, parse-failed and found=false rates, each with a Wilson interval — `test_format_compliance_counts_and_intervals`
- [ ] AC4 Judge prompt per scale: question, response, reference marked as the top score, every score described, feedback before `[RESULT]` — `test_judge_prompt_contents`
- [ ] AC5 Judge output parse: `[RESULT] n`, tolerated variants flagged, out of range or missing -> no score — `test_parse_judge_output`
- [ ] AC6 Judge run: no-answer records get 1 by rule without a call; summary has the 1–5 mean and the 1–3 percent ((s-1)/2), judge model and prompt version — `test_judge_answers_scores_and_summary`
- [ ] AC7 Stability: flip share with Wilson, mean absolute difference, weighted kappa; rule scores left out — `test_judge_stability`
- [ ] AC8 Weighted kappa and Spearman match reference values — `test_weighted_kappa_known_values`, `test_spearman_known_values`
- [ ] AC9 Blind sheet: seeded sample, shuffled, no system or judge columns, key file apart; agreement read from a filled sheet, blank rows skipped — `test_human_sheet_is_blind`, `test_human_agreement_from_filled_sheet`
- [ ] AC10 Security answers come from the planted chunks through `generate_answer` and are scored by the P1-03 scorer — `test_security_answers_scored`
- [ ] AC11 CLI `run`, `judge`, `stability`, `human-sheet`, `project` with fake seams write their files — `test_cli_with_fakes`
- [ ] AC12 Cost projection from a pilot run — `test_project_cost_from_pilot`

Also (by running, `llm`): pilot, 65 answers, judge twice on both scales, 30 security answers; `docs/reports/P1-04-report.md`.

## Out of scope
Citation check level 2 (sentence entailment, s11-2) and faithfulness (later P1/P2 spec); agentic RAG and its
comparison (P3, s13-5); human scores themselves (the owner fills the sheet); judge calibration beyond stability.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
