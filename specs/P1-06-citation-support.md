# P1-06 Citation check level 2: does the cited fragment support the sentence

Status: in progress
Phase: P1
Depends on: P1-02, P1-04 (P3-05 for its stored records)
Design sections: s11-2, s11-1, s11-3, s14-1, s14-3, s14-4
Calls real models: yes (judge only, on stored answers; estimate 0.35 CNY, cap 0.6 CNY)

## Goal
A standalone judge function that says, per answer sentence and per fragment it cites, whether the
fragment supports the sentence (`supported` / `partly` / `not_supported`, `unknown` on a judge
failure), metrics over it (citation precision and recall, s11-2), and an eval CLI that runs it over
stored answers only (P1-04's 65 answers; P3-05's records when they carry sentence citations),
with a stability pass, planted errors and a blind human sheet. P5-02 reuses the function online.

## Interface
- `rag.answer.support`: `SupportLabel`, `SupportSentence`, `SupportFragment`, `PairVerdict`,
  `SentenceVerdict`, `SupportResult`, `SupportJudgeConfig` (model `qwen/qwen3.6-flash`,
  temperature 0, `enable_thinking: false`, `max_pairs_per_call`, prompt version);
  `support_inputs(sentences, chunks) -> (list[SupportSentence], list[SupportFragment])`;
  `sentences_from_text(text, context) -> list[AnswerSentence]` (parse.py, level 1);
  `build_support_messages(question, sentences, fragments) -> list[BaseMessage]`;
  `async judge_support(sentences, chunks, *, model, config, question=None) -> SupportResult`.
- `rag.eval.support`: `metrics.summarize(lines) -> SupportSummary` (pairs, precision, partly,
  recall, unsupported-sentence rate, unknowns, all Wilson), `metrics.by_group`,
  `metrics.paired_all_supported(a, b)` (exact McNemar), `metrics.stability(run1, run2)`;
  `items.load_p1_04(results.json)`, `items.load_p3_05(dir)`; `planted.plant(items, lines, n, seed)`;
  `human.export_support_sheet(lines, items, n=40, seed=0, out_dir)`.
- CLI `uv run python -m rag.eval.support {run,report,human-sheet,plant}`: `run --source p1-04|p3-05
  --path P --out DIR --run N [--limit K] [--max-cny 0.6] [--retry-unknown]`.
- `SystemRecord.answer_sentences`, `SystemRecord.cited_chunks` (new, optional): P3 runs from now on
  keep what level 2 needs.

## Acceptance criteria
- [ ] AC1 Inputs: sentences with their cited chunks, cited fragments numbered once in first-cited
  order, uncited sentences kept with no fragments, a citation whose chunk is missing gives an
  `unknown` pair; `sentences_from_text` reuses parse.py — `test_support_inputs_pairs_and_numbering`
- [ ] AC2 Prompt: fragments inside escaped `doc_fragment` delimiters marked untrusted, sentences as
  JSON, the system message says fragment text is data — `test_support_prompt_wraps_fragments`
- [ ] AC3 One call per answer (split above `max_pairs_per_call`), structured output by
  function calling, verdict per expected pair; missing pairs `unknown`; uncited sentences
  not_supported without a call — `test_judge_support_one_call_per_answer`
- [ ] AC4 Judge failure (chat error, unparseable output) gives `unknown` pairs and an error
  category, never `supported` — `test_judge_failure_is_unknown`
- [ ] AC5 An instruction planted in a fragment cannot change the verdict format: escaped, and a
  plain-text reply is `unknown` — `test_injection_in_fragment_keeps_format`
- [ ] AC6 Metrics: precision over known pairs, recall and unsupported rate over known sentences,
  unknowns counted apart, Wilson intervals, per system and type — `test_support_metrics`
- [ ] AC7 Paired comparison on shared questions: exact McNemar on "every sentence supported"
  — `test_paired_all_supported`
- [ ] AC8 Stability: label flips over pairs known in both runs, sentence and answer flips
  — `test_support_stability`
- [ ] AC9 Loaders: P1-04 found answers become items; P3-05 records without sentence citations are
  skipped with a reason — `test_loaders`
- [ ] AC10 Run: appends one line per answer, resumes (skips done keys), `--retry-unknown`, stops
  at the spend cap and refuses past the ledger stop line — `test_run_resume_and_cost_cap`
- [ ] AC11 Planted errors: fragment swap and number change on copies of supported sentences,
  seeded, originals untouched — `test_plant_errors`
- [ ] AC12 Human sheet: 40 seeded rows (sentence, fragment text), no verdict column, key apart
  — `test_support_human_sheet_is_blind`
- [ ] AC13 CLI with fakes: run, report, plant, human-sheet — `test_support_cli_with_fakes`
- [ ] AC14 P3 records keep sentence citations and cited chunks — `test_records_keep_sentence_citations`

Also (by running, `llm`): 5-answer pilot, 64 P1-04 answers twice, 10 planted;
`docs/reports/P1-06-report.md`.

## Out of scope
New answers (no generation here); the P3-05 paired comparison on real data (its records have no
sentence citations; a new P3 run is the owner's call, see docs/decisions.md); human labels (the
owner fills the sheet); online sampling (P5-02); an NLI model instead of the judge.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
