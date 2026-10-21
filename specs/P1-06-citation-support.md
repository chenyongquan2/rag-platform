# P1-06 Citation check level 2: does the cited fragment support the sentence

Status: review
Phase: P1
Depends on: P1-02, P1-04 (P3-05 for its stored records)
Design sections: s11-2, s11-1, s11-3, s14-1, s14-3, s14-4
Calls real models: yes (judge only, on stored answers; estimate 0.35 CNY, cap 0.6 CNY; pilot projected 0.49, actual 0.534)

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
- [x] AC1 Inputs: sentences with their cited chunks, cited fragments numbered once in first-cited
  order, uncited sentences kept with no fragments, a citation whose chunk is missing gives an
  `unknown` pair; `sentences_from_text` reuses parse.py — `test_support_inputs_pairs_and_numbering`
- [x] AC2 Prompt: fragments inside escaped `doc_fragment` delimiters marked untrusted, sentences as
  JSON, the system message says fragment text is data — `test_support_prompt_wraps_fragments`
- [x] AC3 One call per answer (split above `max_pairs_per_call`), structured output by
  function calling, verdict per expected pair; missing pairs `unknown`; uncited sentences
  not_supported without a call — `test_judge_support_one_call_per_answer`
- [x] AC4 Judge failure (chat error, unparseable output) gives `unknown` pairs and an error
  category, never `supported` — `test_judge_failure_is_unknown`
- [x] AC5 An instruction planted in a fragment cannot change the verdict format: escaped, and a
  plain-text reply is `unknown` — `test_injection_in_fragment_keeps_format`
- [x] AC6 Metrics: precision over known pairs, recall and unsupported rate over known sentences,
  unknowns counted apart, Wilson intervals, per system and type — `test_support_metrics`
- [x] AC7 Paired comparison on shared questions: exact McNemar on "every sentence supported"
  — `test_paired_all_supported`
- [x] AC8 Stability: label flips over pairs known in both runs, sentence and answer flips
  — `test_support_stability`
- [x] AC9 Loaders: P1-04 found answers become items; P3-05 records without sentence citations are
  skipped with a reason — `test_loaders`
- [x] AC10 Run: appends one line per answer, resumes (skips done keys), `--retry-unknown`, stops
  at the spend cap and refuses past the ledger stop line — `test_run_resume_and_cost_cap`
- [x] AC11 Planted errors: fragment swap and number change on copies of supported sentences,
  seeded, originals untouched — `test_plant_errors`
- [x] AC12 Human sheet (changed in review): every pair doubted in either run, every planted pair
  and every hand-made near miss (`--extra`), the rest seeded random supported pairs; no cell
  outside the quoted text holds a label, reason or stratum, owner cells empty; key with stratum
  and known answer apart — `test_support_human_sheet_is_blind`
- [x] AC13 CLI with fakes: run, report, plant, human-sheet — `test_support_cli_with_fakes`
- [x] AC14 P3 records keep sentence citations and cited chunks — `test_records_keep_sentence_citations`

Also (by running, `llm`): 5-answer pilot, 64 P1-04 answers twice, 10 planted;
`docs/reports/P1-06-report.md`.

## Out of scope
New answers (no generation here); the P3-05 paired comparison on real data (its records have no
sentence citations; a new P3 run is the owner's call, see docs/decisions.md); human labels (the
owner fills the sheet); online sampling (P5-02); an NLI model instead of the judge.

## Handoff (filled at the end)
- Files: `src/rag/answer/support.py`, `src/rag/eval/support/{__init__,__main__,records,items,metrics,runner,planted,human,report,cli}.py`, `src/rag/eval/agentic/{records,runner}.py` (`answer_sentences`, `cited_chunks`), `tests/unit/test_answer_support.py` (7), `tests/unit/test_eval_support.py` (10), `README.md`, `docs/decisions.md`, `docs/reports/P1-06-report.md`. Raw lines in git-ignored `data/reports/p1-06/` (copy in `C:/workspace/rag-platform-data/p1-06/`).
- Results (P1-04, 64 answers, 148 sentences, 186 pairs; run 1 / run 2): precision 98.4% [95.4, 99.4] / 96.8%; recall 98.6% [95.2, 99.6] both; unsupported sentences 1/148 / 0/148; every sentence supported 62/64 both; 0 unknown. Stability: 4/186 pair labels (2.2%), 1/148 sentences, 0/64 answers changed, all one step (supported <-> partly <-> not_supported). Planted: 10/10 caught as not_supported (5 fragment swaps, 5 number changes). Spend 0.534 CNY (138 calls); ledger 26.016 -> 26.550.
- Decisions (docs/decisions.md): judge model not NLI; one call per answer, cited fragments only; four labels, unknown out of every denominator; forced tool call via `bind_tools` so `extra_body` travels with it; fragments escaped as in the answer prompt, sentences as JSON; uncited sentences not supported by rule; P3 records keep sentence citations; planted errors only on judged-supported sentences; sheet over-samples non-supported pairs.
- Open issues: the P3-05 paired comparison is impossible on the stored records (no sentence citations; all 300 skipped); a new P3 run is in 「待确认」 (about 2.5 CNY). Pairwise judging marks a sentence supported only by two fragments together as partly (046). Human labels not filled (sheet of 40). Swapped fragments are easy errors; harder near-miss fragments untested. Cap check with concurrency > 1 can overshoot by the calls in flight.
- TDD: tests written first and seen red (import errors) for both modules; four core test expectations were wrong about `all_supported` with an uncited sentence and fixed before the first commit. Mutation check: 8 mutations, 7 killed at first; the planted number rule survived and got a test (now killed).
- Review rounds: 0 (waiting for the reviewer).
