# P1-05 Synthetic questions, critique filters, dev/holdout split

Status: in progress
Phase: P1
Depends on: P0-03 (corpus), P0-04 (read + clean), P0-10 (question sets, retrieval runner), P0-13 (retrieval settings), P1-01 (chat factory, ledger), P1-04 (judge model)
Design sections: s14-2, s14-3, s14-4, s14-6
Calls real models: yes (estimate below; pilot of 20 spans first)

## Goal
A seeded pipeline turns random spans of the raw cleaned page text into HF rag_evaluation-style factoid questions,
filters them with three 1–5 critique scores and a lexical-copy check, and writes a stratified dev set and holdout
set (each >= 100 questions, page-level labels) with a manifest; a sanity retrieval eval compares them with the 65
public questions.

## Cost estimate (written before any real run)
Prices are the configured estimates (CNY per million tokens, input/output): generator `deepseek-v4-flash` 2/8
(reasoning model), critic `qwen/qwen3.6-flash` 2/8 with `enable_thinking: false` (the P1-04 judge). Ledger so far 3.502 CNY.
Ledger evidence: P1-04 answers 1,764 prompt / 148 completion (79 reasoning) per call; judge 402 prompt / 80 completion per call.
- (a) Generation: span 2,000 chars (~550 tokens) + prompt ~200 -> ~750 prompt; completion ~500 (reasoning ~400, the
  question task needs more thought than a cited answer), high case 1,000; +10% retries/parse failures.
  420 x 1.1 x (750 x 2 + 500 x 8) / 1e6 = **2.5 CNY** (high 4.4).
- (b) Critique: groundedness ~850 prompt (span + question), relevance ~200, standalone ~350; ~100 completion each.
  420 x (1,400 x 2 + 300 x 8) / 1e6 = **2.2 CNY** (high, 200 completion each: 3.2).
- (c) Question embeddings for the sanity eval: ~250 x 25 tokens x 0.5 / 1e6 = **< 0.01 CNY**.
- Total expected **4.7 CNY**, high **7.6 CNY**; projected ledger 8.2–11.1 CNY, far below the 28 CNY stop.
Pilot rule: generate and critique 20 spans; project to 420 from the pilot's real tokens; continue only if the
projected ledger total stays below 15 CNY.

## Interface
- `rag.eval.synthetic.sampler`: `Page`, `Span`, `SamplerConfig`, `load_clean_pages(cfg, settings)`,
  `stratum_of(page)`, `allocate(sizes, n)`, `span_problem(text, config)`, `sample_spans(pages, config) -> SampleResult`.
- `rag.eval.synthetic.prompts`: `GENERATION_PROMPT_VERSION`, `CRITIQUE_PROMPT_VERSION`, `CRITERIA`,
  `build_generation_prompt(span)`, `parse_generation(text)`, `build_critique_prompt(criterion, question, span)`, `parse_critique(text)`.
- `rag.eval.synthetic.filters`: `FilterConfig`, `lexical_overlap(question, text) -> Overlap`, `rejection_reasons(item, config)`.
- `rag.eval.synthetic.split`: `split_dev_holdout(items, *, seed, holdout_share, min_each)`.
- `rag.eval.synthetic.pipeline`: `generate_questions(spans, model, ...)`, `critique_questions(items, model, ...)` (async), records with ledger usage.
- `rag.eval.questions.LabeledQuestion.meta` (optional provenance: span, scores, overlap; ignored by metrics).
- CLI `uv run python -m rag.eval.synthetic sample|generate|critique|project|build`; run records in git-ignored
  `data/reports/synthetic/<run_id>/`; sets in `evals/datasets/hf_docs/synthetic_{dev,holdout}.jsonl` + `synthetic_manifest.json`.

## Acceptance criteria
Unit tests in `tests/unit/test_eval_synthetic_{sampler,prompts,filters,split,pipeline,cli}.py`.
- [ ] AC1 Spans come from the cleaned page text (not chunks), are cut at line starts, have the configured length, same seed -> same spans — `test_sample_spans_raw_text_seeded`
- [ ] AC2 Allocation is proportional per stratum (repo + doc type), sums to n, one span per page; excluded pages and non-English pages are never sampled — `test_allocate_proportional`, `test_sample_spans_stratified_and_excluded`
- [ ] AC3 Tiny, code-only and boilerplate spans are skipped and counted by reason — `test_span_problem_reasons`
- [ ] AC4 Generation prompt has the HF factoid rules plus paraphrase rule and the span; parse reads question and answer, rejects missing parts — `test_generation_prompt_and_parse`
- [ ] AC5 Critique prompts (groundedness with span; relevance, standalone without) ask for evaluation before rating; parse reads `Total rating: n` in 1–5 — `test_critique_prompts_and_parse`
- [ ] AC6 Lexical overlap: longest shared word run and content-word recall — `test_lexical_overlap`
- [ ] AC7 Filter: any score below its threshold, missing score, parse failure or copy run over the limit gives a named reason; all reasons kept — `test_rejection_reasons`
- [ ] AC8 Split: seeded, stratified, disjoint, both >= min_each else error — `test_split_dev_holdout`
- [ ] AC9 Pipeline with fakes: generation and three critiques per question, records keep span, usage, generator and critic ids; generator != critic enforced — `test_pipeline_generate_and_critique`
- [ ] AC10 CLI with fake models: sample -> generate (resumable, `--limit`) -> critique -> build writes dev/holdout files loadable by `load_question_set` and a manifest with models, prompt versions, seeds, thresholds, counts, sha256 — `test_cli_end_to_end_with_fakes`

Also (by running, `llm`): pilot 20, then the rest; retrieval eval on dev and holdout; `docs/reports/P1-05-report.md`.

## Out of scope
Human-written hard questions and real queries (s14-2, later); chunk-level labels (optional, not done); retuning
retrieval on the dev set (follow-up); judge prompt changes (P1-04 open issue).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
