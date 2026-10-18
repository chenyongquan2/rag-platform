# P1-05 Synthetic questions, critique filters, dev/holdout split

Status: review
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
Unit tests in `tests/unit/test_eval_synthetic_{sampler,prompts,filters,pipeline,cli}.py`.
- [x] AC1 Spans come from the cleaned page text (not chunks), are cut at line starts, have the configured length, same seed -> same spans — `test_sample_spans_raw_text_seeded`
- [x] AC2 Allocation is proportional per stratum (repo + doc type), sums to n, one span per page; excluded pages and non-English pages are never sampled; a stratum's shortfall is drawn from the others — `test_allocate_proportional`, `test_sample_spans_stratified_and_excluded`, `test_shortfall_reallocated`
- [x] AC3 Tiny, code-only and boilerplate spans are skipped and counted by reason — `test_span_problem_reasons`
- [x] AC4 Generation prompt has the HF factoid rules plus paraphrase rule and the span; parse reads question and answer, rejects missing parts — `test_generation_prompt_and_parse`
- [x] AC5 Critique prompts (groundedness with span; relevance, standalone without) ask for evaluation before rating; parse reads `Total rating: n` in 1–5 — `test_critique_prompts_and_parse`
- [x] AC6 Lexical overlap: longest shared word run and content-word recall — `test_lexical_overlap`
- [x] AC7 Filter: any score below its threshold, missing score, parse failure or copy run over the limit gives a named reason; all reasons kept; token-like strings are redacted before writing — `test_rejection_reasons`, `test_redact_secrets`
- [x] AC8 Split: seeded, stratified, disjoint, both >= min_each else error — `test_split_dev_holdout`
- [x] AC9 Pipeline with fakes: generation and three critiques per question, records keep span, usage, generator and critic ids; generator != critic enforced — `test_pipeline_generate_and_critique`
- [x] AC10 CLI with fake models: sample -> generate (resumable, `--limit`, one event loop for all batches) -> `sample --extend` -> critique -> build writes dev/holdout files loadable by `load_question_set` and a manifest with models, prompt versions, seeds, thresholds, counts, sha256 — `test_cli_end_to_end_with_fakes`

Also (by running, `llm`): pilot 20, then the rest; retrieval eval on dev and holdout; `docs/reports/P1-05-report.md`.

Actual (by running, `llm`): pilot 20 spans 0.222 CNY (generation 0.0059, critique 0.0052 per question; projected 4.67
for 420, matching the estimate). The pilot kept 6 of 20, so the run grew to 800 spans (docs/decisions.md). Records:
generation 800 calls 5.32 CNY (698 prompt, 656 completion of which 626 reasoning per call; the estimate assumed 500
completion), critique 2,397 calls 4.07 CNY; ledger 3.502 -> 13.20 CNY (9.69 CNY; 0.30 CNY of it is calls whose
records were lost when two stalled runs were stopped and resumed). Question embeddings for the sanity eval < 0.01 CNY.

## Out of scope
Human-written hard questions and real queries (s14-2, later); chunk-level labels (optional, not done); retuning
retrieval on the dev set (follow-up); judge prompt changes (P1-04 open issue).

## Handoff (filled at the end)
- Files: `src/rag/eval/synthetic/{__init__,__main__,sampler,prompts,filters,split,pipeline,cli}.py`, `src/rag/eval/questions.py` (`LabeledQuestion.meta`), `tests/unit/test_eval_synthetic_{sampler,prompts,filters,pipeline,cli}.py` (13 tests), `evals/datasets/hf_docs/{synthetic_dev,synthetic_holdout}.jsonl`, `synthetic_manifest.json`, `NOTICE.md`, `README.md`, `docs/decisions.md`, `docs/reports/P1-05-report.md`. Run records in git-ignored `data/reports/synthetic/20261018T135147Z_hf_docs_synthetic/` and `data/reports/p1-05/`.
- Results: 800 spans (420 seed 0 + 380 seed 1 via `sample --extend` after the pilot kept 6/20), 799 questions, 200 kept (relevance < 4: 533, standalone < 4: 208, copied wording: 37, duplicates: 6, groundedness < 4: 1, parse failure: 1), dev 100 / holdout 100. Retrieval (P0-13 settings) Hit@1 / Hit@8 / MRR@8: public 0.831 / 0.985 / 0.896, dev 0.660 / 0.890 / 0.753, holdout 0.670 / 0.880 / 0.753; BM25-route Hit@1 0.754 / 0.570 / 0.500 (not trivially easy, far below 0.95); synthetic questions share fewer words with their pages than the public ones (median longest run 3 vs 4).
- Decisions (docs/decisions.md): spans from cleaned page text, strata repo + doc type, proportional, one span per page, shortfall drawn from other strata; public questions' pages excluded from both sets; English pages only; HF thresholds (all three >= 4), answer <= 300 chars, copy limit 6 words from the public set, exact duplicates dropped; 800 spans instead of 420; 50/50 stratified split seed 0; token-like strings redacted (gitleaks found an `hf_` token in a corpus README); generator = answer model under test (limitation).
- Open issues: page labels only — 33/200 spans also occur verbatim on other pages and all 7 questions missed by every list are such label problems (multi-page labels or overrides needed); 2 dev/holdout pairs ask the same thing (templated timm model cards); relevance critic drops 67%, sets lean to "generic" questions; answers not checked by a person; the two synthetic retrieval runs are marked dirty (an uncommitted spec edit, no retrieval change); P0-13 choices not yet re-checked on dev.
- Review rounds: pending.
