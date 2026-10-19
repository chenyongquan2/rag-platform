# P2-03 Ablation experiments on the dev set

Status: done
Phase: P2
Depends on: P2-01 (rebuild flow), P2-02 (threshold 0.3), P0-13b (`compare`), P1-05 (dev and holdout sets)
Design sections: s14-4, s7-3, s7-5, s9-3, s10-1, s19 (P2 row)
Calls real models: yes, embedding only, for the chunk-size ablation (cap 3 CNY, estimate below)

## Goal
Run the design's ablations on the 99-question dev set (holdout once at the end, public 65 as an external check) and
report, per variant against the baseline, better / worse / unchanged, the exact McNemar or permutation result, Wilson
intervals, latency, and which differences 99 questions cannot show. Defaults change only on a reliable dev win.

## Interface
- `es.vector_index: int8_hnsw | bbq_hnsw | flat` (KB YAML, default `int8_hnsw`); `retrieval.rerank_with_title` (default true), `BatchedTimeoutRerank(include_title=True)`.
- `rag.eval.retrieval.ablation`: `ablation_table(...)`, `render_ablation(...)`; CLI `python -m rag.eval.retrieval ablate BASELINE --variant NAME=RUN ...`.
- `scripts/p2_03/`: cost estimate, pilot, cache prefill, cost guard, variant build and evaluation, kNN bench, drop.

## Acceptance criteria
Code (unit tests):
- [x] AC1 `es.vector_index` sets the dense_vector type, default mapping unchanged, other values refused — `test_index_mapping.py::test_vector_index_option_sets_the_dense_vector_mapping`
- [x] AC2 `include_title=False` sends the body alone, default unchanged — `test_retrieve_rerank.py::test_rerank_without_title_reads_the_body_only`
- [x] AC3 `retrieval.rerank_with_title` reaches the reranker and changes the config version — `test_retrieve_pipeline.py::test_pipeline_passes_the_rerank_title_switch`
- [x] AC4 The ablation table equals `compare`, with verdict, default metrics and input checks — `test_eval_ablation.py::test_rows_say_better_worse_unchanged_with_the_repo_tests`, `::test_default_metrics_and_bad_metric`
- [x] AC5 The rendered table has the resolution sentence, Wilson intervals, latency, the small-difference note — `test_eval_ablation.py::test_render_has_intervals_latency_resolution_and_small_difference_warning`
- [x] AC6 `ablate` prints and writes; a bad `--variant` exits 2 — `test_eval_ablation.py::test_ablate_cli_prints_and_writes`

Experiments (by running; `docs/reports/P2-report.md` section 2):
- [x] AC7 (a) routes, with and without rerank (2.4); AC8 (b) rerank on or off (2.5); AC9 (c) title path in rerank input (2.6)
- [x] AC10 (d) `int8_hnsw` (fresh control), `bbq_hnsw`, `flat`: size, kNN latency, overlap with exact, Hit and MRR, cache hits, cost 0 (2.7)
- [x] AC11 (e) chunk size 200 through the P2-01 rebuild into its own index family, live alias untouched, cost 2.576 CNY of 3 (2.8)
- [x] AC12 (f) rebuild time with cached vectors (without gate) and embedding throughput, load noted (2.9)
- [x] AC13 Holdout once with the final setting, public 65 as check; small differences named (2.10)

## Chunk-size choice and estimate (written before any real call)
Size 200 (live 400): design s7-2 retrieves small chunks, and a larger size would be cut inside the reranker (512 tokens). The estimate (`estimate_chunk_cost.py`, every embed text looked up in the cache; the 400 control finds 100%) was 45,227 new texts, 2.97 CNY with a 15% margin, too close to the cap; a 1,000-text pilot measured 0.2579 tokens per character and projected 2.56 CNY, so it ran. Fallback was size 300 (2.28 CNY).

## Out of scope
Analyzer, fusion weights, `rerank_n`, threshold; `bbq_disk` (Enterprise); the answer step; other chunk sizes.

## Handoff (filled at the end)
- Files: config, mapping, rerank and pipeline code, `ablation.py`, `cli.py` (`ablate`), four unit-test files (7 new tests), `scripts/p2_03/`, `docs/reports/P2-report.md` (section 2), `docs/decisions.md`, `docs/plan.md`. Raw records are in git-ignored `data/reports/p2-03/`.
- Decisions: no default changed (no reliable dev win; with 32 tests on Hit@1 and MRR@8 (24 dev, 8 public 65) only rerank-off Hit@1, p 0.0005, and vector-only no-rerank nDCG@8, p 0.0003 (secondary metric), pass the about 0.0016 threshold; vector-only and BM25-only are direction only and must not be cited for dropping BM25). Chunk size 200 chosen by the reasons above; recorded in `docs/decisions.md`.
- Open issues: 99 questions cannot show differences under 6 points; the first chunk-200 rebuild was killed mid-embedding by my own psutil kill (command lines containing `p203c200_`, about 10:14:30 to 10:14:46 UTC) followed by `drop_variant.py p203c200_` (ES deletion 10:15:09); I meant to stop the slow rebuild and refill the cache faster; established from the transcript and the ES log, see report 2.12; all rebuilds ran with `--no-gate`, the current `rebuild` gate only compares chunk counts (seconds), so a setting-only rebuild is about 5 minutes; an evaluation gate (design s7-5) does not exist yet and would add about 7 minutes, listed in `docs/plan.md` as a P2 follow-up; chunk 200 and the quantization variants were not run on public 65 or holdout; one other chunk size only; `progress.md` row is added by the merge step.
- Review rounds: 2. Round 1 CHANGES (ten notes: multiple-test wording, rebuild-kill facts, `pipefail`, gate and throughput disclosure, table columns, docstring AC numbers, `ChunkReranker` note, spec length, 512-token sentence); round 2 CHANGES (three wording fixes: default gate is a chunk-count floor, kill cause stated as fact, consistent multiple-test count); both fixed after each round; merged.
