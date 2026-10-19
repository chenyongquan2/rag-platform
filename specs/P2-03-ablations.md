# P2-03 Ablation experiments on the dev set

Status: in review
Phase: P2
Depends on: P2-01 (rebuild flow with its injectable gate), P2-02 (rerank threshold 0.3 kept), P0-13b (dev-set confirmation, `compare`), P1-05 (dev and holdout sets)
Design sections: s14-4, s7-3, s7-5, s9-3, s10-1, s19 (P2 row)
Calls real models: yes, embedding only, for the chunk-size ablation (cap 3 CNY, estimate below)

## Goal
Run the design's ablations on the 99-question synthetic dev set (holdout once at the end, public 65 as an external
check) and report, per variant against the chosen baseline, "better N, worse N, unchanged N", the exact McNemar or
paired permutation result, Wilson intervals, latency, and which differences are below what 99 questions can show.
Variants: (a) route, (b) rerank on or off, (c) title path in the rerank input, (d) dense_vector type, (e) one other chunk
size, (f) the time of a full rebuild. Defaults change only on a reliable dev win.

## Interface
- `es.vector_index: int8_hnsw | bbq_hnsw | flat` (KB YAML, default `int8_hnsw`): type of the `embedding` mapping (`rag.index.mapping.index_body_for_kb`).
- `retrieval.rerank_with_title: bool` (KB YAML, default true) and `BatchedTimeoutRerank(include_title=True)`.
- `rag.eval.retrieval.ablation`: `ablation_table(baseline, variants, *, metrics=None, list_name="final", level="page", ...) -> AblationTable`, `render_ablation(table, *, baseline_name)`.
- CLI `python -m rag.eval.retrieval ablate BASELINE --variant NAME=RUN [--variant ...] [--baseline-name N] [--metrics a,b] [--list final] [--out FILE]`.
- `scripts/p2_03/`: cost estimate for a chunk size, variant-index build and evaluation, kNN timing and neighbour overlap, run lists.

## Acceptance criteria
Code (unit tests; the files named are the ones the test lives in):
- [x] AC1 `es.vector_index` sets the dense_vector type (`bbq_hnsw` keeps m, ef_construction and rescore; `flat` has none); the default mapping is unchanged; other values are refused — `test_index_mapping.py::test_vector_index_option_sets_the_dense_vector_mapping`
- [x] AC2 `include_title=False` sends the body alone to the reranker; the default still sends the title path — `test_retrieve_rerank.py::test_rerank_without_title_reads_the_body_only`
- [x] AC3 `retrieval.rerank_with_title` reaches the reranker, defaults to true, and changes the config version — `test_retrieve_pipeline.py::test_pipeline_passes_the_rerank_title_switch`
- [x] AC4 The ablation table gives better, worse, unchanged, test and p identical to `compare`, a verdict, default metrics and refuses unknown metrics or no variant — `test_eval_ablation.py::test_rows_say_better_worse_unchanged_with_the_repo_tests`, `::test_default_metrics_and_bad_metric`
- [x] AC5 The rendered table has the resolution sentence, Wilson intervals, latency and the "below the smallest difference" note — `test_eval_ablation.py::test_render_has_intervals_latency_resolution_and_small_difference_warning`
- [x] AC6 `ablate` prints and writes the table; a bad `--variant` exits 2 with one line on stderr — `test_eval_ablation.py::test_ablate_cli_prints_and_writes`

Experiments (by running; results in `docs/reports/P2-report.md`, section 「P2-03 消融实验」):
- [x] AC7 (a) BM25 only, vector only, both, each with and without rerank, dev set, against both-routes with rerank — report 2.4
- [x] AC8 (b) rerank on against off (pre-rerank list) — report 2.5
- [x] AC9 (c) title path in the rerank input against not — report 2.6
- [x] AC10 (d) `int8_hnsw` (fresh rebuild as control), `bbq_hnsw`, `flat`: index size after force merge, kNN latency, top-20 overlap with the exact (`flat`) list, Hit and MRR; embedding cache hits all, cost 0 — report 2.7
- [x] AC11 (e) one other chunk size through the P2-01 rebuild into a new index family (own prefix, live alias `kb_hf_docs` never touched); the estimate below is checked against the ledger; total spec cost at most 3 CNY or the run is not done and the item is listed under 「待确认」 in `docs/decisions.md` — report 2.8 (ran, 2.576 CNY)
- [x] AC12 (f) full rebuild time with all embeddings cached, and the measured embedding throughput for the real-embedding part, with the machine load noted — report 2.9
- [x] AC13 Holdout run once with the final setting, public 65 as an external check; the report says which differences are below the dev set's minimum detectable difference — report 2.10

## Chunk-size ablation: choice and cost estimate (written before any real call)
- Alternative: `token_split.max_tokens` 200 (live: 400). Direction: smaller, because design s7-2 retrieves small chunks and expands to the parent, and the reranker reads at most 512 tokens (text, title path, source path), so a larger size would be cut off inside the reranker and mix two effects. 300 is the fallback if the pilot shows 200 would pass the cap.
- `scripts/p2_03/estimate_chunk_cost.py` chunked the whole corpus and looked every embed text up in the L0 cache: size 400 control finds 25,920 of 25,920 texts (0 new). Size 300: 33,809 chunks, 22,548 new texts, 15.2 M characters, estimated 2.28 CNY. Size 200: 52,694 chunks, 45,227 new texts, 19.8 M characters, estimated 2.97 CNY (tokens per character 0.2605 measured on P0-12, price 0.0000005 CNY per token, margin 15%).
- 2.97 is too close to the cap, so a pilot embeds a 2% random sample of the new texts first (about 0.06 CNY, kept in the cache) and the run goes ahead only if chars times the measured tokens per character, plus the pilot, is at most 2.9 CNY; otherwise size 300 runs.
- Pilot result: 1,000 texts, 0.0568 CNY, 0.2579 tokens per character, projected 2.56 CNY, so size 200 ran. Actual spec total 2.576 CNY (ledger 13.197077 to 15.772906).

## Out of scope
Changing the analyzer, the fusion weights, `rerank_n`, the threshold (settled in P0-13b and P2-02); `bbq_disk` (Enterprise); the answer step; chunk sizes other than one; deleting the old live index.

## Handoff (filled at the end)
- Files: `src/rag/config/kb.py` (`es.vector_index`, `retrieval.rerank_with_title`), `src/rag/index/mapping.py`, `src/rag/retrieve/{rerank,pipeline}.py`, `src/rag/eval/retrieval/{ablation,cli}.py` (`ablate`), tests in `tests/unit/test_{index_mapping,retrieve_rerank,retrieve_pipeline,eval_ablation}.py` (3 + 4 new tests), `scripts/p2_03/` (estimate, pilot, prefill, cost guard, build and evaluate a variant index, kNN bench, drop), `docs/reports/P2-report.md` (section 2), `docs/decisions.md`, `docs/plan.md`. Raw records are in git-ignored `data/reports/p2-03/`.
- Results (dev 99, vs both routes with rerank 10, Hit@1 0.788 [0.70, 0.86]): BM25 only Hit@1 1 better / 5 worse (p 0.22), nDCG@8 5 / 14 (p 0.026); vector only Hit@5 1 / 10 (p 0.012); rerank off Hit@1 3 / 20 (p 0.0005); no title Hit@1 2 / 6 (p 0.29); bbq_hnsw and flat = int8_hnsw after rerank (0 / 0 / 99), no-rerank flat Hit@1 3 / 0 (p 0.25); chunk 200 Hit@1 6 / 11 (p 0.33), MRR 8 / 14 (p 0.23). Index sizes 158.8 / 135.1 / 130.6 MB (int8 / bbq / flat), kNN P50 6.4 / 6.1 / 14.2 ms. Rebuild with cached vectors 279 to 326 s (26,372 chunks), 488 s (52,694 chunks). Holdout once: Hit@1 0.720, same as P0-13b.
- Decisions: no default changed (no reliable dev win); recorded in `docs/decisions.md`. Chunk size 200 chosen (reranker reads at most 512 tokens, so larger sizes would be cut); cost estimated first (2.97 with margin), then a 1,000-text pilot (2.56 projected), run cost 2.576 CNY of the 3 CNY cap. Embedding cache was filled by `prefill_chunk_cache.py` (4 requests in flight, 1,035 texts a minute) because the ingest job sends one request at a time (175 to 480 a minute).
- Open issues: 99 questions cannot tell differences under 6 points (BM25 only, title path, chunk 200 Hit@1 are directions only); the first chunk-200 rebuild process ended silently after about 20 minutes with no error, cause not found (another rebuild with another prefix of the same KB finished in the same second; not reproduced); chunk 200 and the quantization variants were not run on public 65 or holdout; only one other chunk size; kNN timing is serial on a loaded machine; progress.md row not added (the merge step adds it).
- Review rounds: 0 (waiting for the reviewer)
