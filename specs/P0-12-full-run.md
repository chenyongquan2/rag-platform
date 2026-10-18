# P0-12 P0 full run and report

Status: done
Phase: P0
Depends on: P0-01 to P0-11 (runs the ingest job of P0-06, the SearchPipeline of P0-08, the eval harness of P0-10)
Design sections: s14-6, s19 (P0 row), s10-1, s14-4, s1-2 (also s9-1, s10-2, s16-2)
Calls real models: yes (full-corpus embedding about 5-6 M tokens, about 2.5-2.9 CNY; 65 query embeddings, cached after the first run)

## Goal
The P0 numbers exist: all 2,647 HF docs are in `kb_hf_docs` with real vectors; the 65 public questions are run through
the real retrieval chain with and without rerank and per route; the rerank_n 10/20/30 latency curve sets the default
`rerank_n`; a Chinese report for the owner says which P0 acceptance items are met.

## Interface
- CLI `uv run python -m rag.eval.retrieval run --kb hf_docs [--rerank-n N] [--no-rerank] [--routes bm25,vector]
  [--deadline-ms MS] [--top-k K] [--questions PATH] [--index-prefix P] [--label L] [--reports-dir D]`
  (all AC tests in `tests/unit/test_eval_cli_run.py`; also `test_output_survives_a_non_utf8_console`):
  builds the chain from the KB config and service settings (`make_search_pipeline`, or `make_hybrid_search` with
  `--no-rerank`), runs `run_retrieval_eval`, writes the run with `write_run(public=is_public_kb_file(...))`, prints
  the run directory and the summary table. Model versions in the provenance include the reranker's local snapshot.
- CLI `uv run python -m rag.eval.retrieval rerank-curve --kb hf_docs [--rerank-ns 10,20,30] [--warmup 1] [--out D]`:
  `rerank_latency_curve` over the fusion-only chain and `ChunkReranker`, written with `write_rerank_curve`.
- `rag.eval.retrieval.cli.build_retriever(cfg, settings, *, rerank_n, rerank, routes, index_prefix)` (seam for tests).
- `EsHybridRetriever(routes=("bm25", "vector"))`: the routes it sends (a single-route run skips the other; not degraded).
- Config: `kb/public/hf_docs.yaml` `source.limit: null`, `retrieval.rerank_n` set from the curve.

## Acceptance criteria
- [x] AC1 `run` evaluates every question of the KB's question set through the built retriever and writes `results.json`
  and `report.md` under `public/` for a `kb/public/` config — `tests/unit/test_eval_cli_run.py::test_run_writes_public_record`
- [x] AC2 `--rerank-n N` overrides `retrieval.rerank_n` and the run's recorded KB config shows N; `--no-rerank` builds the
  fusion-only chain and labels the run — `test_run_rerank_options`
- [x] AC3 `--routes bm25` (or `vector`) builds a retriever that sends only that route; unknown route names are a usage
  error (exit 2) — `test_run_routes_option`, `tests/unit/test_retrieve_hybrid.py::test_single_route_sends_one_search`
- [x] AC4 `--deadline-ms` overrides the search deadline; an unknown KB is exit 2 — `test_run_deadline_and_unknown_kb`
- [x] AC5 provenance lists embed model, dim, reranker and reranker revision — `test_run_provenance_models`
- [x] AC6 `rerank-curve` writes `rerank_curve.json`/`.md` with one point per n plus fusion — `test_rerank_curve_command`

Also (by running, numbers in the handoff and in `docs/reports/P0-report.md`): full ingest; runs final/BM25/vector,
rerank vs no-rerank with `compare`; curve at 10/20/30; ES per-search timeout on kNN checked on the full index.

## Out of scope
ONNX int8 and a gateway rerank endpoint (deferred, see report); threshold calibration (P2 script, s10-2); tuning
title boost / num_candidates (dev set, P1/P2: tuning on the 65 public questions is not allowed, s14-6); answer eval (P1-04).

## Handoff (filled at the end)
- Files: `src/rag/eval/retrieval/cli.py` (`run`, `rerank-curve`, `build_retriever`/`build_reranker`/`reranker_revision` seams, console-safe output), `src/rag/retrieve/{hybrid,search,pipeline}.py` (`routes`), `kb/public/hf_docs.yaml` (`limit: null`, `rerank_n: 10`), `tests/unit/test_eval_cli_run.py` (7), `tests/unit/test_retrieve_hybrid.py` (+3), `docs/decisions.md`, `docs/reports/P0-report.md`. 715 unit tests pass. Raw records in git-ignored `data/reports/p0-12/`.
- Runs: full ingest 2,647 sources -> 2,617 pages, 26,372 chunks, 2,594 requests, 5,196,039 tokens, 2.598 CNY, 2,467 s, peak RSS 1.8 GB, index 152 MB. Page level (final top 8): no rerank Hit@1/Hit@5/MRR/nDCG@8 0.508/0.815/0.645/0.699; BM25 route 0.369/0.508/0.442/0.490; vector route 0.646/0.800/0.716/0.758; rerank 10 0.769/0.877/0.813/0.829 (Hit@1 19 better 2 worse, McNemar p 0.0002); rerank 30 0.769/0.938/0.840/0.868 (vs 10: no reliable difference). Rerank P95 (4 threads) 5.9/11.6/17.4 s for 10/20/30; 8 threads n=10 3.0 s; chain P95 5.8 s (4 threads), 3.1 s (8 threads). Other processes used 25-45% of the CPU before each measurement (3 s sample); whole-system means during runs (47-65%) include our own process. Ledger total 2.652 CNY. C: free 22 -> 20 GB.
- Decisions (docs/decisions.md): `rerank_n: 10`; torch threads unchanged (4 x 2) pending a concurrent-load test; ES per-search timeout is rarely applied to kNN (7, 1 and 0 of 45 searches timed out in three runs; raw output in `data/reports/p0-12/knn_timeout-run*.txt`), not reliable; P0-07 handling unchanged; `run` uses the service factories and `service_acl_groups`, `--routes` single-route runs are not degraded; raw records outside `evals/reports/public`.
- Open issues: search_docs P95 target (3 s) not met with defaults; ONNX int8 and gateway rerank comparison not done; threshold 0.3 leaves 4.8 chunks per list on average (uncalibrated); questions 015/017 missed by every list, labels unchecked; RRF puts wrong BM25 rank-1 chunks first (Hit@1 0.508 vs vector 0.646), check on the dev set; `compare` used to crash on a GBK console (fixed here, P0-10 code).
- Review rounds: 2. Round 1 CHANGES (CPU load figures were whole-system and included our own processes; kNN timeout evidence weak and raw output not saved; threshold comment said calibrated; CRLF rewrite of the P1-03 decisions), all fixed. Round 2 APPROVE (coordinator).
