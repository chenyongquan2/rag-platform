# P0-12 P0 full run and report

Status: in progress
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
  [--deadline-ms MS] [--top-k K] [--questions PATH] [--index-prefix P] [--label L] [--reports-dir D]`:
  builds the chain from the KB config and service settings (`make_search_pipeline`, or `make_hybrid_search` with
  `--no-rerank`), runs `run_retrieval_eval`, writes the run with `write_run(public=is_public_kb_file(...))`, prints
  the run directory and the summary table. Model versions in the provenance include the reranker's local snapshot.
- CLI `uv run python -m rag.eval.retrieval rerank-curve --kb hf_docs [--rerank-ns 10,20,30] [--warmup 1] [--out D]`:
  `rerank_latency_curve` over the fusion-only chain and `ChunkReranker`, written with `write_rerank_curve`.
- `rag.eval.retrieval.cli.build_retriever(cfg, settings, *, rerank_n, rerank, routes, index_prefix)` (seam for tests).
- `EsHybridRetriever(routes=("bm25", "vector"))`: the routes it sends (a single-route run skips the other; not degraded).
- Config: `kb/public/hf_docs.yaml` `source.limit: null`, `retrieval.rerank_n` set from the curve.

## Acceptance criteria
- [ ] AC1 `run` evaluates every question of the KB's question set through the built retriever and writes `results.json`
  and `report.md` under `public/` for a `kb/public/` config — `tests/unit/test_eval_cli_run.py::test_run_writes_public_record`
- [ ] AC2 `--rerank-n N` overrides `retrieval.rerank_n` and the run's recorded KB config shows N; `--no-rerank` builds the
  fusion-only chain and labels the run — `test_run_rerank_options`
- [ ] AC3 `--routes bm25` (or `vector`) builds a retriever that sends only that route; unknown route names are a usage
  error (exit 2) — `test_run_routes_option`, `tests/unit/test_retrieve_hybrid.py::test_single_route_sends_one_search`
- [ ] AC4 `--deadline-ms` overrides the search deadline; an unknown KB is exit 2 — `test_run_deadline_and_unknown_kb`
- [ ] AC5 provenance lists embed model, dim, reranker and reranker revision — `test_run_provenance_models`
- [ ] AC6 `rerank-curve` writes `rerank_curve.json`/`.md` with one point per n plus fusion — `test_rerank_curve_command`

Also (by running, numbers in the handoff and in `docs/reports/P0-report.md`): full ingest; runs final/BM25/vector,
rerank vs no-rerank with `compare`; curve at 10/20/30; ES per-search timeout on kNN checked on the full index.

## Out of scope
ONNX int8 and a gateway rerank endpoint (deferred, see report); threshold calibration (P2 script, s10-2); tuning
title boost / num_candidates (dev set, P1/P2: tuning on the 65 public questions is not allowed, s14-6); answer eval (P1-04).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
