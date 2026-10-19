#!/bin/bash
# Spec P2-03 (a)-(c): route, rerank and rerank-title ablations on the synthetic DEV set only.
# Real index kb_hf_docs (read-only), local reranker, query vectors from the Redis L0 cache (no
# gateway call: the cost ledger must not change; the script prints the ledger before and after).
#   run_dev.sh                  baseline, routes, rerank off, no title (dev)
#   run_dev.sh public65         the same variants on the 65 public questions (external check)
# Runs go to data/reports/p2-03/public/<run id>/; the table is made afterwards with
#   uv run python -m rag.eval.retrieval ablate BASE --variant NAME=RUN ... --reports-dir data/reports/p2-03
set -e
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
R=data/reports/p2-03
V=$R/kb_variants
SET=${1:-dev}
if [ "$SET" = "public65" ]; then Q=evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl; else Q=evals/datasets/hf_docs/synthetic_dev.jsonl; fi
uv run python -m rag.cost summary | sed -n 3p
run() { # label kb_dir extra-args...
  local label=$1 dir=$2; shift 2
  uv run python scripts/p2_03/load.py
  echo "== $SET $label"
  KB_DIR=$dir uv run python -m rag.eval.retrieval run --kb hf_docs --questions $Q \
    --label "${SET}_$label" --reports-dir $R --deadline-ms 60000 "$@" | tail -7
}
run base_norerank kb --no-rerank
run bm25_norerank kb --no-rerank --routes bm25
run vector_norerank kb --no-rerank --routes vector
run base_rerank10 kb --rerank-n 10
run bm25_rerank10 kb --rerank-n 10 --routes bm25
run vector_rerank10 kb --rerank-n 10 --routes vector
run notitle_rerank10 $V/kb_notitle --rerank-n 10
uv run python -m rag.cost summary | sed -n 3p
echo ALLDONE
