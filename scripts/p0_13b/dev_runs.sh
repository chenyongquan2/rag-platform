#!/bin/bash
# Spec P0-13b: P0-13 fusion variants and rerank_n on the synthetic DEV set only (never holdout).
# Variant KB dirs (copies of kb/ with one setting changed) in data/reports/p0-13b/kb_variants:
#   kb_t2 (title_boost 2.0), kb_t05 (0.5), kb_w2 (title 2.0 + vector weight 2), kb_t1w2 (title 1.0
#   + vector weight 2); the repo's kb/ is the chosen P0-13 setting (title 1.0, equal weights).
set -e
cd /c/workspace/rag-platform
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
R=data/reports/p0-13b
Q=evals/datasets/hf_docs/synthetic_dev.jsonl
V=data/reports/p0-13b/kb_variants
run() { # name kb_dir extra-args...
  local name=$1 dir=$2; shift 2
  KB_DIR=$dir uv run python -m rag.eval.retrieval run --kb hf_docs --questions $Q \
    --label "dev_$name" --reports-dir $R "$@" | tail -6
}
run t1_norerank kb --no-rerank
for v in t2 t05 w2 t1w2; do run ${v}_norerank $V/kb_$v --no-rerank; done
run t1_rerank10 kb --rerank-n 10
for v in t2 t05 w2 t1w2; do run ${v}_rerank10 $V/kb_$v --rerank-n 10; done
run t1_rerank20 kb --rerank-n 20
echo ALLDONE
