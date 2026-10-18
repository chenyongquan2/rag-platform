#!/bin/bash
S=${1:?scratch dir holding kb_t1/public/hf_docs.yaml}
cd /c/workspace/rag-platform
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
KB=$(cygpath -m $S/kb_t1)
mkdir -p data/reports/p0-13/load
for cfg in 4x2 8x1 8x2; do
  IFS=x read th sl <<<"$cfg"
  RERANKER_TORCH_THREADS=$th RERANK_MAX_CONCURRENCY=$sl uv run python scripts/p0_13/load_test.py data/reports/p0-13/load/load_$cfg.json $KB 2>&1 | grep -v -i "warn\|Loading weights"
done
echo ALLDONE
