#!/bin/bash
# official fusion variant runs (norerank + rerank10), 8 torch threads x 1 slot
S=${1:?dir holding kb_t1, kb_t05, kb_w2, kb_t1w2 (copies in data/reports/p0-13/kb_variants)}
cd /c/workspace/rag-platform
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 RERANKER_TORCH_THREADS=8 RERANK_MAX_CONCURRENCY=1
R=data/reports/p0-13/eval
uv run python -m rag.eval.retrieval run --kb hf_docs --no-rerank --label base_norerank --reports-dir $R
for name in t1 t05 w2 t1w2; do
  KB_DIR=$(cygpath -m $S/kb_$name) uv run python -m rag.eval.retrieval run --kb hf_docs --no-rerank --label ${name}_norerank --reports-dir $R
done
uv run python -m rag.eval.retrieval run --kb hf_docs --rerank-n 10 --label base_rerank10 --reports-dir $R
for name in t1 t05 w2 t1w2; do
  KB_DIR=$(cygpath -m $S/kb_$name) uv run python -m rag.eval.retrieval run --kb hf_docs --rerank-n 10 --label ${name}_rerank10 --reports-dir $R
done
echo ALLDONE
