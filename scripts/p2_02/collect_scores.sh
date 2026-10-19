#!/bin/bash
# Spec P2-02: rerank scores of every candidate (threshold forced to 0) for the threshold sweep.
# Real search chain on the real index (read-only), local reranker, query vectors from the L0 cache
# (the 20 questions of no_answer_extra.jsonl are new: about 20 query embeddings, < 0.001 CNY).
#   collect_scores.sh          dev, public 65, no-answer 10 and 20 (what chooses and checks)
#   collect_scores.sh holdout  the synthetic holdout set: run ONCE, after the threshold is chosen
# Output: data/reports/p2-02/*.json (git-ignored). Then, for example:
#   uv run python -m rag.eval.threshold sweep --dev data/reports/p2-02/dev.json \
#     --check public65=data/reports/p2-02/public65.json \
#     --no-answer original10=data/reports/p2-02/no_answer.json \
#     --no-answer extra20=data/reports/p2-02/no_answer_extra.json
set -e
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
R=data/reports/p2-02
D=evals/datasets/hf_docs
mkdir -p $R
collect() { # questions kind out-name
  uv run python -m rag.eval.threshold collect --kb hf_docs --questions "$1" --kind "$2" \
    --out "$R/$3.json" --deadline-ms 60000
}
if [ "$1" = "holdout" ]; then
  collect $D/synthetic_holdout.jsonl answerable holdout
else
  collect $D/no_answer.jsonl no_answer no_answer
  collect $D/no_answer_extra.jsonl no_answer no_answer_extra
  collect $D/synthetic_dev.jsonl answerable dev
  collect $D/huggingface_doc_qa_eval.jsonl answerable public65
fi
echo ALLDONE
