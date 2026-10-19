#!/bin/bash
# Spec P2-03 (d), (e), (f): build one ablation index family with the P2-01 rebuild flow, measure
# it, evaluate it on the DEV set, delete it. The live alias kb_hf_docs is never touched: every
# index of the family has its own prefix (p203xxx_) and its own docstore.
#   build_variant.sh NAME KB_DIR PREFIX [bench]
#     NAME     label of the runs (int8, bbq, flat, c200, ...)
#     KB_DIR   directory with the variant's hf_docs.yaml (data/reports/p2-03/kb_variants/kb_NAME)
#     PREFIX   index prefix, must start with p203 (e.g. p203bbq_)
#     bench    also run scripts/p2_03/knn_bench.py (quantization ablation)
#   GUARD_MAX=2.9 build_variant.sh ...   kill the rebuild if it spends more than 2.9 CNY
# Steps: init (empty v1 and the alias) -> rebuild (v2 written in full, synced, force-merged, no
# gate: the family is new and only this script reads it; the gate of a real switch is the dev
# eval below) -> status -> kNN bench -> dev eval without and with rerank -> delete the family.
# Stops without building when the C: drive has less than 1.5 GB free.
set -eo pipefail   # a failed or killed command inside a pipe must stop the script
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
NAME=$1; DIR=$2; PREFIX=$3; BENCH=$4
R=data/reports/p2-03
Q=evals/datasets/hf_docs/synthetic_dev.jsonl
case "$PREFIX" in p203*) ;; *) echo "prefix must start with p203"; exit 2;; esac
free_mb=$(df -m /c | awk 'NR==2{print $4}')
echo "free on C: ${free_mb} MB"
if [ "$free_mb" -lt 1500 ]; then echo "STOP: less than 1.5 GB free"; exit 3; fi
uv run python -m rag.cost summary | sed -n 2,3p
uv run python scripts/p2_03/load.py
export KB_DIR=$DIR
echo "== init $NAME"
uv run python -m rag.index.cli init --kb hf_docs --index-prefix $PREFIX
if [ -n "$GUARD_MAX" ]; then   # a variant that embeds for real: stop it at its share of the budget
  uv run python scripts/p2_03/cost_guard.py --max-delta $GUARD_MAX --match $PREFIX     > $R/guard_$NAME.log 2>&1 &
  sleep 5
fi
echo "== rebuild $NAME  start $(date -Is)"
t0=$(date +%s%N)
uv run python -m rag.index.cli rebuild --kb hf_docs --index-prefix $PREFIX --all --no-gate \
  | tee $R/rebuild_$NAME.log
t1=$(date +%s%N)
echo "rebuild wall seconds: $(( (t1 - t0) / 1000000000 ))  end $(date -Is)" | tee -a $R/rebuild_$NAME.log
uv run python scripts/p2_03/load.py
uv run python -m rag.index.cli status --kb hf_docs --index-prefix $PREFIX | tee $R/status_$NAME.log
uv run python -m rag.cost summary | sed -n 2,3p
if [ "$BENCH" = "bench" ]; then
  uv run python scripts/p2_03/knn_bench.py bench --alias ${PREFIX}kb_hf_docs --name $NAME \
    --repeats 10 --out $R/knn_$NAME.json
fi
for mode in "norerank --no-rerank" "rerank10 --rerank-n 10"; do
  set -- $mode
  label=$1; shift
  uv run python scripts/p2_03/load.py
  echo "== dev eval $NAME $label"
  uv run python -m rag.eval.retrieval run --kb hf_docs --questions $Q --index-prefix $PREFIX \
    --label "dev_${NAME}_$label" --reports-dir $R --deadline-ms 60000 "$@" | tail -7
done
uv run python -m rag.cost summary | sed -n 2,3p
uv run python scripts/p2_03/drop_variant.py $PREFIX
echo VARIANTDONE $NAME
