"""Spec P7-01 (1): reranker candidates on the HF docs dev set. No model gateway call.

Real index kb_hf_docs (read-only), query vectors from the Redis L0 cache, the reranker from the
local HF cache (HF_HUB_OFFLINE=1). For each model (RERANKER_MODEL) it runs, on the same fused
candidates the live chain sends (rerank_n 10, KB config unchanged):
  run      the retrieval eval on the 99 dev questions (Hit, MRR; the threshold 0.3 applied)
  curve    rerank-only latency for 10 candidates (3 warm-up questions), nDCG
  collect  every candidate's score with the threshold forced to 0 (dev, public 65, no-answer
           10 + 20 + hard), for the P2-02 threshold sweep
The machine load is printed before each step; the ledger total before and after.

  uv run python scripts/p7_01/rerankers.py SLUG=MODEL [SLUG=MODEL ...] [--steps run,curve,collect]
Output: data/reports/p7-01/ (git-ignored).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "data/reports/p7-01"
D = "evals/datasets/hf_docs"
DEV = f"{D}/synthetic_dev.jsonl"
SETS = [  # (questions, kind, name)
    (DEV, "answerable", "dev"),
    (f"{D}/huggingface_doc_qa_eval.jsonl", "answerable", "public65"),
    (f"{D}/no_answer.jsonl", "no_answer", "no_answer"),
    (f"{D}/no_answer_extra.jsonl", "no_answer", "no_answer_extra"),
    (f"{D}/no_answer_hard.jsonl", "no_answer", "no_answer_hard"),
]


def sh(args: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(["uv", "run", "python", *args], cwd=ROOT, env=env, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", metavar="SLUG=MODEL")
    p.add_argument("--steps", default="run,curve,collect")
    args = p.parse_args()
    steps = set(args.steps.split(","))
    base = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    sh(["-m", "rag.cost", "summary"], base)
    for item in args.models:
        slug, _, model = item.partition("=")
        env = {**base, "RERANKER_MODEL": model}
        print(f"== {slug}: {model}", flush=True)
        if "run" in steps:
            sh(["scripts/p2_03/load.py"], env)
            sh(
                [
                    "-m",
                    "rag.eval.retrieval",
                    "run",
                    "--kb",
                    "hf_docs",
                    "--questions",
                    DEV,
                    "--rerank-n",
                    "10",
                    "--label",
                    f"dev_{slug}_rerank10",
                    "--reports-dir",
                    OUT,
                    "--deadline-ms",
                    "120000",
                ],
                env,
            )
        if "curve" in steps:
            sh(["scripts/p2_03/load.py"], env)
            sh(
                [
                    "-m",
                    "rag.eval.retrieval",
                    "rerank-curve",
                    "--kb",
                    "hf_docs",
                    "--questions",
                    DEV,
                    "--rerank-ns",
                    "10",
                    "--warmup",
                    "3",
                    "--deadline-ms",
                    "120000",
                    "--out",
                    f"{OUT}/curve/{slug}",
                ],
                env,
            )
        if "collect" in steps:
            for questions, kind, name in SETS:
                if not (ROOT / questions).is_file():
                    continue
                sh(["scripts/p2_03/load.py"], env)
                sh(
                    [
                        "-m",
                        "rag.eval.threshold",
                        "collect",
                        "--kb",
                        "hf_docs",
                        "--questions",
                        questions,
                        "--kind",
                        kind,
                        "--out",
                        f"{OUT}/scores/{slug}/{name}.json",
                        "--deadline-ms",
                        "120000",
                    ],
                    env,
                )
        sh(["scripts/p2_03/load.py"], env)
    sh(["-m", "rag.cost", "summary"], base)
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
