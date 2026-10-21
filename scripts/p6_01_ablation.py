"""Ablations of the PDF knowledge base on its small question set (spec P6-01).

    uv run python scripts/p6_01_ablation.py prepare
        writes one knowledge-base folder per variant under data/p601_variants/<name>/ (a copy of
        kb/private/pdf_demo.yaml with one thing switched), to be used with KB_DIR=... and a
        throw-away index prefix:
          KB_DIR=data/p601_variants/nosum uv run python -m rag.ingest.job --kb pdf_demo --all \\
              --index-prefix p601a_
          KB_DIR=data/p601_variants/nosum uv run python -m rag.eval.retrieval run --kb pdf_demo \\
              --index-prefix p601a_ --label nosum --deadline-ms 60000
    uv run python scripts/p6_01_ablation.py summarize full=<run dir> nosum=<run dir> ...
        prints Hit@1, Hit@3, Hit@8 and MRR per question type for each run, and for every run
        after the first the paired counts against it (questions better, worse, equal).

The first run given to `summarize` is the baseline. Variants:
  full     summary index with model summaries and figure descriptions (the committed config)
  nosum    no summary chunks: a table is found only through its own text
  noimg    no figure descriptions: the figure is dropped from the page text
  rulesum  summary chunks written by a rule (caption, column names, first column), no model
  norepl   the full index, but a summary hit is not replaced by its table (`replace_table_summaries`
           off): what the answer would see without the replacement step
  before   the full index, the table replaces its summary before the reranker, not after it
           (the committed config replaces after the reranker: see docs/decisions.md)
With n of 22 questions the numbers show a direction, nothing more.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kb" / "private" / "pdf_demo.yaml"
QUESTIONS = ROOT / "evals" / "datasets" / "pdf_demo" / "questions.jsonl"
VARIANTS = ROOT / "data" / "p601_variants"
KS = (1, 3, 8)


def variant_config(name: str) -> dict[str, Any]:
    data = yaml.safe_load(SOURCE.read_text("utf-8"))
    pdf, image = data["pipelines"][".pdf"], data["pipelines"][".png"]
    if name == "nosum":
        for step in pdf["steps"]:
            if isinstance(step, dict) and "table_summary" in step:
                step["table_summary"]["mode"] = "off"
        data["retrieval"]["replace_table_summaries"] = False
    elif name == "rulesum":
        for step in pdf["steps"]:
            if isinstance(step, dict) and "table_summary" in step:
                step["table_summary"]["mode"] = "rule"
    elif name == "noimg":
        pdf["reader_params"]["describe_images"] = "off"
        image["reader_params"]["describe_images"] = "off"
    elif name == "norepl":
        data["retrieval"]["replace_table_summaries"] = False
    elif name == "before":
        data["retrieval"]["replace_table_summaries_after_rerank"] = False
    else:
        raise SystemExit(f"unknown variant {name!r}")
    return data


def prepare() -> None:
    for name in ("nosum", "noimg", "rulesum", "norepl", "before"):
        folder = VARIANTS / name / "private"
        folder.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(variant_config(name), sort_keys=False, allow_unicode=True)
        (folder / "pdf_demo.yaml").write_bytes(text.encode("utf-8"))
        print(f"wrote {folder / 'pdf_demo.yaml'}")


def load_questions() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in QUESTIONS.read_bytes().decode("utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[q["qid"]] = q
    return out


def first_hit_rank(hits: list[dict[str, Any]], relevant: set[str]) -> int | None:
    """Rank (1-based) of the first chunk of a relevant page in the final list."""
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] in relevant:
            return rank
    return None


def load_run(path: Path) -> dict[str, Any]:
    file = path / "results.json" if path.is_dir() else path
    return json.loads(file.read_bytes().decode("utf-8"))


def ranks_of(run: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    return {
        q["qid"]: first_hit_rank(q["hits"], set(questions[q["qid"]]["relevant_pages"]))
        for q in run["questions"]
    }


def flagged(run: dict[str, Any]) -> list[str]:
    """Questions answered by a degraded search or with a rerank time-out (not clean data)."""
    return [q["qid"] for q in run["questions"] if q.get("degraded") or q.get("rerank_timeout")]


def metrics(ranks: dict[str, int | None], qids: list[str]) -> dict[str, float]:
    n = len(qids) or 1
    out = {f"hit@{k}": sum(1 for q in qids if (ranks[q] or 99) <= k) / n for k in KS}
    out["mrr"] = sum(1 / ranks[q] for q in qids if ranks[q]) / n
    return out


def summarize(pairs: list[tuple[str, Path]]) -> str:
    questions = load_questions()
    runs = {name: load_run(path) for name, path in pairs}
    ranks = {name: ranks_of(run, questions) for name, run in runs.items()}
    by_type: dict[str, list[str]] = defaultdict(list)
    for qid, q in questions.items():
        by_type[q["meta"]["type"]].append(qid)
    groups = {"all": list(questions), **dict(sorted(by_type.items()))}
    lines = ["| run | group | n | hit@1 | hit@3 | hit@8 | mrr |", "|---|---|---|---|---|---|---|"]
    for name in runs:
        for group, qids in groups.items():
            m = metrics(ranks[name], qids)
            lines.append(
                f"| {name} | {group} | {len(qids)} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | "
                f"{m['hit@8']:.2f} | {m['mrr']:.3f} |"
            )
    base_name = pairs[0][0]
    for name in list(runs)[1:]:
        lines += [
            "",
            f"paired against {base_name}: {name}",
            "",
            "| group | better | worse | equal |",
        ]
        lines.append("|---|---|---|---|")
        for group, qids in groups.items():
            better = worse = equal = 0
            for q in qids:
                a, b = ranks[base_name][q] or 99, ranks[name][q] or 99
                better += b < a
                worse += b > a
                equal += b == a
            lines.append(f"| {group} | {better} | {worse} | {equal} |")
        changed = [
            f"{q}: rank {ranks[base_name][q] or 'miss'} -> {ranks[name][q] or 'miss'}"
            for q in questions
            if (ranks[base_name][q] or 99) != (ranks[name][q] or 99)
        ]
        lines.append("")
        lines.append("changed questions: " + ("; ".join(changed) if changed else "none"))
    for name, run in runs.items():
        bad = flagged(run)
        if bad:
            lines.append(f"\nwarning: run {name} has degraded or rerank-timeout answers: {bad}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "prepare":
        prepare()
        return 0
    if len(argv) >= 2 and argv[0] == "summarize":
        pairs = []
        for item in argv[1:]:
            name, sep, path = item.partition("=")
            if not sep:
                raise SystemExit("usage: summarize name=run_dir ...")
            pairs.append((name, Path(path)))
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        print(summarize(pairs), end="")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
