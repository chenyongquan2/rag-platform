"""Spec P7-01 (2): do two runs give the answer step the same pages, question by question?

  uv run python scripts/p7_01/check_contexts.py ANSWER_RUN P3_07_RECORDS_JSONL

The P3-07 standard-RAG records keep only the pages of the fragments (`context_pages`), not every
chunk, so the check is page level: the same pages in the same first-seen order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.eval.answer.runner import load_answer_run


def main(run_path: str, records_path: str) -> int:
    run = load_answer_run(run_path)
    ours = {}
    for q in run.questions:
        pages = [c.doc_id for c in (q.answer.chunks if q.answer else [])]
        ours[q.qid] = list(dict.fromkeys(pages))
    theirs = {}
    for line in Path(records_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("system") == "standard" and rec["qid"] in ours:
            theirs[rec["qid"]] = list(dict.fromkeys(rec.get("context_pages") or []))
    same_order = sum(ours[q] == theirs[q] for q in theirs)
    same_set = sum(set(ours[q]) == set(theirs[q]) for q in theirs)
    print(
        f"paired {len(theirs)} of {len(ours)}: "
        f"same pages in order {same_order}, same set {same_set}"
    )
    for q in sorted(theirs):
        if set(ours[q]) != set(theirs[q]):
            print(f"  {q}: ours {ours[q]}\n  {' ' * len(q)}  P3-07 {theirs[q]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
