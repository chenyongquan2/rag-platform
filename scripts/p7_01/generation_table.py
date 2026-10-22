"""Spec P7-01 (2): paired tables of the candidate chat models against deepseek-v4-flash.

No model call. Inputs (all git-ignored records):
  baseline   P3-07 standard RAG on the 65 public questions (deepseek-v4-flash, today's main):
             records.jsonl + judge.jsonl (run 1) + support lines; the 4 answers that timed out at
             the gateway are replaced by their P3-07 re-run (p3-07-retry), as the candidates'
             gateway errors were re-asked (`answer run --retry-errors`).
  candidate  an answer run (results.json), its judge run (1-3) and optionally its support lines.

  uv run python scripts/p7_01/generation_table.py --p307 C:/workspace/rag-platform-data/p3-07 \
      --candidate NAME=ANSWER_RUN,JUDGE_RUN[,SUPPORT_DIR] ... [--exclude q1,q2] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.eval.answer.judge import load_judge_run
from rag.eval.answer.pairing import PairedOutcome, paired_answer_table, render_paired_answers
from rag.eval.answer.runner import load_answer_run
from rag.eval.support.records import latest_lines, read_lines

PUBLIC = "huggingface_doc_qa_eval:"


def _support(lines_path: Path, system: str | None) -> dict[str, tuple[float | None, int, int]]:
    """qid -> (sentence recall, supported pairs, judged pairs) from support lines."""
    out: dict[str, tuple[float | None, int, int]] = {}
    for line in latest_lines(read_lines(lines_path)):
        if (system and line.system != system) or line.result is None or line.run != 1:
            continue
        sentences = [s for s in line.result.sentences if s.status != "unknown"]
        labels = [p.label for s in line.result.sentences for p in s.pairs if p.label != "unknown"]
        recall = (
            sum(s.status == "supported" for s in sentences) / len(sentences) if sentences else None
        )
        out[line.qid] = (recall, labels.count("supported"), len(labels))
    return out


def baseline(root: Path) -> dict[str, PairedOutcome]:
    recs: dict[str, dict] = {}
    for name in ("p3-07/standard/records.jsonl", "p3-07-retry/standard/records.jsonl"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["qid"].startswith(PUBLIC):
                recs[r["qid"]] = r  # the re-run replaces the timed-out first answer
    scores: dict[str, int | None] = {}
    for name in ("p3-07/judge.jsonl", "p3-07-retry/judge.jsonl"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            if j["system"] == "standard" and j.get("run", 1) == 1 and j["qid"].startswith(PUBLIC):
                scores[j["qid"]] = j.get("score")
    support = _support(root / "p3-07/support/lines.jsonl", "standard")
    retried = {
        json.loads(x)["qid"]
        for x in (root / "p3-07-retry/standard/records.jsonl").read_text("utf-8").splitlines()
    }
    out = {}
    for qid, r in recs.items():
        rec_s = None if qid in retried else support.get(qid)  # the re-run was not checked
        chat = (r.get("usage_by_kind") or {}).get("chat") or r.get("usage") or {}
        out[qid] = PairedOutcome(
            qid=qid,
            score=scores.get(qid),
            found=bool(r.get("found")),
            fully_cited=bool(r.get("found"))
            and r.get("sentences", 0) > 0
            and r.get("unsupported_sentences", 0) == 0,
            citation_recall=rec_s[0] if rec_s else None,
            supported_pairs=rec_s[1] if rec_s else 0,
            judged_pairs=rec_s[2] if rec_s else 0,
            generation_ms=float((r.get("steps_ms") or {}).get("generation") or 0.0),
            cny=float(chat.get("cny") or 0.0),
        )
    return out


def candidate(answers: str, judge: str, support_dir: str | None) -> dict[str, PairedOutcome]:
    run = load_answer_run(answers)
    jr = load_judge_run(judge)
    if jr.answer_run_id != run.run_id:
        raise SystemExit(f"judge run {judge} is not of {run.run_id}")
    scores = {s.qid: s.score for s in jr.scores if s.scale == "1-3"}
    support = _support(Path(support_dir) / "lines.jsonl", None) if support_dir else {}
    out = {}
    for q in run.questions:
        s = support.get(q.qid)
        chat = q.usage_by_kind.get("chat") or q.usage
        out[q.qid] = PairedOutcome(
            qid=q.qid,
            score=scores.get(q.qid),
            found=q.found,
            fully_cited=q.found and q.sentences > 0 and q.unsupported_sentences == 0,
            citation_recall=s[0] if s else None,
            supported_pairs=s[1] if s else 0,
            judged_pairs=s[2] if s else 0,
            generation_ms=q.generation_ms,
            cny=chat.cny,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p307", default="C:/workspace/rag-platform-data/p3-07")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--exclude", default="", help="question ids left out (a sensitivity check)")
    p.add_argument("--out")
    args = p.parse_args()
    base = baseline(Path(args.p307))
    excluded = {q for q in args.exclude.split(",") if q}
    texts = []
    for item in args.candidate:
        name, _, rest = item.partition("=")
        parts = rest.split(",")
        cand = candidate(parts[0], parts[1], parts[2] if len(parts) > 2 else None)
        b = {q: o for q, o in base.items() if q not in excluded}
        c = {q: o for q, o in cand.items() if q not in excluded}
        table = paired_answer_table(b, c)
        texts.append(
            render_paired_answers(table, base_name="deepseek-v4-flash", candidate_name=name)
        )
    text = "\n".join(texts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
