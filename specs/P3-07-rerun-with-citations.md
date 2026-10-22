# P3-07 Full 150-question re-run with citation check

Status: in progress
Phase: P3
Depends on: P3-05 (harness, questions, old records), P3-06 (graph fixes), P1-06 (citation check level 2), P1-04 (answer judge)
Design sections: s13-5, s11-2, s14-4, s14-1, s14-3
Calls real models: yes (budget for this spec: at most 3.5 CNY; ledger stop line 33 CNY)

## Goal
The P3-05 comparison re-run on the same 150 questions with today's `main`: standard RAG (A) and
`ask_docs` (B, P3-06 settings), one pass each, records keeping answer sentences and cited chunks;
scored by the P1-04 judge and the P1-06 citation check; paired A-vs-B tables and a now-vs-P3-05
table per system, with machine load recorded.

## Plan and metrics (fixed before the run)
- Questions: `load_items("hf_docs")`, 150 (single_hop 65, two_hop 20, colloquial 15, no_answer 50).
- Run: `python -m rag.eval.agentic run --out data/reports/p3-07`; one question at a time, A then B.
  Default settings, no agent default changed. Pilot: first 10 questions of the mix (`--take` 4/2/2/2),
  then `project`; continue only if the projection of the whole spec is below 3.5 CNY.
- Judge: P1-04's unchanged, one pass first (pass 1 is the primary score, as in P3-05); a second pass
  only if budget remains. Correct = score 3; refusal on no-answer = no answer given (clean refusal
  without errors listed too). Exact McNemar per type, Wilson intervals.
- Citation check: P1-06's `rag.eval.support run --source p3-05 --path data/reports/p3-07`, one pass;
  precision of pairs, recall of sentences, 'every sentence supported' per system and type; exact
  McNemar on 'every sentence supported' over questions both systems answered.
- Latency median and P95 per system and type; cost per question; CPU `LoadPercentage` every 30 s
  before and during the run into a log file. Embedding ledger rows are now in each question's usage
  (P3-05 had none), so per-question cost is not directly comparable: report chat-only cost beside it.
- Now vs P3-05: same questions, same system, paired by qid: correct per type with McNemar, refusal,
  cost, latency. One pass each; P3-06's two-run agreement (20/23 judge scores equal) is the noise.
- `round2_on_not_enough` off: only if budget remains, on the questions whose B run reached round 2.

## Interface
- `rag.eval.agentic.runs.compare_runs(before, after, before_lines, after_lines, system) -> str`
  (Markdown, no model) and CLI `python -m rag.eval.agentic compare-runs --before DIR --after DIR
  [--system S] [--md FILE]`.

## Acceptance criteria
- [ ] AC1 A graph record keeps answer sentences and cited chunks (ids among the answer's sources) and
  the P1-06 loader makes a judgeable item of both systems — `tests/unit/test_eval_agentic_rerun.py::test_agentic_record_keeps_sentence_citations`
- [ ] AC2 `compare_runs`: per type, correct counts before/after, improved/worsened and McNemar p on
  shared qids; refusal on no-answer; cost and latency median/P95 — `test_compare_runs_per_type`
- [ ] AC3 CLI `compare-runs` writes the Markdown — `test_compare_runs_cli`

Also (by running, `llm`): pilot, full run, judge, citation check, report in `docs/reports/P3-report.md` section 5.

## Out of scope
Changing agent code or defaults (bugs go to findings); human scores; judge prompt changes.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
