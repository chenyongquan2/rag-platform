# P3-07 Full 150-question re-run with citation check

Status: review
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
- `rag.eval.agentic.runs.compare_runs(before, after, before_lines, after_lines, *, system) -> RunComparison`
  (per-type `PairedBinary`, refusal, clean refusal, `RunCostRow`s; no model),
  `render_compare_runs(report: RunComparison) -> str` (Markdown) and CLI `python -m rag.eval.agentic compare-runs --before DIR --after DIR
  [--system S] [--md FILE]`.
- `run --kb-override SECTION.KEY=VALUE` (eval only; e.g. `agent.round2_on_not_enough=false`); every `run` appends a line to `DIR/runs.jsonl` (start time, system, kb, overrides, deadline, question count) and prints the overrides.

## Acceptance criteria
- [x] AC1 A graph record keeps answer sentences and cited chunks (ids among the answer's sources) and
  the P1-06 loader makes a judgeable item of both systems — `tests/unit/test_eval_agentic_rerun.py::test_agentic_record_keeps_sentence_citations`
- [x] AC2 `compare_runs`: per type, correct counts before/after, improved/worsened and McNemar p on
  shared qids; refusal on no-answer; cost and latency median/P95 — `test_compare_runs_per_type`
- [x] AC3 CLI `compare-runs` writes the Markdown — `test_compare_runs_cli`
- [x] AC4 `--kb-override` sets only the named key on a copy, re-validates, rejects unknown keys — `test_kb_override`; `run --kb-override` passes the changed config to the runner and writes it to `runs.jsonl` — `tests/unit/test_eval_agentic_cli.py::test_run_applies_and_records_a_kb_override`

Also (by running, `llm`): pilot, full run, judge, citation check, report in `docs/reports/P3-report.md` section 5.

## Out of scope
Changing agent code or defaults (bugs go to findings); human scores; judge prompt changes.

## Handoff
- Files: `src/rag/eval/agentic/runs.py` (new), `src/rag/eval/agentic/cli.py` (`compare-runs`, `--kb-override`), `tests/unit/test_eval_agentic_rerun.py` (4 tests), `docs/reports/P3-report.md` section 5, `docs/decisions.md` (P3-07). Raw records in git-ignored `data/reports/p3-07{,-retry,-round2off}/` (copy in `C:/workspace/rag-platform-data/p3-07/`).
- Decisions: ES and Redis containers were stopped and were started with `docker start` (no recreate); A's cold pilot (8/10 rerank timeouts right after the start) set aside and A re-run; judge and citation check one pass each (budget); the 4 A questions lost to `llm_timeout` re-asked separately as a reference only; `round2_on_not_enough=false` only on the 24 questions it changes. AC1 passed at once: the runner already filled both fields for both systems (characterization test).
- Results: answerable correct A 89/100, B 92/100 (p = 0.45); all 4 single-hop gains of B are A's answer-model timeouts; without the 7 questions either run broke, A 87/93, B 88/93. Refusal 50/50 both. Citation precision 89.7% vs 92.4%, recall 96.0% vs 96.4%, every sentence supported 91.3% vs 92.5% (paired p = 1.0). B 0.0092 vs A 0.0039 CNY per question; P50 9.2 vs 7.2 s, P95 25.9 vs 13.2 s. B vs P3-05 B: errors 5 -> 0, cost 0.0162 -> 0.0092, P95 20.9 -> 25.9 s, correct 93 -> 92. Round 2 off on 24 no-answer questions: refusal unchanged, median 20.0 -> 2.8 s.
- Open issues: (1) an answer cancelled by the 30 s deadline is reported as `not_found` without an error (two_hop:04, :14); (2) `llm_timeout` answers are `not_found`, `--retry-errors` misses them; (3) warm-up needed after an ES start; (4) `test_cost.py::test_cli_summary_prints_totals` reads `.env`; (5) no-answer questions still the slow, costly class. Spend 3.04 CNY of 3.5 (ledger 26.58 -> 29.62).
- Review rounds: 1 (CHANGES, small): report numbers and finding wording corrected, overrides recorded in `runs.jsonl`, two test gaps closed (CLI override mutation, clean refusal on the after side), verdicts in before/after tables say "after", spec interface fixed. No paid calls.