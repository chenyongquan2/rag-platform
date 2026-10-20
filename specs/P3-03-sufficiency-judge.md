# P3-03 Sufficiency judge, correction loop, judge-accuracy script, hard questions

Status: in progress
Phase: P3
Depends on: P3-01 (graph, hooks, limits), P3-02 (rewrite module, `next_round`, helper model), P2-02 (threshold, no-answer sets), P1-05 (synthetic sets)
Design sections: s13-4 (main), s13-3, s8-2, s10-2, s14-2, s14-4, s16-8, s11-1
Calls real models: yes, the `cheap` model in the judge-accuracy run (budget 2.0 CNY, pilot of 10 questions first); one `llm` test (under 0.02 CNY)

## Goal
The `judge` hook of the `ask_docs` graph decides after every round whether what was found is enough (three
tiers, rules first, cheap model second), and the graph corrects a partly or not enough result: round 2 asks
the missing sub-questions again with several phrasings (multi-query), round 3 with a hypothetical answer
passage (HyDE); three rounds without anything above the threshold end as a real `not_found` with the closest
chunks attached. A script measures how often the judge agrees with a label made from the question sets, and
40 hand-written hard questions (20 two-hop, 20 no-answer) are added to the eval data.

## Interface
- `rag.agent.judge`: `SufficiencyJudgement`, `GroundsVerdict` (structured output), `SufficiencyJudge` (the default
  `judge` hook), `judge_rounds(ctx, rounds, *, open_owners=None) -> JudgeOutcome` (also used by the script),
  `above_threshold(chunks, threshold)`.
- `rag.agent.decisions`: `JudgeOutcome(verdict: enough | partly_enough | not_enough, missing, fallback, used_model)`;
  `RewriteOutcome` gains `queries` (tuple of `SearchQuery`) and `exhausted`. A hook may still return the P3-01
  strings `sufficient` / `insufficient` (read as enough / partly enough with every sub-question missing).
- `rag.agent.state`: `SearchQuery(text, owner, kind)` (kind original | multi_query | hyde); `RetrievalRound` gains
  `query` and `below_threshold`; `AgentState` gains `search_plan`, `sufficiency`.
- `rag.agent.correction`: `corrective_queries(state, ctx, missing) -> RewriteOutcome` (behind
  `QueryRewriter.next_round`): `MultiQueryPlan`, `HydePlan`.
- `rag.agent.helper.ask_helper(ctx, schema, messages, *, timeout_ms=None, model=None)`.
- `AgentConfig`: `judge` (default false), `multi_queries` (3), `hyde` (true). Settings: `agent_judge_profile`
  (`cheap`), `agent_judge_timeout_ms`, `agent_judge_context_tokens`, `agent_round_min_ms`.
- `ask_docs(..., judge_model=None)`; `AgentContext.judge_model`.
- `AskResult`: `sufficiency`, `missing`, `nearest` (closest chunks of a not-found result); `StopReason` gains
  `no_new_query`.
- `python -m rag.eval.judge_accuracy run|report` (`rag.eval.judge_accuracy.{records,metrics,runner,cli}`).
- Data: `evals/datasets/hf_docs/two_hop.jsonl` (20), `no_answer_hard.jsonl` (20).

## Acceptance criteria
Unit tests in `tests/unit/test_agent_judge.py` (J), `test_agent_correction.py` (C), `test_agent_config.py` (G),
`test_eval_judge_accuracy.py` (E), `test_hard_questions_data.py` (D). Fakes only.
- [ ] AC1 ...

## Out of scope
Switching knowledge base after a failed round (the graph serves one KB per call; documented extension point in
`correction.py`); step-back prompts (s8-2); per-caller budgets (P8); the end-to-end Agentic RAG eval (P3-05).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
