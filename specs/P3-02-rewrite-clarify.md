# P3-02 Rewrite and clarify nodes of the ask_docs graph (route model fallback, interrupt and resume)

Status: review
Phase: P3
Depends on: P3-01 (graph skeleton, hooks, sessions, token cap), P1-01 (chat factory, structured output)
Design sections: s13-3, s8-1, s8-2, s8-4, s11-1, s16-1, s16-8 (s13-4 only to know what P3-03 adds)
Calls real models: no for unit tests (scripted fakes); one `llm` test with the `cheap` profile (budget 0.05 CNY)

## Goal
The stub hooks of P3-01 get behaviour: `rewrite` turns a follow-up into a standalone question, writes the
retrieval query as a declarative statement and splits a two-hop question (cheap model, structured output,
rules first, original question as the fallback); `route` keeps rules first and asks the cheap model only
when rules cannot decide; `clarify` stops the run with LangGraph `interrupt`, `ask_docs` returns
`need_clarification` and an opaque `resume_token`, and a second call with `resume_token` and `answer`
continues the same checkpoint thread.

## Interface
- `rag.agent.rules`: `normalise`, `protected_tokens`, `missing_tokens`, `is_question_form`, `looks_like_follow_up`,
  `clarity`, `multi_hop_hint`, `default_clarification` (pure functions).
- `rag.agent.rewrite`: `RewritePlan`, `QueryRewriter` (the default `rewrite` hook; `first_round`, `next_round(state, ctx, missing)`).
- `rag.agent.routing`: `RouteJudgement`, `route_with_model` (the default `route_fallback` hook).
- `rag.agent.helper`: `ask_helper(ctx, schema, messages)` (one guarded structured call of the cheap model: timeout,
  token cap, `HelperFailed(reason)`), `history_lines`.
- `rag.agent.resume`: `ResumeStore` (memory, sqlite), `ResumeTokenInvalid`; `rag.factories.make_resume_store`;
  `rag.agent.sqlite_db.init_agent_db` (both tables of the session file).
- `rag.agent.decisions`: `RouteDecision` (gains `clarity`, `multi_hop`, `two_hop`, `clarify_question`, `fallback`),
  `RewriteOutcome` (gains `fallback`, `used_model`); `AgentHooks(decide_route, route_fallback, clarify, rewrite, judge)`.
- `AgentRuntime.ask(..., helper_model=None, resume_token=None, answer=None)`; same on `ask_docs`.
- `AgentConfig` (per KB): `rewrite`, `declarative`, `max_sub_questions`, `clarify`, `route_model`.
  Settings: `agent_rewrite_profile`, `agent_rewrite_timeout_ms`, `agent_rewrite_history_turns`,
  `agent_rewrite_history_chars`, `agent_resume_ttl_s`.
- `AskResult`: `standalone_question`, `search_queries`, `fallbacks`, `clarified`; `ErrorCategory.INVALID_RESUME_TOKEN`.
- `SearchService.ask_docs(..., resume_token=None, answer=None)`, `SearchService(helper_model=...)`; MCP tool and
  `POST /v1/ask_docs` get `resume_token`, `answer`.

## Acceptance criteria
Unit tests in `tests/unit/test_agent_{rules,rewrite,route,clarify,resume,config}.py`, `test_serve_ask_*.py`.
- [x] AC1 Rules: normalisation; identifiers, code spans, versions, flags, paths and quoted messages are found, plain words are not — `test_agent_rules.py::test_protected_tokens`, `::test_plain_words_are_not_protected`, `::test_normalise`, `::test_missing_tokens_is_exact_and_case_sensitive`
- [x] AC2 Rules: question form, follow-up, clarity, two-hop hint in English and Chinese — `test_agent_rules.py::test_question_and_follow_up_forms`, `::test_clarity_and_multi_hop_signals`, `::test_default_clarification_is_in_the_question_language`
- [x] AC3 Questions the rules settle make zero model calls (statement-form query, greeting, feature off) — `test_agent_rewrite.py::test_rule_only_questions_make_no_model_call`
- [x] AC4 A follow-up becomes a standalone question with the history in the prompt — `::test_follow_up_becomes_standalone_with_history`
- [x] AC5 Retrieval gets the declarative statement, the answer gets the question — `::test_declarative_query_is_retrieved_and_the_question_is_answered`
- [x] AC6 A two-hop question is split, one search per sub-question, capped by `max_sub_questions` — `::test_two_hop_question_is_split_and_capped`
- [x] AC7 A rewrite that loses an identifier, version or code token, or translates the question, is replaced by the original query, with a flag — `::test_lost_identifiers_fall_back_to_the_original_query`, `::test_rewrite_keeps_the_questions_language`
- [x] AC8 Model error, timeout, garbage output, no model: the original question is searched, `degraded` and `fallbacks` are set, the request is answered — `::test_model_failures_fall_back_and_flag_degraded`, `::test_no_model_available_falls_back_too`
- [x] AC9 Token cap: the rewrite is refused before the call, the run still searches and answers — `::test_token_cap_before_rewrite_falls_back_and_still_answers`
- [x] AC10 The round-2 entry takes what was missing (P3-03 fills it in; round 1 behaviour only) — `::test_next_round_signature_takes_what_was_missing`
- [x] AC11 Route: rules alone decide clear and vague questions (no model); `undecided` and `maybe` ask the cheap model; its failure keeps the rules' view with a flag — `test_agent_route.py::test_route_rules_decide_without_a_model`, `::test_route_asks_the_model_only_when_undecided`, `::test_route_decides_two_hop_with_the_model_when_rules_say_maybe`, `::test_route_model_failure_keeps_the_rules_decision`, `::test_a_hook_decision_is_honoured_and_the_rules_are_a_default`
- [x] AC12 Clarify with `interrupt`: need_clarification with the question and an opaque `resume_token`; resume with the answer ends answered through `ask_docs` — `test_agent_clarify.py::test_clarification_round_trip`, `::test_clarification_with_rewrite_uses_the_answer_in_the_standalone_question`
- [x] AC13 One clarification per request: a vague answer is searched as it is — `::test_second_clarification_is_not_asked`
- [x] AC14 Resume checks: wrong caller, KB or session, unknown, reused, expired, finished run, thread of another session, malformed are all rejected, never a new question; a foreign caller does not burn the token; two resumes at once continue once; store semantics for memory and SQLite — `::test_resume_token_is_checked`, `::test_foreign_caller_does_not_consume_the_token`, `::test_token_of_a_finished_thread_is_rejected`, `::test_two_resumes_at_once_continue_the_run_only_once`, `test_agent_resume.py::*`
- [x] AC15 Session lock: released at the interrupt (other requests proceed), taken again on resume (waits, counts against the deadline, a resume that gave up keeps its token) — `test_agent_clarify.py::test_lock_is_not_held_across_the_interrupt`, `::test_resume_waits_for_the_session_lock`, `::test_a_resume_that_gave_up_waiting_keeps_the_token`
- [x] AC16 History is written when the run completes, with the clarifying exchange as two turns; nothing at the interrupt — `::test_history_after_resume_has_the_clarifying_exchange`
- [x] AC17 SQLite checkpoint and token store: resume works from a new runtime (restart), a used token stays used, the file holds no token; the token cap counts both calls; abandoned questions are cleaned up — `::test_resume_after_restart_with_sqlite`, `::test_token_budget_spans_both_calls`, `::test_abandoned_questions_are_cleaned_up`
- [x] AC18 Service, HTTP and MCP: `resume_token` and `answer` inputs, `invalid_resume_token` (400) with fixed text, no token in the business record — `test_serve_ask_service.py::test_resume_arguments_and_errors`, `test_serve_ask_transports.py::test_clarification_over_mcp_and_http`, `::test_mcp_ask_docs`
- [x] AC19 Config: five KB keys and five settings, validated; `hf_docs.yaml` sets them — `test_agent_config.py::test_rewrite_and_clarify_config`, `::test_hf_docs_sets_agent_keys`
- [x] AC20 Spans and business record carry the rewritten query, fallback flags, tokens, model calls and CNY of the new steps — `test_agent_rewrite.py::test_spans_and_record_carry_the_rewrite`, `::test_model_calls_and_cost_of_the_step_are_on_its_span`, `test_serve_ask_service.py::test_rewrite_in_the_business_record`
- [x] AC21 One `llm` test, never run by default: the real `cheap` model on five questions (follow-up, identifiers, two-hop, Chinese, undecided), at most 0.05 CNY — `tests/llm/test_rewrite_gateway.py::test_real_cheap_model_rewrites_and_routes`

## Out of scope
Judging sufficiency, re-asking with another wording, multi-query and HyDE (P3-03: only the round-2 entry exists);
translation between languages (s8-4, P4); acronym and synonym tables; choosing a KB; SSE (P3-04); MCP elicitation;
cleaning the checkpoint threads of ordinary runs; an `llm` run over many questions (P3-05).

## Handoff (filled at the end)
- Files: new `src/rag/agent/{rules,rewrite,routing,helper,resume,decisions,sqlite_db}.py`; changed `src/rag/agent/{hooks,nodes,graph,runtime,result,state,context,session,__init__}.py`, `src/rag/types.py`, `src/rag/factories.py` (`make_resume_store`), `src/rag/config/{kb,settings}.py`, `kb/public/hf_docs.yaml`, `src/rag/observe/attributes.py`, `src/rag/serve/{service,schema,errors,http_app,mcp_server}.py`, `src/rag/testing/agent.py` (`structured_reply`, `RaisingChatModel`, `SlowChatModel`, `kb_config(**agent)`), `README.md`, `.env.example`, `docs/decisions.md`. Tests: new `test_agent_{rules,rewrite,route,clarify,resume}.py`, `tests/llm/test_rewrite_gateway.py`; extended `test_agent_config.py`, `test_serve_ask_{service,transports}.py`. 918 unit tests pass (850 before), 46 integration tests pass, one new `llm` test (run once: 0.0138 CNY for a full pass; 0.041 CNY in total over three runs while the test was being fixed).
- Existing tests changed (spec change): `test_agent_graph.py` two assertions (`resume_token` is no longer the thread id), `test_agent_config.py::test_hf_docs_sets_agent_keys` (the yaml sets five more keys), `test_serve_ask_transports.py::test_mcp_ask_docs` (two more tool inputs).
- Decisions (docs/decisions.md, 2026-10-20 P3-02): rules first and a two-check gate on the model's rewrite (identifiers verbatim, language kept); failures fall back to the original question with `degraded` and `fallbacks` (no new category); route signals from rules, small model only for undecided / maybe, features off by default per KB and on in `hf_docs.yaml`; `resume_token` is a random token whose hash is stored with caller, KB, session, thread, once-only and 30-minute expiry, checked in one transaction, same `invalid_resume_token` (400) for every refusal (the design said token = thread id); lock released at the interrupt and taken again on resume, history written at completion as two turns, token cap spans both calls; `question` stays required but unused with a token; real-model numbers.
- Open issues: route rules and stop words are tuned on hand-written examples, not measured on the question set (P3-05); the cheap model asked back on "this model" with no history, stricter than expected (P3-05 should count how many asked-back questions could have been answered); checkpoint threads of ordinary runs are never cleaned (only expired clarification threads are); the default gateway client is bound to the event loop it first ran on (one loop per service process, as before); rewrite plus route calls add 2 to 8 s for questions the rules cannot settle, against a 15 s P95 target (not measured end to end with a real retrieval); the model may add facts that are not in the question (`private=True` was added correctly in one case): the identifier check only guards what the question contained; design 13.3 and 16.8 in `docs/design` still say token = thread id and have no `invalid_resume_token` row (not edited; recorded in decisions).
- Review rounds:
