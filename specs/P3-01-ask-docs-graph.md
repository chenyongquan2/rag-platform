# P3-01 ask_docs graph skeleton: nodes, state, sessions, limits, MCP tool and HTTP endpoint

Status: review
Phase: P3
Depends on: P1-02 (`rag.answer`), P0-09 (search service, MCP, HTTP), P0-11 (spans, business records), P1-01 (chat factory)
Design sections: s13-3, s13-2, s4-2, s16-1, s16-8, s16-7, s11-1, s11-2, s10-3, s8-1, s16-4 (token cap), s12-1 (history and cache)
Calls real models: no (fake chat models and fake retrieval only; spend 0 CNY)

## Goal
`rag.agent.ask_docs` runs our own LangGraph `StateGraph` (not `create_agent`): `route` (rules only) -> `rewrite`
(stub) -> `retrieve` -> `judge` (stub) -> `answer` -> `verify`, with a `clarify` stub. One run is one
checkpoint thread `"{session_id}:{run_id}"` (a run id the server makes; the caller's request id is only for correlation); chat history is stored per caller, KB and session id; requests of one
session run one after another; the run has a recursion limit, a total deadline and a per-request token
cap. The result is exposed as the MCP tool `ask_docs` and `POST /v1/ask_docs` (non-streaming; SSE is P3-04).

## Interface
- `rag.agent`: `ask_docs(question, *, cfg, retriever, settings, runtime=None, session_id=None,
  request_id=None, caller_id="local", acl_groups=("public",), deadline_ms=None, model=None,
  filters=None, run_id=None) -> AskResult`;
  `AgentRuntime(settings, store, locks, checkpointers, hooks, clock)` (`from_settings` creates both SQLite files once, `ask`,
  `run_config`, `state_values`, `session_key(caller_id, kb, session_id)`);
  `AgentHooks(decide_route, clarify, rewrite, judge)` (P3-02 and P3-03 replace the stubs);
  `AgentState`, `RetrievalRound`, `HistoryTurn`, `thread_id(session_id, request_id)`;
  `TokenBudget`, `RequestTokenCapExceeded`, `guard_model`; `SessionStore`, `SessionLocks`;
  `AgentDeadlineExceeded` (waiting for the session lock outlived the deadline).
- `rag.types`: `AskResult` (answer, sentences, sources, status answered | not_found | need_clarification,
  found, rounds, timings_ms, degraded, rerank_timeout, coverage_unconfirmed, stop_reason, notice,
  tokens_used, used_history, cache_allowed, request_id, session_id, thread_id, resume_token=None);
  `ErrorCategory.AGENT_DEADLINE`, `TOKEN_CAP_REACHED`.
- `rag.factories`: `make_session_store(url, keep_turns)`, `make_session_locks(url)`,
  `make_checkpointer_source(url, serde)` (async SQLite saver by connection string, one connection
  per request).
- `KnowledgeBaseConfig.agent = AgentConfig(max_rounds)`; settings `agent_deadline_ms`,
  `agent_answer_reserve_ms`, `agent_recursion_limit`, `agent_token_cap`, `agent_answer_reserve_tokens`,
  `agent_history_turns`, `session_store_url`, `session_lock_url` (`checkpoint_url` exists).
- `SearchService.ask_docs(credentials, *, kb, question, session_id=None, deadline_ms=None, request_id=None)
  -> AskResponse` (async); MCP tool `ask_docs(kb, question, session_id, deadline_ms)`; `POST /v1/ask_docs`.

## Acceptance criteria
Unit tests in `tests/unit/test_agent_*.py` and `tests/unit/test_serve_ask_*.py` (fakes only, no `llm`, no `integration`).
- [x] AC1 Retrieval results accumulate by reducer across rounds of one request; `route` clears them first, so a second run on the same thread starts empty — `test_agent_graph.py::test_rounds_accumulate_and_route_clears_them`
- [x] AC2 `thread_id` is `session_id:run_id`; session ids with a colon, odd characters or the no-session name `-` are refused — `test_agent_state.py::test_thread_id_format`
- [x] AC3 Two requests of one session never see each other's chunks (own thread, own sources, only text turns carried as history); two runs with the same caller-supplied request id have separate threads, so a run cut by the deadline is rebuilt from its own chunks — `test_agent_graph.py::test_same_session_requests_do_not_share_chunks`, `test_agent_concurrency.py::test_runs_with_the_same_request_id_do_not_share_a_checkpoint_thread`
- [x] AC4 `route` is rules only: it keeps the requested KB, skips retrieval for a greeting (fixed reply, no retrieval, no model call) and calls no model — `test_agent_graph.py::test_route_rules_skip_retrieval_for_greetings`, `::test_route_calls_no_model`
- [x] AC5 Happy path: one retrieval per sub-question, answer with citations, `verify` copies degraded, rerank_timeout, index version, data generation and sets found; per-step timings — `test_agent_graph.py::test_ask_docs_happy_path`, `::test_verify_propagates_retrieval_flags`
- [x] AC6 Nothing above the threshold: status not_found, no model call; the chat model is only made when there is a fragment to answer from — `test_agent_graph.py::test_not_found_makes_no_model_call`, `::test_default_model_is_not_built_when_no_call_is_needed`
- [x] AC7 History is stored per session behind a connection string, last N turns kept; a call with history sets `used_history` and `cache_allowed=False` (no cache read or write) — `test_agent_session.py::test_session_store_keeps_last_turns`, `test_agent_graph.py::test_call_with_history_is_not_cache_eligible`
- [x] AC8 Same session runs one after another, different sessions overlap; file-lock variant holds a cross-process lock — `test_agent_session.py::test_same_session_runs_one_after_another`, `::test_different_sessions_run_in_parallel`, `::test_file_lock_is_held_while_running`; through `ask_docs`: one at a time, history in order, nothing left in the lock registry, different sessions overlap — `test_agent_concurrency.py::test_same_session_runs_one_at_a_time_through_ask_docs`, `::test_different_sessions_overlap_through_ask_docs`
- [x] AC9 Total deadline: past it the graph stops looping and answers from the chunks already found with `coverage_unconfirmed`, notice and `agent_deadline`; a run cut off mid-node returns the collected chunks only; waiting for the session lock counts against the deadline — `test_agent_limits.py::test_deadline_answers_from_collected_chunks`, `::test_hard_deadline_returns_collected_chunks`, `::test_waiting_for_the_lock_counts_against_the_deadline`
- [x] AC10 Token cap: tokens are counted (usage or estimate); once spent, further model calls are refused (also the answer's retry), `token_cap_reached` — `test_agent_limits.py::test_token_cap_stops_further_model_calls`, `::test_tokens_counted_from_usage_or_estimate`, `::test_gateway_usage_is_counted_against_the_cap`
- [x] AC11 `recursion_limit` is set explicitly on every run and ends a runaway loop with the collected chunks — `test_agent_limits.py::test_recursion_limit_is_set_and_enforced`
- [x] AC12 Stub hooks for rewrite, clarify, judge are wired: replacing them changes behaviour with no graph change (rewrite sub-questions, judge loops to max_rounds, clarify ends the run) — `test_agent_graph.py::test_extension_points_are_wired`
- [x] AC13 Settings and KB config: explicit keys, validation, `hf_docs.yaml` sets `agent` — `test_agent_config.py::test_agent_settings_defaults_and_validation`, `::test_hf_docs_sets_agent_keys`
- [x] AC14 Factories by connection string: session store (sqlite, memory), async checkpointer persists state per thread — `test_agent_session.py::test_session_store_factory`, `test_agent_graph.py::test_sqlite_checkpointer_persists_state_per_thread`
- [x] AC15 Service: same caller and ACL handling as search_docs (hidden KB = kb_not_found, caller groups reach retrieval, arguments checked), error categories mapped, business record and node spans — `test_serve_ask_service.py::test_ask_docs_permissions_and_arguments`, `::test_ask_docs_errors_map_to_categories`, `::test_ask_docs_spans_and_business_record`
- [x] AC16 MCP tool `ask_docs` (in-process SDK client): listed with schema, structured result equals the HTTP JSON, errors are category and request id only — `test_serve_ask_transports.py::test_mcp_ask_docs`
- [x] AC17 HTTP `POST /v1/ask_docs`: happy path, `X-Request-Id`, rejected body, error statuses — `test_serve_ask_transports.py::test_http_ask_docs`
- [x] AC18 The session lock is released after an exception, a cancellation and a deadline cut — `test_agent_concurrency.py::test_lock_is_released_however_a_request_ends`
- [x] AC19 History and lock are keyed by caller, KB and session id (hash), not the session id alone — `test_agent_concurrency.py::test_history_is_not_shared_between_callers_or_knowledge_bases`, `test_serve_ask_service.py::test_history_is_kept_per_caller_and_knowledge_base`
- [x] AC20 Cold databases: twelve concurrent requests on fresh SQLite files (checkpoint and sessions) all succeed; both are created once when the runtime is built — `test_agent_concurrency.py::test_cold_sqlite_databases_survive_concurrent_requests`
- [x] AC21 `cache_allowed` is false for any incomplete or flagged result and for history; `used_history` comes from the loaded history even when the run is cut before its first checkpoint; each checkpoint is written before the next step (`durability="sync"`) — `test_agent_limits.py::test_cut_run_reports_the_history_it_was_given`, `::test_checkpoint_is_written_before_the_next_step_starts`, `test_agent_graph.py::test_verify_propagates_retrieval_flags`
- [x] AC22 The route hook may be async; a run that stops at an `interrupt` returns need_clarification — `test_agent_graph.py::test_route_hook_may_be_async`, `::test_interrupt_in_a_hook_returns_need_clarification`

## Out of scope
Rewrite, clarify (`interrupt`, `resume_token`), sufficiency judge and re-asking (P3-02, P3-03: the hooks are stubs);
choosing a KB when the caller gives none (s8-3); SSE and the page (P3-04); the exact cache (P5, only the
`cache_allowed` flag); Postgres checkpointer; per-caller budgets (P8); evaluation (P3-05); `llm` tests.

## Handoff (filled at the end)
- Files: `src/rag/agent/{__init__,state,budget,session,hooks,context,result,nodes,graph,runtime}.py`, `src/rag/testing/agent.py` (fakes), `src/rag/types.py` (`AskResult`, `StopReason`, two categories), `src/rag/factories.py` (`make_checkpointer_source`, `make_session_store`, `make_session_locks`), `src/rag/config/{kb,settings,__init__}.py`, `kb/public/hf_docs.yaml`, `src/rag/retrieve/{observe,pipeline}.py` (shared retrieval metrics; `agent` left out of the config version), `src/rag/observe/{attributes,spans}.py`, `src/rag/serve/{service,schema,errors,http_app,mcp_server}.py`, `tests/unit/test_agent_{config,state,session,graph,limits}.py`, `tests/unit/test_serve_ask_{service,transports}.py`, `README.md`, `.env.example`, `docs/decisions.md`. 31 new unit tests, no `llm`, no `integration` test (SQLite and file locks run on tmp dirs); three existing tests updated for the fourth tool and the new route.
- Decisions (docs/decisions.md, 2026-10-20 P3-01): deadline 30 s with 12 s kept for the answer, soft (between rounds) and hard (cancel, chunks only); token cap 20,000 with 6,000 kept for the answer, refusal as a `BudgetExceeded` subclass; one SQLite connection per request and a serializer allow-list; session lock in process by default, file lock through a helper thread; KB chosen by the caller, route skips only greetings; `max_rounds` reached answers with `coverage_unconfirmed` (P3-03 turns it into "not found"); `status` has the three values of s16-1; history keeps text only; `agent_deadline` also for the recursion limit; `agent` not in the config version.
- Open issues: P3-02 must still add the resume entry point: `AgentRuntime._run` now returns need_clarification when a run stops at `interrupt` (its `result` key is missing), but nothing calls `Command(resume=...)` yet, and `resume_token` is the checkpoint thread (`session_id:run_id`), so resuming means passing that run id as `run_id` and rebuilding the context; nothing measured against real models or real Elasticsearch (fakes only; the graph over a real `SearchPipeline` is untested until P3-05); the stub hooks make exactly one round, so the loop, the guards and the clarify edge are tested only with test hooks; the search thread of a cancelled run keeps running until its own deadline (a blocking call cannot be cancelled); a run cut off by the hard deadline returns the chunks without an answer text; token defaults are estimates; `resume_token` and `filters` are not tool arguments yet.
- Review rounds: 1 (CHANGES, fixed in the second round: cold-database race, thread per run, session key, concurrency tests, async route hook, interrupt tolerance, cache flags).
