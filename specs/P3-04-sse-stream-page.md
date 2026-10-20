# P3-04 SSE streaming endpoint and the single-page Q&A web page

Status: review
Phase: P3
Depends on: P3-01 (graph, runtime, lock, deadline, token cap), P3-02 (resume token), P3-03 (judge, HyDE)
Design sections: s16-9 (main), s16-1, s16-8, s13-3, s11-1, s11-2, s15-1
Calls real models: no (fake models everywhere; one optional real-gateway latency run, at most 0.05 CNY)

## Goal
`POST /v1/ask_docs/stream` answers the same request as `POST /v1/ask_docs` but as Server-Sent Events: the
page sees progress after the first node, the answer text while it is written, then the checked answer. The
rag-api HTTP app also serves one static page at `GET /` (HTML, CSS and JS files, no build step, no
framework, no outside fetch) that uses it: question box, knowledge-base list, progress line, streamed
answer, clickable `[n]` markers that show the source passage, clarification prompt, cancel button, error
display and the two feedback buttons (`POST /v1/feedback`). Closing the page or pressing cancel cancels the
graph run.

## Interface
Wire format (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`,
`X-Request-Id`). One event is `event: <name>\ndata: <one line of JSON>\n\n`; a line starting with `:` is a
heartbeat comment sent every `sse_heartbeat_s` seconds while nothing else is sent.

| event | data |
|---|---|
| `progress` | `{node, phase, round?, ms?, ...}`: `node` is `queued`, `route`, `clarify`, `rewrite`, `retrieve`, `judge`, `answer` or `verify`; `phase` is `waiting` (queued), `start`, `end` or `retry` (answer: the first attempt cited nothing, the text so far is void; `attempt`). `end` carries `ms` and, per node, `retrieve` (route), `chunks` and `chunks_total` (retrieve), `queries` (rewrite), `verdict` (judge), `found` (answer). No query text. |
| `token` | `{text, attempt}`: a piece of the answer text as the model wrote it (markers like `[1]` included), only from the `answer` node |
| `answer` | the `POST /v1/ask_docs` response body (`AskResponse`): sentences with citations, `sources`, `citation_check.removed_markers`, `found`, `sufficiency`, `degraded`, `fallbacks`, `stop_reason`, ... |
| `need_clarification` | `{request_id, question, resume_token, session_id}` (instead of `answer`) |
| `error` | `{error, request_id, message}`: category and fixed sentence only; terminal, no `done` follows |
| `done` | `{request_id, status, rounds, degraded, rerank_timeout, coverage_unconfirmed, stop_reason, tokens_used, timings_ms}`; `timings_ms` adds `first_progress` and `first_token` (ms since the request came in) |

- `rag.agent.events`: `StreamEmit`, `progress_event(...)`, the `streamed` node decorator (writes `progress`
  with LangGraph's stream writer); `answer_node` writes `token` events from `stream_answer`.
- `AgentRuntime.ask(..., emit=None)`: one run implementation for `ask` and the stream; `emit(name, data)` is
  called for every custom event plus `queued`. `SessionLocks.busy(key)`.
- `SearchService.ask_docs_stream(credentials, *, kb, question, session_id, deadline_ms, request_id,
  resume_token, answer, heartbeat_s) -> AsyncIterator[StreamEvent | Heartbeat]` (raises `ServiceError` before
  its first item for every failure that happens before streaming starts); `SearchService.feedback(...)`;
  `SearchService.cancel_streams()` (shutdown).
- `rag.serve.stream`: `StreamEvent(name, data)`, `Heartbeat`, `encode(item) -> bytes`.
- HTTP: `POST /v1/ask_docs/stream` (body of `/v1/ask_docs`), `POST /v1/feedback {request_id, useful,
  comment?}`, `GET /` (page), `GET /ui/app.js`, `GET /ui/app.css`; the existing `GET /v1/knowledge_bases`
  feeds the list.
- Settings: `sse_heartbeat_s` (10.0). `ErrorCategory.CLIENT_CANCELLED` (`client_cancelled`).
- `python -m rag.serve http --dev-fake`: the same app over fake retrieval and a slow fake model (no gateway,
  no Docker), for the page and for latency numbers.
- Files: `src/rag/serve/web/{index.html,sse.js,render.js,app.js,app.css}`; `scripts/stream_latency.py` times a stream from the client.

## Acceptance criteria
Tests: `tests/unit/test_agent_stream.py` (A), `test_serve_stream.py` (S), `test_serve_stream_cancel.py` (C, a
real uvicorn server on an ephemeral port), `test_serve_web.py` (W; runs `tests/web/ui_test.js` with node and
is skipped without node). Fakes only. Names are relative to the file named first in each group.
- [x] AC1 Every node writes `progress` start and end with the round; retrieve says how many chunks it found, judge its verdict — `test_agent_stream.py::test_progress_events_for_every_node_with_round_and_chunks`
- [x] AC2 `token` events carry the answer node's text only; rewrite, route, judge and HyDE model text is never streamed — `test_agent_stream.py::test_token_events_come_only_from_the_answer_node`, `::test_hyde_and_judge_text_never_reaches_a_token_event`
- [x] AC3 A retried answer attempt is marked (`progress` phase `retry`, `token.attempt`) — `test_agent_stream.py::test_a_retried_answer_attempt_is_marked`
- [x] AC4 `ask` and the stream share one run: same result either way, the non-stream path emits nothing — `test_agent_stream.py::test_the_streamed_run_gives_the_same_result_as_ask`
- [x] AC5 A request that waits for the session lock reports `queued`; one that does not, does not — `test_agent_stream.py::test_a_request_waiting_for_the_session_lock_reports_queued`
- [x] AC6 Event order: progress, then token, then `answer`, then `done`; every node's `end` after its `start` — `test_serve_stream.py::test_event_order_progress_token_answer_done`
- [x] AC7 `answer` has the `AskResponse` shape and lists the removed markers; the streamed text still holds them — `test_serve_stream.py::test_answer_event_carries_the_checked_answer_and_the_removed_markers`
- [x] AC8 `need_clarification` event, then a second stream with `resume_token` and `answer` finishes the run — `test_serve_stream.py::test_need_clarification_then_resume_over_the_stream`
- [x] AC9 `done` has timings, rounds, flags, tokens, `first_progress`, `first_token` — `test_serve_stream.py::test_done_event_has_timings_rounds_flags_and_token_usage`
- [x] AC10 Failures before the first byte (bad JSON, bad input, unknown and hidden KB, bad resume token, not ready) are JSON errors with the status of the non-stream endpoint — `test_serve_stream.py::test_errors_before_streaming_are_json_with_the_status_of_ask_docs`, `::test_a_service_that_is_not_ready_answers_503_before_any_byte`
- [x] AC11 A failure after the first byte is an `error` event (category and request id, no internals), then the stream ends — `test_serve_stream.py::test_a_failure_after_the_first_byte_is_an_error_event_then_close`
- [x] AC12 Heartbeat comment lines while the run is silent — `test_serve_stream.py::test_heartbeat_comment_lines_while_nothing_else_is_sent`
- [x] AC13 Response headers (`text/event-stream`, no-cache, `X-Accel-Buffering: no`, `X-Request-Id`) — `test_serve_stream.py::test_stream_response_headers`
- [x] AC14 Same caller and ACL handling as `ask_docs`: the caller's groups reach retrieval — `test_serve_stream.py::test_stream_uses_the_callers_access_like_ask_docs`
- [x] AC15 The total deadline in the middle of the stream ends with an `answer` event (chunks only, `stop_reason` deadline), not an error — `test_serve_stream.py::test_deadline_in_the_middle_of_the_stream_ends_with_an_answer_event`
- [x] AC16 Token cap, session history and business record work as in the non-stream call — `test_serve_stream.py::test_the_token_cap_still_applies_to_a_streamed_run`, `::test_a_streamed_run_writes_the_history_and_one_business_record`, `::test_cancel_streams_ends_the_runs_of_open_streams`
- [x] AC17 Client disconnect during the answer: no further model call, the session lock is released, spans closed — `test_serve_stream_cancel.py::test_disconnect_while_the_answer_is_written_stops_the_model`
- [x] AC18 Disconnect before the answer: no model call at all; checkpoint stays at the last finished node — `test_serve_stream_cancel.py::test_disconnect_during_retrieval_makes_no_model_call_and_keeps_the_checkpoint`
- [x] AC19 A cancelled request writes one business record flagged `client_cancelled` (with the node it was in), and the request span ends `cancelled` — `test_serve_stream_cancel.py::test_a_cancelled_request_is_recorded_as_client_cancelled`
- [x] AC20 A request that waited behind the cancelled one runs once the lock is free — `test_serve_stream_cancel.py::test_a_waiting_request_runs_after_the_first_one_disconnects`
- [x] AC21 `GET /` serves the page with CSP (`default-src 'self'`, no inline script or style), `nosniff`, no outside URL; `/ui/app.js` and `/ui/app.css` are served — `test_serve_web.py::test_root_serves_the_page_with_security_headers`, `::test_the_page_loads_nothing_from_outside_and_has_no_inline_code`
- [x] AC22 `GET /v1/knowledge_bases` (the page's list) shows only the KBs the caller may use — `test_serve_web.py::test_the_knowledge_base_list_hides_what_the_caller_may_not_use`
- [x] AC23 `POST /v1/feedback` writes one record line with the request id, the verdict and a capped comment; bad bodies are `invalid_request` — `test_serve_web.py::test_feedback_writes_one_record`, `::test_feedback_rejects_bad_bodies`
- [x] AC24 The page's SSE parser handles comments, split chunks, multi-line `data:` and unknown events — `test_serve_web.py::test_sse_parser_in_node`, `::test_the_pages_own_code_in_node` (`app.js` against a fake document and a scripted server)
- [x] AC25 Rendering uses text nodes only: `<img onerror>` and `<script>` in an answer and in a passage stay text; the code has no `innerHTML` — `test_serve_web.py::test_rendering_untrusted_text_creates_no_elements`, `::test_app_js_never_sets_html`
- [x] AC26 Settings and the dev fake mode: `sse_heartbeat_s` validated; `--dev-fake` builds a service that streams without any network — `test_serve_stream.py::test_stream_settings`, `::test_dev_fake_service_streams_without_a_gateway`
- [x] AC27 (by running) the page in a real browser against a running dev-fake server: streamed answer, markers, removed markers, clarification, cancel, error, XSS payloads, feedback; time to first progress and first token measured; section 「P3-04 流式和页面」 of `docs/reports/P3-report.md`

- [x] AC28 (review 1) A real server told to exit with a stream open shuts down within seconds: the run is cancelled before uvicorn waits for open responses, the client gets a `server_shutdown` error event, the record says `client_cancelled` with `cancelled_by: shutdown`; a stream asked for after that is 503 — `test_serve_stream_cancel.py::test_a_shutdown_with_a_stream_open_cancels_it_within_seconds`, `::test_a_stream_asked_for_after_the_shutdown_began_is_refused_with_503`, `test_serve_cli.py::test_the_server_waits_a_few_seconds_for_open_streams_then_cancels_them`
- [x] AC29 (review 1) A client that stops reading is cut off past `sse_queue_max_events` (`cancelled_by: slow_client`) — `test_serve_stream_cancel.py::test_a_client_that_does_not_read_is_cut_off_when_its_queue_is_full`
- [x] AC30 (review 1) Request and session ids are matched whole (no trailing newline) — `test_serve_web.py::test_feedback_rejects_bad_bodies`, `::test_a_trailing_newline_is_not_part_of_a_session_id`
- [x] AC31 (review 1) HTTP routes refuse another Host (421) and a POST that is not `application/json` (415) with the usual error body; localhost names, the listen address and `service_allowed_hosts` pass — `test_serve_guard.py` (all)
- [x] AC32 (review 1) `rag.serve.devfake` is imported only with `--dev-fake`, which refuses other addresses before anything starts and writes its records to `data/observe/dev-fake/` — `test_serve_cli.py::test_dev_fake_is_not_imported_unless_asked_for`, `::test_dev_fake_refuses_other_addresses_and_writes_its_records_elsewhere`, `test_serve_stream.py::test_dev_fake_service_streams_without_a_gateway`
- [x] AC33 (review 1) The page decodes the stream as a stream (bytes cut inside a character), keeps the question, the reply and the token when a reply fails, errors or is cancelled, and marks text streamed before a failure as unchecked — `test_serve_web.py::test_the_pages_own_code_in_node`

## Out of scope
Feedback into traces and metrics (P5; here one record line); server-side persistence of the page's history;
authentication of browser users (P8; the caller is the configured local user as for every HTTP call); a
reverse proxy configuration file (the headers are set; nginx notes in the README); WebSocket; streaming over
MCP (stays non-streaming: s16-9); React pages for evaluation data (later phases).

## Handoff (filled at the end)
- Files: new `src/rag/agent/events.py`, `src/rag/serve/{stream,devfake}.py`, `src/rag/serve/web/{index.html,sse.js,render.js,app.js,app.css}`, `src/rag/testing/live.py`, `scripts/stream_latency.py`, `tests/unit/test_{agent_stream,serve_stream,serve_stream_cancel,serve_web}.py`, `tests/web/ui_test.js`; changed `src/rag/agent/{nodes,runtime,session}.py` (one `astream` loop behind `ask`; `emit`; `SessionLocks.busy`), `src/rag/serve/{service,http_app,schema,cli}.py` (`_ask_docs` shared by `ask_docs` and `ask_docs_stream`, the stream route, `/v1/feedback`, `GET /` and `/ui/*`, `--dev-fake`), `src/rag/types.py` (`ErrorCategory.CLIENT_CANCELLED`), `src/rag/config/settings.py` (`sse_heartbeat_s`), `src/rag/testing/agent.py` (`SlowStreamingChatModel`), `README.md`, `.env.example`, `docs/decisions.md`, `docs/reports/P3-report.md` (section 2). No existing test changed.
- Decisions (docs/decisions.md, 2026-10-20 P3-04): tokens come from the answer node through the stream writer, not from LangGraph's `messages` mode (the token-cap guard replaces the run's callbacks, so a guarded model emits no message chunks: verified on langgraph 1.2.15); one run implementation for `ask` and the stream; the run lives in its own task behind a queue and the closing of the response cancels it; cancelled requests are one `client_cancelled` business record line (not an error); errors before the first event are plain JSON with the status of `POST /v1/ask_docs`, later ones are `error` events (no `done`), no heartbeat before the first event; a deadline in the middle ends with an `answer` event (`stop_reason: deadline`); `answer` event = `AskResponse`; page without a framework, CSP without inline code, text nodes only; the knowledge-base list reuses `GET /v1/knowledge_bases`; feedback writes one record keyed by the rated request; `--dev-fake` on local addresses only.
- Open issues: first text on the real gateway came after 11 to 16 s (rewrite, search, judge and the model's first token; 2 runs, no P95); nothing was run behind nginx; the browser check used the fake service and Edge only (no Firefox, no narrow screen, `queued` and mid-stream `error` only in tests); the search thread of a cancelled run cannot be stopped (it ends at its own deadline but does not advance the graph); the event queue has no bound; `queued` is only seen for an in-process lock (a file lock held by another process is not detected); feedback checks the request id's shape, not that it exists, and has no login; the dev server of this run wrote its records to the ignored `data/observe/`.
- Review rounds: 1 (CHANGES: shutdown did not cancel open streams, TextDecoder stream mode untested, resume token thrown away on a failed reply, trailing newline accepted in ids, no Host or Content-Type check, and nits; all fixed, see docs/decisions.md 「审查第 1 轮」). Round 0: TDD note: each stage's tests were written first and run red (agent events: `emit` unknown; stream: 404 on the route; web: missing files). The feedback endpoint and the KB-list test were written together with or after the code (the service code was committed in the stream commit, its tests came with the page commit), so those two tests never saw red. 27 mutation checks: 23 caught at once, 2 equivalent, 2 caught after adding a test.
