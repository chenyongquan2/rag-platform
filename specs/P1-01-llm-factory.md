# P1-01 Chat model factory

Status: review
Phase: P1
Depends on: P0-05 (cost ledger, budget reservation, timeout profiles)
Design sections: s11-1 (model entry, structured output), s16-2 (llm/ package, explicit config, per-use timeouts), s16-8 (error categories)
Calls real models: yes (six short `llm` probe calls; spent 0.002752 CNY at the estimated price)

## Goal
One factory that builds every LangChain chat model from config: `init_chat_model` against the OpenAI-compatible
gateway with three named profiles (`online`, `offline`, `cheap`), every HTTP attempt reserving its estimated cost
and writing its real usage (reasoning tokens included) to the shared cost ledger; a structured-output helper for
router and judge nodes; error categories for model failures; a manual switch to `FakeChatModel`.

## Interface
- `rag.config.ChatProfile(model="", connect_timeout_s, read_timeout_s, deadline_s, max_retries, backoff_s,
  backoff_max_s, max_tokens, temperature)`;
  `ServiceSettings` adds `chat_backend: "gateway" | "fake"`, `chat_fake_reply`, `chat_structured_output_method:
  "function_calling" | "json_schema" | "json_mode"`, `chat_online`, `chat_offline`, `chat_cheap: ChatProfile`
  (env: JSON, e.g. `CHAT_ONLINE='{"read_timeout_s": 8}'`; keys left out keep that profile's defaults);
  `ModelPrice.estimated: bool` (the chat price is marked as an estimate).
- `rag.llm`: `ChatProfileName`; `resolve_chat_profile(profile, settings) -> ChatProfile`;
  `make_chat_model(profile, settings=None, *, ledger=None, transport=None, async_transport=None, clock, sleep) -> BaseChatModel`;
  `LedgerTransport` / `AsyncLedgerTransport` (httpx transports that retry within the deadline, reserve and record);
  `structured_output(model, schema, *, method=None, settings=None) -> Runnable[..., schema]`;
  `StructuredOutputError`; `chat_error_category(exc) -> ErrorCategory | None`.
- `rag.types.ErrorCategory` adds `llm_timeout`, `llm_error`, `budget_exceeded`, `structured_output_failed`.
- `rag.testing.FakeChatModel.default_response` (reply used after the script, instead of raising).

## Acceptance criteria
Unit tests in `tests/unit/test_llm.py` replace the gateway with `httpx.MockTransport` (no network).
- [x] AC1 Profile defaults: online connect 3 s, read 10 s, deadline 12 s, 1 retry; cheap 3 s, 8 s, 1 retry, 1024 tokens; offline long read timeout, no deadline, more retries; all on `chat_model`; a partial env override keeps the profile's other values; structured-output method defaults per the probe — `test_chat_profile_settings_defaults_and_overrides`
- [x] AC2 The factory builds a `ChatOpenAI` through `init_chat_model` with SDK retries off; requests carry the profile's model, `max_completion_tokens`, temperature, connect and read timeouts, and `stream_options.include_usage` on streams — `test_factory_builds_profile_models`
- [x] AC3 Missing `LLM_BASE_URL`/`LLM_API_KEY` or a model without a price fails before any request — `test_factory_requires_gateway_and_price`
- [x] AC4 Every 2xx answer appends one `chat` ledger line with the profile, prompt tokens, output tokens including reasoning tokens, `reasoning_tokens`, the served model and `price_estimated`; reasoning tokens that the gateway leaves out of `completion_tokens` are added — `test_invoke_records_usage_with_reasoning_tokens`
- [x] AC5 Streams pass every chunk through and record the usage of the last chunk once; a stream closed early records an estimated line — `test_stream_records_usage_from_last_chunk`, `test_stream_closed_early_records_estimate`
- [x] AC6 Every attempt reserves (prompt characters + `max_tokens` × `n`) before it is sent; over the stop raises `BudgetExceeded` (category `budget_exceeded`) and sends nothing; reservations are released afterwards — `test_budget_reserved_before_send_and_refused_over_stop`
- [x] AC7 Async calls record the same way — `test_async_invoke_records_usage`
- [x] AC8 A read timeout after sending (also mid-stream) records an estimated line and is not retried (`llm_timeout`); connect failures, 408/429/5xx are retried within the profile's deadline, each attempt's timeouts cut to the time left; 5xx and connection errors are `llm_error` without ledger lines; a `Retry-After` longer than `backoff_max_s` is not waited for — `test_read_timeout_is_not_retried_and_connect_failures_are`, `test_deadline_limits_retries_and_attempt_timeouts`, `test_gateway_errors_classified_and_retry_after_capped`
- [x] AC8b A call cancelled or interrupted after it was handed to the network (async `CancelledError` while waiting or streaming, sync `KeyboardInterrupt`) records an estimated line and releases its reservation — `test_cancelled_call_records_estimate`
- [x] AC8c The budget check keeps a running total per ledger file and reads only lines appended since — `tests/unit/test_cost.py::test_total_spent_reads_only_new_lines`
- [x] AC9 The API key echoed in an error body is masked — `test_error_body_masks_api_key`
- [x] AC10 `structured_output` parses a scripted tool call into the Pydantic object with `FakeChatModel`; no tool call or invalid arguments raise `StructuredOutputError` (`structured_output_failed`) — `test_structured_output_parses_with_fake_model`, `test_structured_output_failures_are_classified`
- [x] AC11 Through the factory, `function_calling` sends one forced tool and `json_schema` sends `response_format`; both parse the answer — `test_structured_output_methods_request_shape`
- [x] AC12 `chat_backend=fake` returns `FakeChatModel` with the fixed reply: no HTTP, no ledger line — `test_fake_backend_fallback`
- [x] AC13 (`llm`) One short reply on the online profile: real usage in the ledger, reasoning tokens counted — `tests/llm/test_chat_gateway.py::test_real_gateway_reply_counts_reasoning_tokens`
- [x] AC16 (`llm`) The gateway honours `max_completion_tokens`; a 16-token cap is spent on reasoning (empty answer, `finish_reason=length`) — `tests/llm/test_chat_gateway.py::test_real_gateway_max_tokens_caps_reasoning`
- [x] AC14 (`llm`) Both structured-output methods probed; the configured default works — `tests/llm/test_chat_gateway.py::test_real_gateway_structured_output_methods`
- [x] AC15 (`llm`) `.stream()` yields several content chunks and the ledger gets the stream's real usage — `tests/llm/test_chat_gateway.py::test_real_gateway_streams_chunks`

Also (checked by running, not by a test): the `llm` file ran once (4 passed), then the reply and stream probes once more after the review changes (2 passed); ledger total 0.000040 CNY before, 0.002792 CNY after; `function_calling` works, `json_schema` gets HTTP 400 `This response_format type is unavailable now`; results in `docs/decisions.md`.

## Out of scope
Answer prompts, numbered-citation parsing, `found`/`answer_parse_failed` (P1-02); Pydantic schemas of router/judge
nodes (P1-04, P3); a total per-request deadline passed down the graph (P3); circuit breaker (P8); switching off
reasoning for the cheap profile (needs a gateway probe; open issue).

## Handoff (filled at the end)
- Files: `src/rag/llm/{__init__,factory,transport,structured,errors}.py`, `src/rag/config/{settings,__init__}.py` (`ChatProfile`, `chat_*` settings, `ModelPrice.estimated`), `src/rag/types.py` (4 error categories), `src/rag/testing/fakes.py` (`FakeChatModel.default_response`), `src/rag/cost/ledger.py` (running total, budget message), `tests/unit/{test_llm,test_cost}.py`, `tests/llm/test_chat_gateway.py`, `README.md`, `.env.example`, `docs/decisions.md`.
- Decisions (docs/decisions.md): retries and ledger in an httpx transport below the openai SDK, SDK retries off (sees every attempt, one deadline per call; LangChain callbacks and a ChatOpenAI subclass compared); `function_calling` default (gateway rejects `json_schema`); profiles online connect 3 s/read 10 s/deadline 12 s/1 retry/4096 tokens, cheap 3/8/10 s/1/1024, offline 10/180 s/no deadline/4/8192; read timeouts (also mid-stream) are never retried; cancelled calls get an estimated line; budget check reads only new ledger lines; reasoning tokens are inside `completion_tokens` and recorded separately; chat price stays an estimate (2 and 8 CNY per million, `price_estimated: true`); `CHAT_BACKEND=fake` is manual; categories `llm_timeout`, `llm_error`, `budget_exceeded`, `structured_output_failed`.
- Open issues: a non-streamed answer longer than about 10 s of generation fails with `llm_timeout` until P1-02/P1-04 measure real answer latency (streaming is not affected); the real `deepseek-v4-flash` price is unknown (the gateway shows no bill per call); switching off reasoning for the cheap profile and `json_mode` were not probed; a cancellation cannot tell whether the request had left, so it is always billed as an estimate; reservations are per process (as in P0-05); `ChatOpenAI` drops `reasoning_content` (not needed).
- Review rounds: round 1 CHANGES (cancellation not billed, online worst case about 61 s, full ledger read per attempt, proxy warning, `n` missing from the estimate); all fixed. 17 unit tests in `test_llm.py` plus 1 in `test_cost.py`; 109 unit tests pass in total; `llm` probes ran once, reply and stream again after the fixes.
