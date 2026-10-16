# P1-01 Chat model factory

Status: in progress
Phase: P1
Depends on: P0-05 (cost ledger, budget reservation, timeout profiles)
Design sections: s11-1 (model entry, structured output), s16-2 (llm/ package, explicit config, per-use timeouts), s16-8 (error categories)
Calls real models: yes (four short `llm` probes, estimated < 0.05 CNY in total)

## Goal
One factory that builds every LangChain chat model from config: `init_chat_model` against the OpenAI-compatible
gateway with three named profiles (`online`, `offline`, `cheap`), every HTTP attempt reserving its estimated cost
and writing its real usage (reasoning tokens included) to the shared cost ledger; a structured-output helper for
router and judge nodes; error categories for model failures; a manual switch to `FakeChatModel`.

## Interface
- `rag.config.ChatProfile(model="", timeout_s, max_retries, max_tokens, temperature, retry_after_max_s)`;
  `ServiceSettings` adds `chat_backend: "gateway" | "fake"`, `chat_fake_reply`, `chat_structured_output_method:
  "function_calling" | "json_schema" | "json_mode"`, `chat_online`, `chat_offline`, `chat_cheap: ChatProfile`
  (env: JSON, e.g. `CHAT_ONLINE='{"timeout_s": 10}'`; keys left out keep that profile's defaults);
  `ModelPrice.estimated: bool` (the chat price is marked as an estimate).
- `rag.llm`: `ChatProfileName`; `resolve_chat_profile(profile, settings) -> ChatProfile`;
  `make_chat_model(profile, settings=None, *, ledger=None, transport=None, async_transport=None) -> BaseChatModel`;
  `LedgerTransport` / `AsyncLedgerTransport` (httpx transports that reserve, record and cap `Retry-After`);
  `structured_output(model, schema, *, method=None, settings=None) -> Runnable[..., schema]`;
  `StructuredOutputError`; `chat_error_category(exc) -> ErrorCategory | None`.
- `rag.types.ErrorCategory` adds `llm_timeout`, `llm_error`, `budget_exceeded`, `structured_output_failed`.
- `rag.testing.FakeChatModel.default_response` (reply used after the script, instead of raising).

## Acceptance criteria
Unit tests in `tests/unit/test_llm.py` replace the gateway with `httpx.MockTransport` (no network).
- [ ] AC1 Profile defaults: online and cheap short timeout and 1 retry, offline long timeout and more retries, all on `chat_model`; a partial env override keeps the profile's other values; structured-output method defaults per the probe — `test_chat_profile_settings_defaults_and_overrides`
- [ ] AC2 The factory builds a `ChatOpenAI` through `init_chat_model`; requests carry the profile's model, `max_completion_tokens`, temperature and timeout, and `stream_options.include_usage` on streams — `test_factory_builds_profile_models`
- [ ] AC3 Missing `LLM_BASE_URL`/`LLM_API_KEY` or a model without a price fails before any request — `test_factory_requires_gateway_and_price`
- [ ] AC4 Every 2xx answer appends one `chat` ledger line with the profile, prompt tokens, output tokens including reasoning tokens, `reasoning_tokens`, the served model and `price_estimated`; reasoning tokens that the gateway leaves out of `completion_tokens` are added — `test_invoke_records_usage_with_reasoning_tokens`
- [ ] AC5 Streams pass every chunk through and record the usage of the last chunk once; a stream closed early records an estimated line — `test_stream_records_usage_from_last_chunk`, `test_stream_closed_early_records_estimate`
- [ ] AC6 Every attempt reserves (prompt characters + `max_tokens`) before it is sent; over the stop raises `BudgetExceeded` (category `budget_exceeded`) and sends nothing; reservations are released afterwards — `test_budget_reserved_before_send_and_refused_over_stop`
- [ ] AC7 Async calls record the same way — `test_async_invoke_records_usage`
- [ ] AC8 A read timeout after sending records an estimated line per attempt, is retried `max_retries` times and ends as `llm_timeout`; 5xx and connection errors are `llm_error` without ledger lines; a `Retry-After` longer than the profile's `retry_after_max_s` is not waited for — `test_timeout_is_retried_recorded_and_classified`, `test_gateway_errors_classified_and_retry_after_capped`
- [ ] AC9 The API key echoed in an error body is masked — `test_error_body_masks_api_key`
- [ ] AC10 `structured_output` parses a scripted tool call into the Pydantic object with `FakeChatModel`; no tool call or invalid arguments raise `StructuredOutputError` (`structured_output_failed`) — `test_structured_output_parses_with_fake_model`, `test_structured_output_failures_are_classified`
- [ ] AC11 Through the factory, `function_calling` sends one forced tool and `json_schema` sends `response_format`; both parse the answer — `test_structured_output_methods_request_shape`
- [ ] AC12 `chat_backend=fake` returns `FakeChatModel` with the fixed reply: no HTTP, no ledger line — `test_fake_backend_fallback`
- [ ] AC13 (`llm`) One short reply on the online profile: real usage in the ledger, reasoning tokens counted — `tests/llm/test_chat_gateway.py::test_real_gateway_reply_counts_reasoning_tokens`
- [ ] AC14 (`llm`) Both structured-output methods probed; the configured default works — `tests/llm/test_chat_gateway.py::test_real_gateway_structured_output_methods`
- [ ] AC15 (`llm`) `.stream()` yields several content chunks and the ledger gets the stream's real usage — `tests/llm/test_chat_gateway.py::test_real_gateway_streams_chunks`

Also (checked by running, not by a test): ledger total before and after the `llm` run; results in `docs/decisions.md`.

## Out of scope
Answer prompts, numbered-citation parsing, `found`/`answer_parse_failed` (P1-02); Pydantic schemas of router/judge
nodes (P1-04, P3); a total per-request deadline passed down the graph (P3); circuit breaker (P8); switching off
reasoning for the cheap profile (needs a gateway probe; open issue).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
