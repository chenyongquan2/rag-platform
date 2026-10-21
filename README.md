# rag-platform

Multi-knowledge-base RAG platform: Elasticsearch hybrid retrieval (BM25 + vector) with application-level RRF and cross-encoder reranking, LlamaIndex ingestion, LangChain model calls, LangGraph agentic RAG, MCP tools, and an evaluation harness.

Status: design done, implementation starting (phase P0).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
uv sync                          # create .venv from uv.lock (CPU-only torch)
cp .env.example .env             # fill in gateway URLs and keys; .env is git-ignored
uv run pre-commit install        # ruff, ruff format, unit tests, gitleaks on every commit
uv run pytest                    # unit tests only (default)
uv run pytest -m integration     # needs scripts/dev_up.sh (Elasticsearch, Redis, Collector, Jaeger)
uv run pytest -m llm             # calls the real model gateway and costs money; only when a spec says so
uv run pre-commit run --all-files
```

Test markers: `unit` (default; added to unmarked tests; no network, no Docker), `integration`
(local Docker services), `llm` (real model calls). The default run excludes `integration` and `llm`.
Unit tests use `rag.testing.FakeEmbedding` and `FakeChatModel`; LlamaIndex `Settings.llm` is a stub
that raises when called. The gitleaks hook is built from source and needs a Go toolchain on `PATH`.

## Local services

`docker-compose.yml` runs the infrastructure; the Python app runs on the host.

```bash
scripts/dev_up.sh                                   # docker compose up -d, then wait until healthy
uv run python -m rag.index.cli init --kb hf_docs    # create kb_hf_docs_v1 and the kb_hf_docs alias
uv run python -m rag.index.cli status --kb hf_docs  # alias, versions, document counts, data generation
uv run python -m rag.index.cli rebuild --kb hf_docs   # new index version, gate, alias switch (new model or chunking)
uv run python -m rag.index.cli rollback --kb hf_docs  # alias back to the previous version, nothing deleted
docker compose down                                 # stop (Elasticsearch data stays in the es-data volume)
```

| Service | Image | Host address |
|---|---|---|
| Elasticsearch (single node, Basic, 2 GB heap) | `docker.elastic.co/elasticsearch/elasticsearch:9.5.5` | `http://127.0.0.1:29200` |
| Redis cache (allkeys-lru, 256 MB, no persistence) | `redis:8.10.2-alpine` | `redis://127.0.0.1:26379/0` |
| OTel Collector (OTLP in; traces to Jaeger and `data/otel/traces.jsonl`, at most 4 x 20 MB; metrics in Prometheus format) | `otel/opentelemetry-collector-contrib:0.162.0` | gRPC `127.0.0.1:24317`, HTTP `127.0.0.1:24318`, metrics `http://127.0.0.1:28889/metrics` |
| Jaeger (all-in-one, in-memory) | `jaegertracing/jaeger:2.22.0` | UI `http://127.0.0.1:36686` |

Host ports are the standard port + 20000 and bound to 127.0.0.1 only. **Elasticsearch security is
disabled** (no TLS, no password) for local development; never expose this setup beyond the machine.
Connection defaults in `ServiceSettings` match these ports; override with `ES_URL`, `REDIS_URL`,
`OTLP_ENDPOINT`. Integration tests skip with a message when a service is not reachable, and use
`test<run>_`-prefixed indices and Redis keys that they delete afterwards.

## Ingestion

```bash
uv run python -m rag.ingest.job --kb hf_docs --limit 50 --fake-embed  # no cost; writes fake_kb_hf_docs
uv run python -m rag.ingest.job --kb hf_docs --limit 50               # real embeddings (ledger, budget)
uv run python -m rag.ingest.job --kb hf_docs --all                    # every document
```

Each run reads and cleans every document, writes new and changed chunks, then deletes old ones;
the docstore (`data/docstore/<index>.json`) is saved only after the whole run succeeded, and a
report goes to `data/reports/<kb>/ingest-<run>.json`. A run stops without changing anything when
more than 5% of the documents fail to read, when it would pass fewer than half of the docstore's
documents (`--allow-mass-delete` to confirm), or when the estimate does not fit the budget.

### Private knowledge base: the MetaTrader 5 manual (CHM)

`kb/private/mt5.yaml` (spec P4-01) holds the MT5 API manual in Chinese and English. It is private:
only the caller named in `access.callers` may use it, and nothing extracted from the manual is
committed (`data/raw/`, `data/reports/`, the docstore and the cache are git-ignored).

```bash
# once: unpack the two CHM files (needs 7-Zip, `7z` on PATH or in C:/Program Files/7-Zip; or chmlib)
uv run python -m rag.data.chm unpack --chm MetaTrader5SDK_Chinese.chm --out data/raw/mt5/zh
uv run python -m rag.data.chm unpack --chm MetaTrader5SDK_English.chm --out data/raw/mt5/en
uv run python -m rag.ingest.chm_report --kb mt5 [--limit 50]   # read, clean, chunk; counts only; no model calls
uv run python -m rag.ingest.job --kb mt5 --limit 50 --fake-embed
uv run python -m rag.ingest.job --kb mt5 --all                 # real embeddings: costs money (ledger)
```

Pages are read from the parsed HTML (code, tables, lists kept; navigation, empty paragraphs and
anchors counted and dropped); each page has a `language` and a `page_key` shared by its two
versions, and `retrieval.dedupe_by_page_key: true` keeps one language version per page in a
search result. A cleaning rule can carry `max_page_share` / `max_corpus_share`: the run stops
before writing anything when the rule removes more than that.

Warning: a run deletes every page of the index that the run's source does not deliver. A narrower
`sections` in a volume, or `--limit`, run against the real `kb_mt5` index removes the left-out
pages; use them only with `--index-prefix` on a throw-away index. The full, reproducible ingest is
`uv run python -m rag.ingest.job --kb mt5 --all` with the committed `kb/private/mt5.yaml`
(21,913 chunks; the real embeddings cost about 2.07 CNY in total, and a finished page is not
paid for twice because the embedding cache and docstore are reused: set `DOCSTORE_URL` and
`EMBED_INGEST_CACHE_URL` as in `docs/reports/P4-report.md` section 1.9).

## Embeddings and cost

`rag.embed.make_embedding("ingest" | "query")` builds every embedding model: the same gateway
model, dimensions and text prefix on both sides, only timeout and retries differ. Vectors go
through the L0 cache (`data/cache/embeddings.sqlite3` for ingestion, the cache Redis for queries).
`EMBED_BACKEND=fake` switches to `FakeEmbedding` when the gateway is down or the budget is used up.

```bash
uv run python -m rag.cost summary   # money spent so far, per kind and model
```

Every real model call appends a line to the cost ledger, by default
`<parent of the main checkout>/rag-platform-data/cost/ledger.jsonl` (here
`C:/workspace/rag-platform-data/cost/ledger.jsonl`): one git-ignored file shared by the main
checkout and every worktree, so the budget stop sees all spend. Each request reserves its
estimated cost first and stops with `BudgetExceeded` once spent + reserved + estimate would pass
`BUDGET_STOP_CNY`. At most `EMBED_MAX_CONCURRENCY` requests per client are in flight.

## Chat models

`rag.llm.make_chat_model("online" | "offline" | "cheap")` builds every chat model with
`init_chat_model` against the gateway (`LLM_BASE_URL`, `LLM_API_KEY`). Profiles differ in connect
and read timeouts, a deadline per call (online 12 s, cheap 10 s), retries and `max_tokens` (which
covers reasoning and answer tokens of the reasoning model); override one with JSON, e.g.
`CHAT_ONLINE='{"read_timeout_s": 8}'`. The ledger transport does the retries (never after a read
timeout) and every attempt reserves its estimated cost and writes its usage to the cost ledger.
`rag.llm.structured_output(model, Schema)` returns Pydantic objects via `function_calling` (the
gateway rejects `json_schema`). `CHAT_BACKEND=fake` switches every profile to a `FakeChatModel`
with a fixed reply; it is never switched automatically.

## Answers with numbered citations

`rag.answer` turns a question plus ranked chunks into an `Answer` (no retrieval inside):

```python
ctx = build_context([result.chunks], cfg.answer.context_tokens, metadata_keys=cfg.metadata.llm)
async for event in stream_answer(question, ctx, kb=cfg.kb, config=cfg.answer):
    ...  # AnswerDelta (text as generated), AnswerRetry, then AnswerDone(answer)
answer = await generate_answer(question, ctx, kb=cfg.kb, config=cfg.answer)  # same, no deltas
```

The context builder dedupes chunk ids, splits the token budget per sub-question and wraps each
fragment `[n]` in `<doc_fragment n="n" untrusted="true">` delimiters (`suspicious_instruction`
labelled). The prompt (zh or en, by the question's language unless the KB's `answer.language`
fixes it) asks for `[n]` after every sentence or the fixed not-found sentence. The model (`online`
profile) is always streamed; the answer is parsed (`【1】`, `[1,2]`, `[1][2]` accepted), unknown
numbers are removed and uncited sentences counted (`answer.citation_check`). No citation at all:
one retry, then `answer_parse_failed` with the chunks only.

## Observability

`rag.observe` (spec P0-11, design s15-1): call `init_observability(settings, "<service>")` once at
startup. It exports traces and metrics over OTLP to the Collector (`OTLP_PROTOCOL=grpc`, default,
or `http/protobuf` with `OTLP_ENDPOINT=http://127.0.0.1:24318`), instruments LlamaIndex and
LangChain/LangGraph with OpenInference (`OTEL_AUTO_INSTRUMENT`, `OTEL_TRACE_CONTENT`; embedding
vectors are never recorded), writes JSON logs with trace and request ids to
`data/observe/logs/<service>.jsonl` and feeds token and CNY counters from the cost ledger.

```python
with request_span("search_docs", kb=kb, caller=caller, config_version=cfg_sha) as scope:
    with layer_span("retrieve", attributes={attributes.ROUTE: "bm25"}):
        ...
    scope.set_result(result)              # flags -> span attributes and error metrics
    write_business_record(business_record_from_result(result, tool="search_docs", ...))
```

Business records (one JSON line per request: per-route top 10, rerank scores, threshold decision,
flags, timings, trace id) go to `data/observe/records/business.jsonl`. Records and logs rotate by
size; `data/observe/` is git-ignored. Attribute and metric names live in
`rag/observe/attributes.py`. Tests use `rag.testing.memory_observability()` (`OTEL_MODE=memory`);
`OTEL_MODE=off` records nothing.

## Retrieval

`rag.retrieve.make_hybrid_search(cfg, settings, acl_groups=[...])` builds the retrieval chain of
one knowledge base: `EsHybridRetriever` sends a BM25 `multi_match` (content + boosted title_path)
and a kNN search (filter inside `knn`) in one `_msearch`, both limited to the knowledge base, the
caller's ACL groups and the whitelisted caller filters (`doc_type`, `language`,
`source_rel_prefix`); RRF fuses them (k=60, deterministic ties) and `TextDedupe` drops equal
bodies. One failed route (error, time-out, or no query vector) gives a `degraded` result from the
other; none gives `RetrievalError` (`retrieval_failed` / `retrieval_timeout`). The total time
limit is `deadline_ms` per call or `SEARCH_DEADLINE_MS`.

```python
# index_prefix="fake_" with EMBED_BACKEND=fake queries the fake-vector index
search = make_hybrid_search(cfg, settings, acl_groups=["public"], redis=redis)
result = search.retrieve("How do I log in?", kb=cfg.kb, top_k=8, filters={"language": "en"})
result.chunks, result.routes, result.degraded, result.timings_ms
```

`rag.retrieve.make_search_pipeline(...)` (same arguments) adds the rest of the chain:
`BatchedTimeoutRerank` scores the first `rerank_n` candidates with the local
`BAAI/bge-reranker-base` cross-encoder (question + title path + body, batches of
`RERANK_BATCH_SIZE`, each on `RERANKER_TORCH_THREADS` torch threads, at most
`RERANK_MAX_CONCURRENCY` reranks at once (defaults: all physical cores, and the physical
cores divided by that, i.e. one; spec P0-13); the model is loaded once
per process from the local HF cache, so set `HF_HUB_OFFLINE=1` and build pipelines at service
start), stopping by the deadline (`rerank_timeout`: scored first, the rest in fusion order);
`RerankThreshold` drops chunks below `threshold` (`not_found` with the best 3 in
`below_threshold`); the list is cut to `top_k`; `expand: neighbors | parent` in the KB config
adds context pieces within `expand_max_tokens`. `Chunk.score` stays the fusion score,
`Chunk.rerank_score` is in [0, 1].

```python
pipeline = make_search_pipeline(cfg, settings, acl_groups=["public"], redis=redis)
result = pipeline.retrieve("How do I log in?", kb=cfg.kb, top_k=8)
result.chunks, result.rerank, result.threshold, result.not_found, result.below_threshold
pipeline.get_context(result.chunks[0].chunk_id, "neighbors")  # page content + updated if gone
```

## Search service (MCP and HTTP)

One process serves three tools: `search_docs(kb, query, filters, top_k, deadline_ms)`,
`get_context(chunk_id, mode)` and `list_knowledge_bases()`. At start it loads the reranker in the
background (about a minute; logged as `search service ready in ... s`); a search that arrives
earlier waits for the load within its deadline, then answers `not_ready`. Every request runs as
the caller in `SERVICE_CALLER` with the document groups in `SERVICE_ACL_GROUPS` (until OAuth in
P8); a knowledge base is visible when its `access.callers` has `*` or that caller. Responses mark
every chunk text `untrusted`; errors are only `{error, request_id, message}`, the details are in
the log under the same request id. Each request writes one business record line and a trace.

```bash
uv run python -m rag.serve stdio                       # MCP over stdio
uv run python -m rag.serve http --host 127.0.0.1 --port 28080
#   POST /v1/search_docs  {"kb": "hf_docs", "query": "...", "top_k": 5}
#   POST /v1/get_context  {"chunk_id": "...", "mode": "neighbors"}
#   GET  /v1/knowledge_bases, GET /healthz, GET /readyz (503 until loaded)
#   MCP over streamable-http at http://127.0.0.1:28080/mcp
```

Register the stdio server with an MCP client, for example Claude Code (the server reads `.env`
from its working directory, so run it from the repo; `HF_HUB_OFFLINE=1` keeps the reranker load
offline; add `EMBED_BACKEND=fake` and `SEARCH_INDEX_PREFIX=fake_` to search the fake-vector
index without paying for query embeddings):

```bash
claude mcp add rag-docs --env HF_HUB_OFFLINE=1 -- \
  uv run --directory C:/workspace/rag-platform python -m rag.serve stdio
```

or in a client's JSON config (`.mcp.json`, Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "rag-docs": {
      "command": "uv",
      "args": ["run", "--directory", "C:/workspace/rag-platform", "python", "-m", "rag.serve", "stdio"],
      "env": {"HF_HUB_OFFLINE": "1"}
    }
  }
}
```

For the HTTP server, register `http://127.0.0.1:28080/mcp` as a streamable-http server
(`claude mcp add --transport http rag-docs http://127.0.0.1:28080/mcp`). It has no
authentication until P8, so keep `SERVICE_HOST` on 127.0.0.1.

## Answering with the ask_docs graph

`ask_docs` is our own LangGraph `StateGraph` (`rag.agent`): `route` (rules first, the small model
only for what rules cannot decide) -> `clarify` (ask the caller back) -> `rewrite` -> `retrieve`
-> `judge` -> `answer` (the `rag.answer` layer) -> `verify`. What each step does is a function
in `AgentHooks` (`rag.agent.hooks`); the sufficiency judge and the correction rounds are in
`rag.agent.judge` and `rag.agent.correction` (below). It is exposed as the MCP tool
`ask_docs(kb, question, session_id, deadline_ms, resume_token, answer)` and `POST /v1/ask_docs`
(JSON body `{kb, question, session_id?, deadline_ms?, resume_token?, answer?}`), non-streaming (the streaming form is below),
with the caller, ACL, error categories, spans and business record line of `search_docs`.

```python
runtime = AgentRuntime.from_settings(settings)  # one per process: it owns the session locks
result = await ask_docs(
    "How do I push a model?",
    cfg=cfg,
    retriever=pipeline,
    settings=settings,
    runtime=runtime,
    session_id="s1",
)
result.answer, result.sentences, result.sources, result.rounds, result.timings_ms
result.status, result.coverage_unconfirmed, result.stop_reason, result.cache_allowed
result.standalone_question, result.search_queries, result.fallbacks  # what the rewrite did
```

- One run is one checkpoint thread `"{session_id}:{run_id}"` (SQLite file `CHECKPOINT_URL`); the
  run id is made by the server, the caller's request id is only for correlation. The retrieval
  fields of the state are appended per round and cleared by `route`, so a request never sees
  the chunks of another one in the same session.
- The turns of a session (question and answer text only) are stored per caller, KB and session id
  (`SESSION_STORE_URL`, last `AGENT_HISTORY_TURNS` kept) and passed in as input. A call that
  carried history has `cache_allowed=false`: the exact cache of P5 must not read or write it.
- Requests with the same caller, KB and `session_id` run one after another (`SESSION_LOCK_URL`: `memory://`
  inside this process, `file://<dir>` also across processes); different sessions run in
  parallel. The wait counts against the deadline (`agent_deadline` error when it runs out).
- Limits of one request: total deadline `AGENT_DEADLINE_MS` (a caller's `deadline_ms` replaces
  it; no further round starts when less than `AGENT_ANSWER_RESERVE_MS` is left, and a run
  cut off mid-node returns the chunks found so far), `AGENT_RECURSION_LIMIT` graph steps,
  `AGENT_TOKEN_CAP` model tokens (gateway usage, else an estimate; calls before the answer
  keep `AGENT_ANSWER_RESERVE_TOKENS` free for it), and the KB's `agent.max_rounds`. A run that
  stopped at a limit says so: `coverage_unconfirmed`, `stop_reason`, `notice`, and the
  categories `agent_deadline` / `token_cap_reached`.

### Rewrite, route and clarify (P3-02)

Rules run first and make no model call (`rag.agent.rules`): whitespace and Unicode are
normalised; identifiers (snake_case, CamelCase, dotted names, versions, flags such as `--lr` and
`-U`, paths including Windows paths, acronyms such as RLHF, model names such as
`bert-base-uncased`), code spans and quoted messages are found (a long message is kept by its
first 40 characters, a fenced code block by the identifiers inside it); and the question is
classified as clear, vague or undecided and as one lookup or possibly two. The
small model (`AGENT_REWRITE_PROFILE`, default `cheap`, structured output by function calling)
is called only where it can change something, and only if the KB's `agent` section allows it:

- `rewrite` (`agent.rewrite`): a follow-up becomes a standalone question using the session's
  history (this goes to the answer step); the search query is written as a declarative
  statement (`agent.declarative`; this goes to retrieval); a two-hop question becomes at most
  `agent.max_sub_questions` queries, one search each. A statement-form question with no history
  is only normalised: no model call. The model's output is untrusted (it read the question
  and earlier answers made from documents, which are given to it in marked untrusted blocks).
  Every identifier of the question must survive in the queries as a whole token (`3.1` is not
  `3.12`, `--lr` is not `--lr-scheduler`; identifiers and code spans are always whole, only a
  long quoted message is matched by its first 40 characters; sizes and precisions such as
  `8-bit`, `int8`, `fp16` are not required literally), the language must stay the question's,
  and the rewrite must not be off the topic. That last check is deliberately weak: lexical
  overlap cannot tell a vocabulary bridge ("make my model smaller" to "reduce model size with
  quantization") from a hijack, so it rejects only an output with an instruction-like phrase
  ("reveal the system prompt", "ignore the previous instructions") or with almost none of its
  topic words in the question (minus its instruction-like sentences) and the history. The
  defences that do not depend on it: the prompt carries the question, the history and the list
  of strings to keep in marked untrusted blocks (forged tags, full-width or with invisible
  characters, are neutralised), and the answer step is given the user's original question.
  Otherwise the original question is searched. A model failure, timeout (never into the answer's time),
  unparsable answer or refused call (token cap) does the same. The request is never failed:
  it is answered, `degraded` is set and `fallbacks` names the step and reason
  (`rewrite:timeout`, `rewrite:identifiers_lost`, `rewrite:ungrounded`, `route:parse_failed`,
  ...). The answer step is given the question as the user asked it and, when the rewrite
  changed it, the standalone version too, labelled; `standalone_question` is set only then.
- `route` (`agent.route_model`): when the rules call a question undecided (it points at "this
  model" with some topic words) or possibly two-hop, the small model decides; if it fails, the
  rules' view stands.
- `clarify` (`agent.clarify`): a question that names nothing to search ("How do I fix this
  error?") stops the run with LangGraph `interrupt`. `ask_docs` returns `status:
  need_clarification`, `answer` is the question to ask the user, `resume_token` is an opaque
  token. The session lock is released at that point and the history is not written.
  Call again with the same `kb` and `session_id`, `resume_token` and `answer` (the user's
  reply; `question` is required as always but not used) to continue the same checkpoint thread:
  the reply is added to the question and the run goes on (at most one clarification per
  request; a vague reply is searched as it is). A token works once, only for the caller, KB and
  session it was issued for, only while the run waits, for `AGENT_RESUME_TTL_S`; anything else
  is `invalid_resume_token` (HTTP 400), the same answer for every reason. A call that fails or
  is cancelled while running the thread gives the token back: the same token and call can be
  retried and the run goes on from the step that failed (with the first reply). A call that
  dies without ending leaves the token in use until a lease (twice `SEARCH_MAX_DEADLINE_MS`)
  lapses; only the call that holds the current claim can end it, and the tokens a failed
  attempt spent count on the retry. When the run ends, the session history gets the question, the question asked back,
  the reply and the answer as two turns (also when it was cut off and has no answer text: the
  reply is then stored with the notice; if the history store fails, the answer is still
  returned, with `degraded` and `history:write_failed`). The request's token cap counts both
  calls.

### Sufficiency judge and correction rounds (P3-03)

With `agent.judge` on, the `judge` step sorts what each round found into three tiers (design
13.4) per sub-question: **enough** (a chunk at or above the rerank threshold, and the small
model says the chunks hold what the sub-question asks for), **partly enough** (some chunk passed
but a sub-question is still without grounds; `missing` names which) and **not enough** (nothing
passed the threshold in any round). Rules come first: a sub-question whose chunks are all below
the threshold is missing without asking, and when that is true of all of them the model is not
called; with `agent.judge_skip_score` a sub-question that has a chunk scored at or above it is
grounded without asking (P3-06). The model (`AGENT_JUDGE_PROFILE`, default `cheap`; `ask_docs(judge_model=...)`) sees the
chunks that passed, best first, capped to `AGENT_JUDGE_CONTEXT_TOKENS`, in a marked untrusted
block, and answers grounded or not per numbered sub-question. A failure, timeout, refused call or
an answer that does not name every sub-question makes the rules decide alone: `judge:<reason>`
in `fallbacks`, `degraded` set, the request still answered.

Then the loop (at most `agent.max_rounds`, three): round 2 asks the missing sub-questions only,
in `agent.multi_queries` other phrasings each (multi-query); round 3 searches a hypothetical
answer passage (HyDE, `agent.hyde`). The phrasings pass the rewrite's rules (identifiers whole,
same language, no instruction-like text, on topic, nothing searched twice) and a HyDE passage is
a search query only: it is never shown to the judge or the answer. Results of all rounds
accumulate and are deduplicated by chunk id. No round starts that cannot finish
(the answer's time plus `AGENT_ROUND_MIN_MS`, or 0.9 times the last round's time when that is more,
must be left, else `stop_reason: deadline`). A search of round two or later is made only with
`AGENT_SEARCH_MIN_MS` before the answer's time (the ones that do not fit are dropped, one phrasing
of every sub-question first; none fits: `stop_reason: deadline` and straight to the answer); after
a rerank timeout in round two or later only the chunks the reranker scored are kept (in round
one the unscored fused chunks stay for the answer, as in `search_docs`; the judge never counts
them as grounds; `rerank_timeout` stays set either way: it says the search was cut short); a
search of a later round that fails does not fail the request: `stop_reason: retrieval_error`,
`retrieval_error` (`retrieval_timeout` / `retrieval_failed`, also in `error_categories`),
`degraded`, and it is answered from the earlier rounds. Round one still fails the request. A
question the route flagged as possibly needing two pages, and a sub-question whose only sure
chunk is also another's, are never settled by `agent.judge_skip_score` alone. When no
new phrasing can be made (model failure, every phrasing refused, or `agent.rewrite` off) the run
stops with `stop_reason: no_new_query`. Three rounds with nothing above the threshold are a plain
`not_found` (no stop reason) with `nearest`, the closest chunks of all rounds; part of the
question still missing after the last round is answered from what was found with
`stop_reason: max_rounds`. Switching knowledge base after a failed round is not built (one call
serves one KB); `correction.py` documents the extension point.

The result has `sufficiency`, `missing` and `nearest`. How often the judge agrees with a label
made from the question sets (the gold page is already among the chunks round one returned; a
no-answer question is never enough):

```bash
uv run python -m rag.eval.judge_accuracy run --kb hf_docs     --set dev=evals/datasets/hf_docs/synthetic_dev.jsonl     --set na=evals/datasets/hf_docs/no_answer_extra.jsonl     --out data/reports/p3-03/run [--limit N] [--max-cny 2.0] [--rescue]
uv run python -m rag.eval.judge_accuracy report data/reports/p3-03/run/records.json
```

It prints accuracy, precision and recall of "enough" with Wilson intervals, the confusion matrix,
the split by set and question type, and the comparison with the rules alone (an exact paired McNemar
test) and with always running two rounds (wasted rounds, missed corrections and the range of rounds;
no accuracy, which would only be the share of "not enough" labels). `--rescue` also runs rounds two and three for questions that were not enough.

| Setting (env var) | Default | Meaning |
|---|---|---|
| `AGENT_DEADLINE_MS` | 30000 | Total time of one request; at most `SEARCH_MAX_DEADLINE_MS` |
| `AGENT_ANSWER_RESERVE_MS` | 7000 | Time kept for writing the answer (below the deadline); answer step P99 6.0 s over 208 graph answers, 6.8-7.0 s on a busy machine (P3-05, P3-06) |
| `AGENT_ANSWER_MAX_TOKENS` | 1536 | Output tokens (reasoning included) of the graph's answer call; P99 of 208 answers was 731 (P3-06) |
| `AGENT_RECURSION_LIMIT` | 25 | LangGraph steps of one run (6 for one round, 3 more per further round) |
| `AGENT_TOKEN_CAP` | 20000 | Model tokens of one request |
| `AGENT_ANSWER_RESERVE_TOKENS` | 6000 | Tokens calls before the answer must leave (below the cap) |
| `AGENT_HISTORY_TURNS` | 6 | Past turns of a session given to the graph; 0 = none |
| `AGENT_REWRITE_PROFILE` | `cheap` | Chat profile of the small model (route and rewrite) |
| `AGENT_REWRITE_TIMEOUT_MS` | 6000 | Longest one such call may take (never into the answer's time) |
| `AGENT_REWRITE_HISTORY_TURNS` | 3 | Earlier turns shown to the small model; 0 = none |
| `AGENT_REWRITE_HISTORY_CHARS` | 400 | Characters of each question or answer shown to it |
| `AGENT_RESUME_TTL_S` | 1800 | How long a question asked back waits for its answer |
| `AGENT_JUDGE_PROFILE` | `cheap` | Chat profile of the sufficiency judge |
| `AGENT_JUDGE_TIMEOUT_MS` | 9000 | Longest one judge call (never into the answer's time) |
| `AGENT_JUDGE_CONTEXT_TOKENS` | 1500 | Estimated tokens of fragments shown to the judge |
| `AGENT_ROUND_MIN_MS` | 8000 | Time one more round needs, on top of the answer's time (floor; see `AGENT_ROUND_RATIO`) |
| `AGENT_ROUND_RATIO` | 0.9 | A further round also needs this share of the last round's time (P3-06) |
| `AGENT_SEARCH_MIN_MS` | 6000 | Least time a search of round two or later gets (after the answer's time); with less it is not made; a whole search took 4.5 s on a busy machine (P3-06) |
| `AGENT_RERANK_RESERVE_MS` | 3000 | The query-vector wait of a round-two search ends this long (at most half its limit) before its deadline, so the rerank has time; `search_docs` is not affected (P3-06) |
| `SSE_HEARTBEAT_S` | 10 | Seconds of silence before the stream sends a heartbeat comment line |
| `SSE_QUEUE_MAX_EVENTS` | 5000 | Events of one stream waiting for the client; past it the run is cancelled |
| `SERVICE_SHUTDOWN_TIMEOUT_S` | 5 | Seconds the server waits for connections to close at shutdown |
| `SERVICE_ALLOWED_HOSTS` | empty | Host names (besides localhost and the listen address) the HTTP routes answer to |
| `CHECKPOINT_URL` | `sqlite://data/checkpoints/langgraph.db` | Checkpoints (`memory://` for tests) |
| `SESSION_STORE_URL` | `sqlite://data/sessions/sessions.db` | Session history and resume tokens (`memory://` for tests) |
| `SESSION_LOCK_URL` | `memory://` | Session locks (`file://data/locks` for several processes) |
| KB config `agent.max_rounds` | 3 | Retrieval rounds of one request at most |
| KB config `agent.rewrite` | false (`hf_docs`: true) | The rewrite step may call the small model |
| KB config `agent.declarative` | true (`hf_docs`: false since P3-06) | Write the search query as a statement |
| KB config `agent.max_sub_questions` | 3 | Most searches of a split question (1: never split) |
| KB config `agent.clarify` | false (`hf_docs`: true) | Ask the caller back when the question is too vague |
| KB config `agent.route_model` | false (`hf_docs`: true) | The route may ask the small model about what rules leave open |
| KB config `agent.judge` | false (`hf_docs`: true) | Judge after each round, correct with more rounds |
| KB config `agent.multi_queries` | 3 | Phrasings of each missing sub-question in round 2 |
| KB config `agent.hyde` | true | Round 3 searches a hypothetical answer passage |
| KB config `agent.round2_on_not_enough` | true | false: a plain `not_enough` is "not found" at once; only `partly_enough` gets another round (P3-06; not on by default, see decisions) |
| KB config `agent.judge_skip_score` | null (`hf_docs`: 0.98) | A sub-question with a chunk scored at or above this needs no judge call (P3-06) |

## Streaming answers and the Q&A page (P3-04)

`POST /v1/ask_docs/stream` takes the body of `POST /v1/ask_docs` and answers with Server-Sent
Events (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`,
`X-Request-Id`). It is the same run as `ask_docs` (same caller and ACL, argument checks, session
lock, deadline, token cap, business record line, spans): `AgentRuntime.ask(..., emit=...)` and
`SearchService.ask_docs_stream` share their code with the non-streaming call. MCP stays
non-streaming. A failure before the first event (bad JSON or argument, unknown or hidden KB, a
resume token that is refused, a service that is not ready) is a plain JSON error with the status
code of `POST /v1/ask_docs`; once the first event is out the status is 200 and a failure is an
`error` event. A request that has to wait behind an earlier one of its session starts with the
`queued` progress event, so the status is 200 for it, and a refused resume token then arrives as an
`error` event.

An event is `event: <name>` and one line `data: <JSON>`, then a blank line. A line that starts with
`:` is a heartbeat (`: keep-alive`), sent every `SSE_HEARTBEAT_S` seconds (10) while the run is
silent, so a proxy that closes idle connections keeps the stream open. Behind nginx the
`X-Accel-Buffering: no` header turns its buffering off for this response.

| Event | Data | When |
|---|---|---|
| `progress` | `{node, phase, round?, ms?, ...}` | `node` is `queued`, `route`, `clarify`, `rewrite`, `retrieve`, `judge`, `answer` or `verify`. `phase` is `waiting` (queued behind the session's earlier request), `start`, `end` (with `ms`; `route`: `retrieve`, `clarify`; `rewrite`: `queries`; `retrieve`: `chunks` found by this round and `chunks_total` of all rounds; `judge`: `loop`, `verdict` when the KB judges; `answer`: `found`, `status`) or `retry` (answer: the attempt cited nothing, its text is void, `attempt`). No query or model text. |
| `token` | `{text, attempt}` | A piece of the answer as the model writes it, markers like `[1]` included. Only the `answer` node writes these. A new `attempt` number means the text so far is void. |
| `answer` | the `AskResponse` of `POST /v1/ask_docs` | The checked answer: `sentences` with their citations, `sources`, `citation_check.removed_markers` (the `[n]` in the streamed text that match no source), `found`, `sufficiency`, `degraded`, `fallbacks`, `stop_reason`, `notice`. It replaces the streamed text. |
| `need_clarification` | `{request_id, question, resume_token, session_id}` | Instead of `answer`: send the user's reply with `resume_token` and `answer` to the same endpoint (same `kb` and `session_id`). |
| `error` | `{error, request_id, message}` | Category (design 16.8), request id, a fixed sentence; never details. The last event: no `done`. |
| `done` | `{request_id, status, rounds, degraded, rerank_timeout, coverage_unconfirmed, stop_reason, tokens_used, timings_ms}` | The last event of a run that did not fail. `timings_ms` has the steps, `request`, and `first_progress` and `first_token` (ms since the request came in; no `first_token` when no text was written). |

A deadline that passes in the middle of the stream is not an error: the run is cut where it is and
the `answer` event carries what was found (`stop_reason: deadline`, `coverage_unconfirmed`, no
answer text; the page replaces the text it had streamed). A client that closes the connection
(closes the page, presses Cancel) cancels the run: no further model call, the session lock is
released, the checkpoint stays at the last finished node, the span ends `cancelled` and the
business record gets one line flagged `client_cancelled` (`extra.cancelled_at`: the node it was in).
That is not an error category of the service: nobody receives it. A client that stops
reading is cut off the same way once `SSE_QUEUE_MAX_EVENTS` events (5000) wait for it
(`cancelled_by: slow_client`); one that connects and never reads is not timed out.

When the server shuts down (`python -m rag.serve http` gets Ctrl-C or SIGTERM) it first
cancels the runs of the open streams: each stream ends with an `error` event
(`server_shutdown`, no `done`), the request is recorded as `client_cancelled` with
`cancelled_by: shutdown`, and the server waits at most `SERVICE_SHUTDOWN_TIMEOUT_S` (5 s)
for the connections to close. (uvicorn runs the app's own shutdown code only after open
responses have ended, so this is done by the server itself: `rag.serve.server`.)

Every HTTP route except `/mcp` answers only to the `Host` names `localhost`, `127.0.0.1`,
`[::1]`, the address the server listens on and `SERVICE_ALLOWED_HOSTS` (a JSON list such as `["docs.example.com"]`; set
your public name there when you deploy, and the name a container health check uses): another
`Host` is refused with 421, and a POST whose `Content-Type` is not `application/json` with
415, both with the usual error body. This keeps a web page on another site, or a DNS name
rebound to 127.0.0.1, from driving a service on a developer's machine. Listening on
`0.0.0.0` does not make any name valid: list the names.

`GET /` serves the page: `index.html` and `/ui/{sse.js,render.js,app.js,app.css}` from
`src/rag/serve/web/` (no build step, no framework, no outside request). The page sets
`Content-Security-Policy: default-src 'self'; script-src 'self'; ...; frame-ancestors 'none'` (no
inline script or style is needed), `X-Content-Type-Options: nosniff`, `Referrer-Policy:
no-referrer`. It lists the knowledge bases of `GET /v1/knowledge_bases` (only those the caller may
use), keeps a session id in `sessionStorage`, reads the stream with `fetch` (EventSource cannot
POST), shows progress, the streamed answer, and after the `answer` event each `[n]` as a button
that opens the passage (title, source, text; removed markers are struck through with a note),
asks the question back when the run needs it (the question, the reply and the token stay
until the reply has ended with an answer, so a failed or cancelled reply can be sent
again), cancels with an `AbortController`, greys out text that was streamed before a run
failed (it was not checked), shows errors
(category and request id) and sends the two feedback buttons to `POST /v1/feedback`
(`{request_id, useful, comment?}`, a comment up to 500 characters): one business record line
(`tool: feedback`, `request_id` the rated request's, `extra.useful`, `extra.comment`). Everything
that comes from a model or a document is put into the page as text, never as HTML.

```bash
uv run python -m rag.serve http --dev-fake --port 28099   # fake search and model, no gateway; open http://127.0.0.1:28099/
uv run python scripts/stream_latency.py --url http://127.0.0.1:28099 --kb demo_docs --question "How do I push a model with push_to_hub?" --runs 20
```

`--dev-fake` serves the knowledge base `demo_docs` on a local address only (its business
records go to `data/observe/dev-fake/`, not to the real file); questions with `xss`,
`retry`, `slow` or `nothing` in them show the markup case, a second answer attempt, a long answer
(to press Cancel) and a not-found result. The JavaScript is tested with node
(`tests/web/ui_test.js`, run by `tests/unit/test_serve_web.py`; skipped when node is missing).

## Retrieval evaluation

Any retriever with `retrieve(question, kb, top_k) -> RetrievalResult` can be evaluated
(`rag.eval.retrieval.Retriever`); `RetrievalResult.routes` carries each route's own list for
per-route metrics. Question sets live in `evals/datasets/<kb>/` as JSONL with page labels
(doc_ids) and optional chunk labels; the 65 HF questions are
`evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl` (source, license and changes in
the `NOTICE.md` next to it).

```python
cfg = load_kb_config(cfg_path, kb_root=settings.kb_dir)
record = run_retrieval_eval(retriever, load_question_set(path, kb=cfg.kb), kb_config=cfg)
# evals/reports/public/<run_id>/ for kb/public/ configs, else the git-ignored private/
write_run(record, public=is_public_kb_file(cfg_path, settings.kb_dir))
```

```bash
uv run python -m rag.eval.retrieval compare <run_a> <run_b>   # McNemar, permutation, changed questions
uv run python -m rag.eval.retrieval export-hf                 # rewrite the HF question set
```

A run records Hit@k, MRR, nDCG@k (ideal DCG capped at k) and Recall@k at page and chunk level for
the final list and each route, latency, git commit, KB config, index version, data generation,
question-set hash, model ids and the `uv.lock` hash. A k larger than the final list is capped
at its length (with a warning). `rerank_latency_curve` measures a reranker
for rerank_n 10, 20, 30.

## Answer evaluation

The standard (non-agentic) path end to end: search pipeline, `build_context`,
`generate_answer`; then an LLM judge against the reference answers (spec P1-04). `run`,
`security` and `judge` call real models and cost money (every call is in the cost ledger).
Records go to the git-ignored `data/reports/answer-eval/`.

```bash
uv run python -m rag.eval.answer run --kb hf_docs --limit 5 --label pilot   # a pilot first
uv run python -m rag.eval.answer project --answers <pilot run> --judge <pilot judge run>
uv run python -m rag.eval.answer run --kb hf_docs --label full
uv run python -m rag.eval.answer judge --answers <run> --scales 1-5,1-3 --repeat 2
uv run python -m rag.eval.answer stability <judge run 1> <judge run 2>
uv run python -m rag.eval.answer security                 # P1-03 set, rule-scored
uv run python -m rag.eval.answer human-sheet --answers <run> --n 50   # blind CSV for people
uv run python -m rag.eval.answer human-agreement --sheet S --key K --judge <judge run>
```

A run records per question the answer, cited chunk ids, citation check counts, parse outcome,
latencies and the ledger's tokens and CNY, plus format compliance with Wilson intervals. The
judge (`qwen/qwen3.6-flash`, thinking off, temperature 0, prompt `p1-04-hf-v1`) uses the HF
rag_evaluation (1-5) and agent_rag (1-3) rubrics; answers that are not given score 1 by rule.

## Agentic RAG against standard RAG

Spec P3-05 (design s13-5): the same 150 questions (65 public single-hop, 20 two-hop, 15 colloquial
rewrites in `colloquial.jsonl`, 50 no-answer) go through standard RAG and the real `ask_docs`
graph, are scored with the P1-04 judge (1-3 scale), and compared pairwise per question type
(exact McNemar, permutation test, Wilson intervals, smallest detectable difference), with cost,
latency and what the correction loop did. `run` and `judge` call real models; each question is
appended to `data/reports/p3-05/<system>/records.jsonl` and a second run skips finished ones.

```bash
uv run python -m rag.eval.agentic run --system both --take single_hop=4 --take two_hop=2 \
    --take colloquial=2 --take no_answer=2          # a pilot of 10 questions first
uv run python -m rag.eval.agentic project            # cost of the full run from the records so far
uv run python -m rag.eval.agentic run --system both --max-cny 4.3   # continues the pilot
uv run python -m rag.eval.agentic judge --runs 2
uv run python -m rag.eval.agentic report --md data/reports/p3-05/report.md
uv run python -m rag.eval.agentic human-sheet --n 30   # blind A/B sheet for the owner
# P3-06: a new graph run on chosen questions, set beside the P3-05 records (A, B before, B after)
uv run python -m rag.eval.agentic run --system agentic --out data/reports/p3-06 --qids ID,ID,...
uv run python -m rag.eval.agentic judge --out data/reports/p3-06
uv run python -m rag.eval.agentic compare --before data/reports/p3-05 --after data/reports/p3-06
```

## Synthetic question sets

Dev and holdout sets generated from the corpus (spec P1-05, design s14-2): random spans of the
cleaned page text (not index chunks), stratified by repository and doc type, one span per page,
the 65 public questions' pages left out. `deepseek-v4-flash` writes one factoid question per span
in the user's own words; `qwen/qwen3.6-flash` rates groundedness, relevance and standalone 1-5
(HF rag_evaluation prompts); a question is kept when all three are at least 4, it does not copy
more than 6 words in a row from its span and its answer is under 300 characters. `generate` and
`critique` cost money and resume where they stopped; run records go to the git-ignored
`data/reports/synthetic/<run_id>/`.

```bash
uv run python -m rag.eval.synthetic sample --kb hf_docs --n 420 --seed 0
uv run python -m rag.eval.synthetic generate --run <run dir> --limit 20     # a pilot first
uv run python -m rag.eval.synthetic critique --run <run dir>
uv run python -m rag.eval.synthetic project --run <run dir> --n 420
uv run python -m rag.eval.synthetic sample --kb hf_docs --n 380 --seed 1 --extend <run dir>
uv run python -m rag.eval.synthetic build --run <run dir> --kb hf_docs --seed 0
uv run python -m rag.eval.retrieval run --kb hf_docs     --questions evals/datasets/hf_docs/synthetic_dev.jsonl --label synthetic-dev
```

`build` writes `evals/datasets/hf_docs/synthetic_{dev,holdout}.jsonl` (page labels, the span and
scores in `meta`) and `synthetic_manifest.json` (models, prompt versions, seeds, thresholds,
counts, sha256). Tune on dev only; report holdout once per phase.
