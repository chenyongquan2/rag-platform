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
in `AgentHooks` (`rag.agent.hooks`); the sufficiency judge is still a stub that P3-03 fills in,
the edges, the loop guards and the state exist. It is exposed as the MCP tool
`ask_docs(kb, question, session_id, deadline_ms, resume_token, answer)` and `POST /v1/ask_docs`
(JSON body `{kb, question, session_id?, deadline_ms?, resume_token?, answer?}`), non-streaming,
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
normalised, identifiers, code spans, versions, flags, paths and quoted messages are found, and
the question is classified as clear, vague or undecided and as one lookup or possibly two. The
small model (`AGENT_REWRITE_PROFILE`, default `cheap`, structured output by function calling)
is called only where it can change something, and only if the KB's `agent` section allows it:

- `rewrite` (`agent.rewrite`): a follow-up becomes a standalone question using the session's
  history (this goes to the answer step); the search query is written as a declarative
  statement (`agent.declarative`; this goes to retrieval); a two-hop question becomes at most
  `agent.max_sub_questions` queries, one search each. A statement-form question with no history
  is only normalised: no model call. Every identifier of the question must survive in the
  queries and the language must stay the question's, else the original question is searched.
  A model failure, timeout, unparsable answer or refused call (token cap) does the same. The
  request is never failed: it is answered, `degraded` is set and `fallbacks` names the step and
  reason (`rewrite:timeout`, `rewrite:identifiers_lost`, `route:parse_failed`, ...).
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
  is `invalid_resume_token` (HTTP 400), the same answer for every reason. When the run
  completes, the session history gets the question, the question asked back, the reply and the
  answer as two turns. The request's token cap counts both calls.

| Setting (env var) | Default | Meaning |
|---|---|---|
| `AGENT_DEADLINE_MS` | 30000 | Total time of one request; at most `SEARCH_MAX_DEADLINE_MS` |
| `AGENT_ANSWER_RESERVE_MS` | 12000 | Time kept for writing the answer (below the deadline) |
| `AGENT_RECURSION_LIMIT` | 25 | LangGraph steps of one run (6 for one round, 3 more per further round) |
| `AGENT_TOKEN_CAP` | 20000 | Model tokens of one request |
| `AGENT_ANSWER_RESERVE_TOKENS` | 6000 | Tokens calls before the answer must leave (below the cap) |
| `AGENT_HISTORY_TURNS` | 6 | Past turns of a session given to the graph; 0 = none |
| `AGENT_REWRITE_PROFILE` | `cheap` | Chat profile of the small model (route and rewrite) |
| `AGENT_REWRITE_TIMEOUT_MS` | 6000 | Longest one such call may take (never into the answer's time) |
| `AGENT_REWRITE_HISTORY_TURNS` | 3 | Earlier turns shown to the small model; 0 = none |
| `AGENT_REWRITE_HISTORY_CHARS` | 400 | Characters of each question or answer shown to it |
| `AGENT_RESUME_TTL_S` | 1800 | How long a question asked back waits for its answer |
| `CHECKPOINT_URL` | `sqlite://data/checkpoints/langgraph.db` | Checkpoints (`memory://` for tests) |
| `SESSION_STORE_URL` | `sqlite://data/sessions/sessions.db` | Session history and resume tokens (`memory://` for tests) |
| `SESSION_LOCK_URL` | `memory://` | Session locks (`file://data/locks` for several processes) |
| KB config `agent.max_rounds` | 3 | Retrieval rounds of one request at most |
| KB config `agent.rewrite` | false (`hf_docs`: true) | The rewrite step may call the small model |
| KB config `agent.declarative` | true | Write the search query as a statement |
| KB config `agent.max_sub_questions` | 3 | Most searches of a split question (1: never split) |
| KB config `agent.clarify` | false (`hf_docs`: true) | Ask the caller back when the question is too vague |
| KB config `agent.route_model` | false (`hf_docs`: true) | The route may ask the small model about what rules leave open |

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
