"""Every span attribute, resource attribute and metric name this project sets (spec P0-11).

OTel's GenAI naming rules are still changing and have moved to their own repository (design
s15-1). Code elsewhere uses these constants only, so a rename touches this file alone. Names of
our own fields use the `rag.` prefix; names taken from OTel semantic conventions keep theirs.
"""

from __future__ import annotations

# Instrumentation scope of our tracer and meter.
INSTRUMENTATION_SCOPE = "rag.observe"

# Resource (one per process).
SERVICE_NAME = "service.name"
SERVICE_VERSION = "service.version"
GIT_COMMIT = "vcs.ref.head.revision"  # OTel VCS conventions (development status)

# Request and layer spans.
REQUEST_ID = "rag.request_id"
TOOL = "rag.tool"  # search_docs, ask_docs, ...
CALLER = "rag.caller"
KB = "rag.kb"
CONFIG_VERSION = "rag.config_version"  # hash of the KB config used
INDEX_VERSION = "rag.index_version"
DATA_GENERATION = "rag.data_generation"
STEP = "rag.step"  # layer name: retrieve, rerank, embed_query, generate, ...
ROUTE = "rag.route"  # retrieval route: bm25, vector
TOP_K = "rag.top_k"
RERANK_N = "rag.rerank_n"
FINAL_K = "rag.final_k"
THRESHOLD = "rag.threshold"
HITS = "rag.hits"  # chunks returned
DEGRADED = "rag.result.degraded"
RERANK_TIMEOUT = "rag.result.rerank_timeout"
ERROR_CATEGORIES = "rag.result.error_categories"  # flags of the result (design s16-8)
ERROR_CATEGORY = "rag.error.category"  # category of the failure in this span
ERROR_TYPE = "error.type"  # OTel general convention: class of the error
OUTCOME = "rag.outcome"  # ok | error | cancelled

# Context of an answer (design s10-3, s11-3; spec P1-02): set on the `generate` step span.
CONTEXT_FRAGMENTS = "rag.context.fragments"  # numbered fragments given to the model
CONTEXT_TOKENS = "rag.context.tokens"  # their estimated tokens
CONTEXT_BUDGET = "rag.context.budget"  # the token budget they had to fit into
CONTEXT_DROPPED = "rag.context.dropped"  # chunks left out by the budget
CONTEXT_DUPLICATES = "rag.context.duplicates"  # repeated chunk ids removed
CONTEXT_SUSPICIOUS = "rag.context.suspicious"  # fragments flagged suspicious_instruction

# Answer generation and citation check level 1 (design s11-1, s11-2; spec P1-02).
ANSWER_LANGUAGE = "rag.answer.language"  # zh | en
ANSWER_FOUND = "rag.answer.found"
ANSWER_SENTENCES = "rag.answer.sentences"
ANSWER_UNSUPPORTED = "rag.answer.unsupported_sentences"  # sentences without a valid [n]
ANSWER_REMOVED_MARKERS = "rag.answer.removed_markers"  # [n] numbers matching no fragment
ANSWER_ATTEMPTS = "rag.answer.attempts"  # model calls, retry included
ANSWER_PARSE_FAILED = "rag.answer.parse_failed"
ANSWER_MODEL_CALLS = "rag.answer.model_calls"  # cost ledger lines written by this answer
ANSWER_COST_CNY = "rag.answer.cost_cny"  # CNY of those lines
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"  # GenAI conventions
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"  # reasoning tokens included

# Value of ERROR_CATEGORY when an exception has no category (a bug, not a known failure).
UNCLASSIFIED = "unclassified"
OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"
OUTCOME_CANCELLED = "cancelled"  # cancelled task or interrupt, not a failure of the service

# Model usage (from cost ledger lines). Request model and token type follow the GenAI
# conventions; kind and profile are ours.
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"  # input | output
MODEL_KIND = "rag.model.kind"  # embedding | chat
MODEL_PROFILE = "rag.model.profile"  # ingest | query | online | offline | cheap
USAGE_ESTIMATED = "rag.model.usage_estimated"  # tokens estimated, not from the gateway

# Metric names (design s15-1: requests, errors, degraded, rerank timeouts, step latency, and
# model calls, tokens and cost per caller).
METRIC_REQUESTS = "rag.requests"
METRIC_ERRORS = "rag.errors"
METRIC_DEGRADED = "rag.requests.degraded"
METRIC_RERANK_TIMEOUTS = "rag.requests.rerank_timeout"
METRIC_STEP_DURATION = "rag.step.duration"
METRIC_MODEL_CALLS = "rag.model.calls"
METRIC_MODEL_TOKENS = "rag.model.tokens"
METRIC_MODEL_COST = "rag.model.cost"
