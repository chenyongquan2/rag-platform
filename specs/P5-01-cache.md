# P5-01 Caching: L1 exact cache and L2 semantic cache in shadow mode

Status: review
Phase: P5
Depends on: P3-06, P2-01
Design sections: s12-1, s12-2, s12-3, s12-4, s16-8, s7-4, s7-5, s7-6, s15-1, s14-4
Calls real models: yes (query embeddings of generated questions, about 0.01 CNY; budget 0.5 CNY for this spec)

## Goal
`search_docs` and `ask_docs` (also the stream) answer an identical request from Redis without running retrieval or the model; a wrong or incomplete result can never come out of the cache. An L2 semantic cache runs in shadow mode: it records what it would have returned, never returns it. A counter-example test set and a report prove the numbers.

## Interface
- `rag.cache.keys`: `CacheKeyFields` (frozen dataclass), `cache_key(fields) -> str`, `canonical_question(text) -> str`, `acl_hash`, `filters_hash`, `answer_config_version(cfg, settings)`, `prompt_version()`.
- `rag.cache.store`: `CacheBackend` protocol (`get/set/set_nx/release/scan_delete/generation`), `RedisCacheBackend`, `MemoryCacheBackend` (fake clock), `NullCacheBackend`, `CircuitBreaker`, `CacheUnavailable`.
- `rag.cache.exact.ExactCache`: `lookup(...) -> Lookup`, `store(...)`, `run_sync(...)`, `run_async(...)` (single-flight), `drop_kb(kb) -> int`.
- `rag.cache.rules`: `search_cacheable(result) -> Verdict`, `ask_cacheable(result) -> Verdict` (reason string when not cacheable).
- `rag.cache.semantic`: `entity_guard(a, b)`, `MemorySemanticStore`, `EsSemanticStore`, `ShadowRecorder`; `rag.cache.report`: `shadow_report(paths)`; `rag.cache.pairs`: generator and evaluator; CLI `python -m rag.cache {drop,shadow-report,pairs,bench}`.
- Settings (all explicit): `cache_mode` (`off|l1|l1_shadow`, default `l1_shadow` in `SearchService.from_settings`; a bare `SearchService(...)` has no cache), `cache_url` (empty = `redis_url`), `cache_key_prefix`, `cache_timeout_ms` (50), `cache_ttl_s`, `cache_not_found_ttl_s`, `cache_max_value_bytes`, `cache_breaker_failures`, `cache_breaker_open_s`, `cache_lock_ttl_ms`, `cache_lock_wait_ms`, `cache_semantic_threshold`, `cache_semantic_ttl_s`, `cache_semantic_store` (`es|memory`), `cache_semantic_lexical_guard`, `cache_shadow_dir`.
- Response field `cache: {layer, hit, outcome, age_s}` on `SearchResponse` and `AskResponse` (null when no cache layer), `cache` in the stream's `done` event only when a cache layer is active.

## Key schema (s12-1, s12-3)
Redis key `<cache_key_prefix><index_prefix><kb>:<kind>:<sha256 of the canonical JSON of the fields>`; `kind` is `search` or `ask`.
Fields: `kind`, `kb`, `index_prefix`, `data_generation` (Redis counter of s7-4, read before every lookup), `config_version` (`search_config_version`: KB config, embedding model, reranker), `answer_version` (ask only: hash of the KB `agent` and `answer` config, the chat model ids and profiles, the `agent_*` settings, the prompt texts), `question` (canonical: NFKC, no invisible characters, one space, trailing `? ! .` removed; lower-cased only when the question has no protected token, so `LoRA` and `lora` stay apart; no stemming), `filters` (sorted JSON), `acl` (sorted unique groups), `top_k` (search), `language` (answer language of the question and KB setting). Value: JSON `{v, kind, created_at, key_fields, payload}`; payload = our data class without request-specific parts (no request id, no routes, no timings of the first run). Compression off; larger than `cache_max_value_bytes` is not stored.

## Rules
- Never write: `degraded`, `rerank_timeout`, `answer_parse_failed`, `coverage_unconfirmed`, any error category, `need_clarification`, `clarified`, `used_history`, `cache_allowed=false`, a result with retrieval or generation error. Cached: complete `not_found` results (shorter TTL), clean answers.
- `ask_docs` with `session_id` or `resume_token` never reads the cache (a hit would skip the history write); it may write when `cache_allowed`.
- Stream: a hit emits `progress{node: cache, phase: hit, layer: l1}`, `answer`, `done` (no tokens); a miss runs normally and writes only a clean complete run.
- Invalidation: by key (generation, versions); no delete. Admin: `drop_kb` removes a KB's keys with SCAN + UNLINK.
- Failure: Redis errors and timeouts never fail a request (outcome `bypass` or `error`, counter, span attribute); a circuit breaker stops trying for `cache_breaker_open_s` after `cache_breaker_failures` failures in a row.
- Single flight: one in-process future per key plus `SET NX` lock in Redis; followers wait (bounded by the request deadline), then use the leader's result or compute themselves when the leader failed.
- L2 (shadow, `search_docs` only): on an L1 miss with a clean fresh result, take the question vector from the L0 cache (no second model call), look up near stored questions with the same kb, filters, ACL, config version, generation and top_k; accept a candidate only when distance is within `cache_semantic_threshold` AND the entity guard passes (protected tokens from `rag.agent.rules`, names, numbers, negation words equal; compared without case) AND the lexical guard passes (content words differ by more than two words; my addition, switchable); write one JSONL record per miss to `cache_shadow_dir`; then store the fresh question. The returned result is never changed. Wrong hit = the cached top-1 chunk differs from the fresh top-1, or a gold page present in the fresh list is missing in the cached one (no gold: top-1 only).

## Acceptance criteria
Tests in `tests/unit/test_cache_*.py` unless named. Integration tests need the compose Redis and Elasticsearch.
- [x] AC1 Every key field changes the key; equal sets of groups and reordered filters do not; canonical question rules — `test_cache_keys.py`
- [x] AC2 Memory and Redis-shaped backends: TTL, size limit, set_nx, prefix delete with SCAN only; breaker states with a fake clock — `test_cache_store.py`
- [x] AC3 One test per never-cache rule, cached kinds, payload round trip without request-specific data — `test_cache_rules.py`
- [x] AC4 Single flight: N concurrent identical requests give one computation (threads, asyncio, two "processes" on one backend); leader failure and cancellation — `test_cache_flight.py`
- [x] AC5 Redis down or slow never raises; bypass and error outcomes; breaker opens and recovers — `test_cache_exact.py`
- [x] AC6 `search_docs`: hit skips the pipeline, ACL groups and filters isolate, generation bump misses, degraded result not stored, response carries `cache`, spans, metrics, business record — `test_cache_service.py`; settings and factory — `test_cache_settings.py`; drop — `test_cache_admin.py`
- [x] AC7 `ask_docs` and stream: hit skips the graph and the model, history and resume never read, stream hit shape, only clean runs written — `test_cache_ask.py`
- [x] AC8 Entity guard accepts only equal entities; 474 counter-example pairs give zero L1 false hits by the key and a measured guard false-accept — `test_cache_guard.py`, `test_cache_pairs.py`, `test_cache_pair_eval.py`
- [x] AC9 Shadow mode: output byte-identical with the layer on and off; records written; would-hit, wrong-hit, Wilson intervals; report command — `test_cache_semantic.py`, `test_cache_report.py`
- [x] AC10 Pair generator is deterministic, 474 counter and 100 positive pairs, file loads and matches the generator — `test_cache_pairs.py`; replay workload and summary — `test_cache_bench.py`
- [x] AC12 (review round 1) A follower that times out or is cancelled leaves the leader and the other followers alone; only results the cache would store are shared, never a resume token; the wait for another process is capped by `cache_lock_wait_ms` and ends when the lock is released — `test_cache_flight_review.py`
- [x] AC13 (review round 1) The shadow lookup runs on one background worker: a slow store does not slow a request, a full queue drops work, a failing store is tried less and logged once without a traceback; ES entries expire; the data generation is read from `redis_url`; integration tests that start a server use `server_env()` — `test_cache_shadow_async.py`, `test_cache_semantic.py`, `test_cache_settings.py`, `test_cache_integration_guard.py`
- [x] AC14 (review round 1) Guard covers identifier order, question words, modals, Chinese opposites; second pair slice reported on its own; per-KB opt-out and `cache_ask_answers`; `--final` does not overwrite — `test_cache_guard.py`, `test_cache_pairs.py`, `test_cache_pair_eval.py`, `test_cache_service.py`, `test_cache_ask.py`
- [x] AC11 Real Redis (unique prefix) and real ES (`test<run>_` index) — `tests/integration/test_cache_redis.py`, `test_cache_semantic_es.py`

## Out of scope
L3 answer cache; turning L2 on (needs the wrong-hit interval, see the report); alert rules and dashboards (P5-02); MT5 question set (local only, not used); a semantic cache for `ask_docs` (design s12-2: it caches retrieval results; the graph has no single one).

## Handoff
- Files: `src/rag/cache/` (`keys, store, rules, exact, layer, factory, semantic, report, pairs, pair_eval, pair_embed, bench, measure, info, cli, __main__`), `src/rag/serve/{service,schema,cli}.py`, `src/rag/retrieve/{search,pipeline}.py` (`query_vector`; `cache` excluded from the config version), `src/rag/config/{settings,kb}.py` (`cache_*`, KB key `cache`), `src/rag/observe/{attributes,metrics}.py`, `src/rag/testing/server.py` (`server_env`), `evals/datasets/hf_docs/cache_pairs.jsonl` (737 pairs, slices v1 and v2), tests `test_cache_*.py` and `tests/integration/test_cache_{redis,semantic_es}.py`, `README.md`, `.env.example`, `docs/decisions.md`, `docs/reports/P5-report.md` section 1.
- Decisions: `docs/decisions.md` (P5-01 and its review round 1). Deviations from the design: `ask_docs` answers are cached by L1 (`cache_ask_answers`, default on; design 12.1 has no L3); a lexical guard and an identifier-order check are added to the entity guard; the semantic store defaults to memory (ES version purges expired entries); the shadow runs on a background worker.
- Numbers (details and caveats in the report): replay 72.5% hit rate (the definition's maximum for that workload, no evidence about real traffic), hit 10 ms against miss 3.1 s, lookup plus write about 6 ms; 474 + 163 counter-example pairs, 0 L1 false hits; second slice false accepts before and after the guard change: dev 76 and 0 of 99, holdout 49 and 0 of 64 (the zero is by construction); the first slice's old holdout number is not final. Spent about 0.038 CNY.
- Open issues: L2 stays off; the guard still has blind spots (single-letter terms such as `A to B`, `who` and `whom`, scope words, synonyms); real-traffic hit rate unknown; a slow (not down) Redis tested only with fakes; no two-process test with real server processes.
- Review rounds: 1 (CHANGES: 2 must-fix, 7 should-fix, nits; all addressed, see report 1.11).
