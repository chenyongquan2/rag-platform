# P0-02 Local infrastructure

Status: review
Phase: P0
Depends on: P0-01
Design sections: s7-3, s7-5, s16-5, s15-1 (Collector), s12-1 (data generation)
Calls real models: no

## Goal
`docker compose up -d` starts Elasticsearch (pinned 9.x minor, Basic, 2 GB heap), Redis (cache, allkeys-lru,
256 MB), the OTel Collector (OTLP in, traces to Jaeger and a local file) and Jaeger, all bound to 127.0.0.1.
`rag.index` creates versioned knowledge-base indices from the s7-3 mapping, switches the `kb_<kb>` alias
atomically, refuses to delete the aliased index, and keeps a per-KB data generation in Redis.

## Interface
- `docker-compose.yml`, `deploy/otel-collector.yaml`, `scripts/dev_up.sh` (start and wait for health).
- `rag.factories`: `make_es_client(url, *, request_timeout=30.0, max_retries=2) -> Elasticsearch` (http, https),
  `make_redis_client(url, *, socket_timeout=5.0) -> redis.Redis` (redis, rediss; str responses).
- `rag.index.mapping`: `VectorIndexOptions(type, m, ef_construction, oversample)`,
  `build_index_body(*, analyzer, dims, vector=VectorIndexOptions()) -> dict`, `index_body_for_kb(cfg, dims)`.
- `rag.index.names`: `KbIndexNames(kb, prefix="")` with `.alias`, `.index(version)`, `.pattern`, `.version_of(index)`.
- `rag.index.manage`: `create_kb_index(es, names, version, body)`, `list_versions(es, names)`,
  `aliased_indices(es, names)`, `switch_alias(es, names, new_index, *, expected_current=...)`,
  `delete_kb_index(es, names, index)`, `init_kb(es, names, body) -> (index, created)`; errors `IndexExistsError`,
  `AliasConflictError`, `AliasedIndexDeletionError`, `IndexStateError`.
- `rag.index.generation`: `get_data_generation(redis, kb, *, key_prefix)`, `incr_data_generation(redis, kb, *, key_prefix)`.
- CLI: `uv run python -m rag.index.cli {init,status,switch} --kb hf_docs` (`switch --version N` also adds 1 to the
  data generation; `--index-prefix` for tests).
- Settings: defaults of `es_url`, `redis_url`, `otlp_endpoint` follow the compose host ports on 127.0.0.1.

## Acceptance criteria
Unit tests in `tests/unit/`, integration tests (marker `integration`) in `tests/integration/`.
- [x] AC1 Mapping matches s7-3: shards 1, replicas 0, refresh 1s, `dynamic: strict`, content/title_path share the analyzer, 1024-dim cosine int8_hnsw with explicit m, ef_construction and rescore oversample 3, keyword/integer/boolean fields — `test_index_mapping.py::test_mapping_matches_design`
- [x] AC2 The analyzer comes from the KB config — `test_index_mapping.py::test_mapping_uses_kb_analyzer`
- [x] AC3 Vector options are validated (dims > 0; rescore only for quantized types; flat has no HNSW params) — `test_index_mapping.py::test_vector_options_validation`
- [x] AC4 Names: `kb_<kb>_v<N>`, alias `kb_<kb>`, prefix, strict version parsing — `test_index_names.py::test_names_and_version_parsing`
- [x] AC5 ES and Redis clients are picked by scheme — `test_factories.py::test_es_and_redis_clients_by_scheme`
- [x] AC6 Settings defaults match the compose ports — `test_config.py::test_service_settings_defaults`
- [x] AC7 A created index has the mapping and settings as stored by ES — `test_index_es.py::test_create_kb_index_applies_mapping`
- [x] AC8 The configured analyzer is applied (`_analyze` on `content` with cjk gives bigrams) — `test_index_es.py::test_analyzer_applied_to_text_fields`
- [x] AC9 Alias switch is atomic: concurrent searches always see exactly one index — `test_index_es.py::test_alias_switch_is_atomic_for_readers`
- [x] AC10 A stale switch fails as a whole and leaves the alias unchanged — `test_index_es.py::test_alias_switch_conflict_changes_nothing`
- [x] AC11 Deleting the aliased index or a foreign name is refused; an old version can be deleted — `test_index_es.py::test_delete_guard`
- [x] AC12 `init` creates v1 and the alias once and is idempotent; `status` shows them; `switch` moves the alias and adds 1 to the generation — `test_index_es.py::test_cli_init_and_status`
- [x] AC13 Data generation: stable reads, +1 per incr, per-KB, never goes back after the key is lost — `test_redis_generation.py::test_data_generation_counter`
- [x] AC14 A span sent over OTLP reaches Jaeger and the local file — `test_otel_collector.py::test_collector_exports_to_jaeger_and_file`
- [x] AC15 Integration tests skip with a clear message when a service is down; test indices and keys use a unique prefix and are removed — `tests/integration/conftest.py` (checked by running with the services stopped)
- [x] AC16 Running services match the compose pins: ES version equals the compose tag, Basic license, 2 GB heap; Redis is 256 MB allkeys-lru — `test_index_es.py::test_es_matches_compose_pin`, `test_redis_generation.py::test_redis_is_an_lru_cache`

## Out of scope
Writing chunks (EsVectorStore, P0-06), queries (P0-07), the rebuild-and-switch procedure and its tests (P2-01),
OTel SDK setup in the app and metrics/logs pipelines (P0-11), Prometheus/Grafana/Langfuse (P5), app images (P8).

## Handoff (filled at the end)
- Files: `docker-compose.yml`, `deploy/otel-collector.yaml`, `scripts/dev_up.sh`, `.gitignore`, `.env.example`, `README.md`, `src/rag/{factories.py,config/settings.py,index/}`, `tests/unit/{test_index_mapping,test_index_names,test_factories,test_config}.py`, `tests/integration/`, `docs/decisions.md`.
- Images: `docker.elastic.co/elasticsearch/elasticsearch:9.5.5` (pulled directly), `redis:8.10.2-alpine`, `otel/opentelemetry-collector-contrib:0.162.0`, `jaegertracing/jaeger:2.22.0` (Docker Hub through the registry mirrors already configured in Docker Desktop). Measured idle memory: ES 2.5 GB, others < 40 MB each.
- Decisions (docs/decisions.md): host ports = standard + 20000 on 127.0.0.1 (other projects hold the standard ports; Windows reserves blocks in 1024-15000); settings use 127.0.0.1 (localhost costs ~2 s per connection); ES security and ML off locally; mapping adds explicit m/ef_construction, `exclude_source_vectors: true`, `dynamic: strict`; data generation seeded with the current time in ms so eviction or a Redis restart never moves it back; Jaeger v2 (v1 is end of life; query API is `/api/v3` only).
- Open issues: concurrent switch/delete of one KB rely on the ingest lock (P0-06). The CLI was run once, so the local ES has an empty `kb_hf_docs_v1` behind `kb_hf_docs`; P0-06 adds fields to `rag.index.mapping` before writing anything outside it (strict mapping). Host drive C: is 98% full (24 GB free). No container memory limits.
- Tests: 39 unit (5 new), 10 integration. Review rounds: pending.
