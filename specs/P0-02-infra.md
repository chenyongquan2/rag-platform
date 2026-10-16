# P0-02 Local infrastructure

Status: in progress
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
  `make_redis_client(url) -> redis.Redis` (redis, rediss).
- `rag.index.mapping`: `VectorIndexOptions(type, m, ef_construction, oversample)`,
  `build_index_body(*, analyzer, dims, vector=VectorIndexOptions()) -> dict`, `index_body_for_kb(cfg, dims)`.
- `rag.index.names`: `KbIndexNames(kb, prefix="")` with `.alias`, `.index(version)`, `.pattern`, `.version_of(index)`.
- `rag.index.manage`: `create_kb_index(es, names, version, body)`, `list_versions(es, names)`,
  `aliased_indices(es, names)`, `switch_alias(es, names, new_index, *, expected_current=...)`,
  `delete_kb_index(es, names, index)`, `init_kb(es, names, body)`; errors `AliasConflictError`,
  `AliasedIndexDeletionError`.
- `rag.index.generation`: `get_data_generation(redis, kb, *, key_prefix)`, `incr_data_generation(redis, kb, *, key_prefix)`.
- CLI: `uv run python -m rag.index.cli init --kb hf_docs` and `status --kb hf_docs` (`--index-prefix` for tests).
- Settings: defaults of `redis_url` and `otlp_endpoint` follow the compose host ports.

## Acceptance criteria
Unit tests in `tests/unit/`, integration tests (marker `integration`) in `tests/integration/`.
- [ ] AC1 Mapping matches s7-3: shards 1, replicas 0, refresh 1s, `dynamic: strict`, content/title_path share the analyzer, 1024-dim cosine int8_hnsw with explicit m, ef_construction and rescore oversample 3, keyword/integer/boolean fields — `test_index_mapping.py::test_mapping_matches_design`
- [ ] AC2 The analyzer comes from the KB config — `test_index_mapping.py::test_mapping_uses_kb_analyzer`
- [ ] AC3 Vector options are validated (dims > 0; rescore only for quantized types; flat has no HNSW params) — `test_index_mapping.py::test_vector_options_validation`
- [ ] AC4 Names: `kb_<kb>_v<N>`, alias `kb_<kb>`, prefix, strict version parsing — `test_index_names.py::test_names_and_version_parsing`
- [ ] AC5 ES and Redis clients are picked by scheme — `test_factories.py::test_es_and_redis_clients_by_scheme`
- [ ] AC6 Settings defaults match the compose ports — `test_config.py::test_service_settings_defaults`
- [ ] AC7 A created index has the mapping and settings as stored by ES — `test_index_es.py::test_create_kb_index_applies_mapping`
- [ ] AC8 The configured analyzer is applied (`_analyze` on `content` with cjk gives bigrams) — `test_index_es.py::test_analyzer_applied_to_text_fields`
- [ ] AC9 Alias switch is atomic: concurrent searches always see exactly one index — `test_index_es.py::test_alias_switch_is_atomic_for_readers`
- [ ] AC10 A stale switch fails as a whole and leaves the alias unchanged — `test_index_es.py::test_alias_switch_conflict_changes_nothing`
- [ ] AC11 Deleting the aliased index or a foreign name is refused; an old version can be deleted — `test_index_es.py::test_delete_guard`
- [ ] AC12 `init` creates v1 and the alias once and is idempotent; `status` shows them — `test_index_es.py::test_cli_init_and_status`
- [ ] AC13 Data generation: stable reads, +1 per incr, per-KB, never goes back after the key is lost — `test_redis_generation.py::test_data_generation_counter`
- [ ] AC14 A span sent over OTLP reaches Jaeger and the local file — `test_otel_collector.py::test_collector_exports_to_jaeger_and_file`
- [ ] AC15 Integration tests skip with a clear message when a service is down; test indices and keys use a unique prefix and are removed — `tests/integration/conftest.py` (checked by running with the services stopped)

## Out of scope
Writing chunks (EsVectorStore, P0-06), queries (P0-07), the rebuild-and-switch procedure and its tests (P2-01),
OTel SDK setup in the app and metrics/logs pipelines (P0-11), Prometheus/Grafana/Langfuse (P5), app images (P8).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
