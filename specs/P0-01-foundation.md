# P0-01 Project foundation

Status: in progress
Phase: P0
Depends on: —
Design sections: s5-1, s16-1 (return fields), s16-2, s16-3, s16-7, s16-8 (error categories), s11-1
Calls real models: no

## Goal
A uv-managed Python 3.12 package `rag` with all framework dependencies installed together and locked, typed
service and knowledge-base config, our own data classes, name registries, connection-string factories,
deterministic fakes for tests, test markers and pre-commit (ruff, unit tests, secret scan).

## Interface
- `rag.config.ServiceSettings` (pydantic-settings, reads env and `.env`): `es_url`, `redis_url`, `docstore_url`,
  `checkpoint_url`, `lock_url`, `otlp_endpoint`, `kb_dir`, `llm_base_url`, `llm_api_key: SecretStr`,
  `embed_base_url`, `embed_api_key: SecretStr`, `chat_model`, `embed_model`, `embed_dim`, `reranker_model`,
  `budget_cny`, `budget_stop_cny`, `cost_ledger_path`; `.redacted() -> dict` (secrets masked).
- `rag.config.KnowledgeBaseConfig` (s5-1): `kb`, `source`, `language`, `pipelines[ext] -> {reader, clean, steps}`,
  `metadata{embed, llm}`, `es{analyzer}`, `retrieval{top_k_per_route, rerank_n, final_k, threshold}`, `eval{dir}`,
  `access{callers}`; `load_kb_config(path)`, `load_kb_configs(kb_dir) -> dict[str, KnowledgeBaseConfig]`.
- `rag.types`: `Chunk`, `Citation`, `AnswerSentence`, `RetrievalResult`, `Answer`, `ErrorCategory`.
- `rag.registry`: `Registry[T]` with `register(name)` decorator, `add`, `get`, `create`, `names`;
  `UnknownNameError`, `DuplicateNameError`; module registries `connectors`, `readers`, `steps`.
- `rag.factories`: `make_docstore(url)`, `make_checkpointer(url)`, `make_lock_provider(url)`, `LockProvider`.
  Schemes now: docstore `file://`, checkpoint `sqlite://`/`memory://`, lock `file://`/`memory://`;
  other schemes raise `NotImplementedError` naming the scheme.
- `rag.llama_guard`: `RaisingLLM`, `LLMCallForbiddenError`, `install_llama_guard(embed_model)`.
- `rag.testing`: `FakeEmbedding`, `FakeChatModel`, `RaisingLLM` (re-export), `use_safe_llama_settings(dim)`.

## Acceptance criteria
Each line maps to one test in `tests/unit/`.
- [ ] AC1 Service settings have defaults for every field without env/.env — `test_config.py::test_service_settings_defaults`
- [ ] AC2 Env vars override settings; secrets never appear in repr or `redacted()` — `test_config.py::test_service_settings_env_override_and_secret_masking`
- [ ] AC3 A KB config with only `source` loads with every other field defaulted — `test_config.py::test_kb_config_minimal_yaml_uses_defaults`
- [ ] AC4 `kb/public/hf_docs.yaml` loads; steps with params are normalized — `test_config.py::test_kb_config_hf_docs_example`
- [ ] AC5 Unknown KB config keys and duplicate KB names are rejected — `test_config.py::test_kb_config_rejects_unknown_keys`, `test_config.py::test_load_kb_configs_rejects_duplicate_names`
- [ ] AC6 Data classes carry the s16-1 return fields and flags — `test_types.py::test_retrieval_result_fields_and_flags`, `test_types.py::test_answer_roundtrip_json`
- [ ] AC7 Registry: decorator registration, create with params, clear error on unknown and duplicate names — `test_registry.py::test_register_and_create`, `test_registry.py::test_unknown_name_error_lists_known_names`, `test_registry.py::test_duplicate_name_rejected`
- [ ] AC8 Factories pick implementation by scheme; unsupported schemes raise NotImplementedError — `test_factories.py::test_file_docstore_persists_and_reloads`, `test_factories.py::test_sqlite_checkpointer`, `test_factories.py::test_lock_providers`, `test_factories.py::test_unsupported_schemes_raise`
- [ ] AC9 FakeEmbedding is deterministic, has the configured dim, unit norm — `test_testing_fakes.py::test_fake_embedding_deterministic_and_dim`
- [ ] AC10 FakeChatModel returns scripted responses in order, records calls, fails when exhausted, supports scripted tool calls — `test_testing_fakes.py::test_fake_chat_model_scripted`, `test_testing_fakes.py::test_fake_chat_model_structured_output`
- [ ] AC11 RaisingLLM raises on every call path; tests run with it installed as `Settings.llm` — `test_testing_fakes.py::test_raising_llm_raises`, `test_testing_fakes.py::test_settings_llm_is_raising_stub_in_tests`
- [ ] AC12 Markers `unit`/`integration`/`llm` are registered and the default run excludes integration and llm — `test_project.py::test_pytest_markers_and_default_selection`

Also (checked by running, not by a test): `uv sync` works from the lock; `pre-commit run --all-files` passes.

## Out of scope
Real connectors, readers and steps (P0-04); ES mapping and Docker services (P0-02); embedding client, L0 cache
and cost ledger writer (P0-05); Redis/Postgres implementations of the factories (later phases); OTel setup.

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
