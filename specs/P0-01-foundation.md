# P0-01 Project foundation

Status: done
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
  `access{callers}`; `load_kb_config(path, *, kb_root="kb")`, `is_public_kb_file(path, kb_root)`, `load_kb_configs(kb_dir) -> dict[str, KnowledgeBaseConfig]`.
- `rag.types`: `Chunk`, `Citation`, `AnswerSentence`, `RetrievalResult`, `Answer`, `ErrorCategory`.
- `rag.registry`: `Registry[T]` with `register(name)` decorator, `add`, `get`, `create`, `names`;
  `UnknownNameError`, `DuplicateNameError`; module registries `connectors`, `readers`, `steps`.
- `rag.factories`: `make_docstore(url, kb)`, `make_checkpointer(url)` (context manager), `make_lock_provider(url)`,
  `LockProvider.lock(name, timeout)`, `LockTimeoutError`, `url_path(url)`. Schemes now: docstore `file://<dir>` (one JSON per KB), checkpoint
  `sqlite://`/`memory://`, lock `file://`/`memory://`; other schemes raise `NotImplementedError` naming the scheme.
- `rag.llama_guard`: `RaisingLLM`, `LLMCallForbiddenError`, `install_llama_guard(embed_model)`.
- `rag.testing`: `FakeEmbedding`, `FakeChatModel`, `RaisingLLM` (re-export), `use_safe_llama_settings(dim)`.

## Acceptance criteria
Each line maps to one test in `tests/unit/`.
- [x] AC1 Service settings have defaults for every field without env/.env — `test_config.py::test_service_settings_defaults`
- [x] AC2 Env vars override settings; secrets (including credentials inside connection strings) never appear in repr or `redacted()`; `budget_stop_cny <= budget_cny` — `test_config.py::test_service_settings_env_override_and_secret_masking`, `test_config.py::test_service_settings_masks_credentials_in_urls`, `test_config.py::test_service_settings_budget_stop_not_above_budget`
- [x] AC3 A KB config under `kb/public/` with only `source` loads with every other field defaulted (access: every caller) — `test_config.py::test_kb_config_minimal_yaml_uses_defaults`
- [x] AC4 `kb/public/hf_docs.yaml` loads; steps with params are normalized — `test_config.py::test_kb_config_hf_docs_example`
- [x] AC5 Unknown KB config keys, duplicate KB names and non-public configs without `access` are rejected — `test_config.py::test_kb_config_rejects_unknown_keys`, `test_config.py::test_kb_config_access_must_be_explicit_outside_public`, `test_config.py::test_kb_config_public_only_directly_under_kb_root`, `test_config.py::test_load_kb_configs_rejects_duplicate_names`
- [x] AC6 Data classes carry the s16-1 return fields and flags; `error_categories` is in the JSON — `test_types.py::test_retrieval_result_fields_and_flags`, `test_types.py::test_answer_roundtrip_json`
- [x] AC7 Registry: decorator registration, create with params, clear error on unknown and duplicate names — `test_registry.py::test_register_and_create`, `test_registry.py::test_unknown_name_error_lists_known_names`, `test_registry.py::test_duplicate_name_rejected`
- [x] AC8 Factories pick implementation by scheme; unsupported schemes raise NotImplementedError — `test_factories.py::test_url_path_keeps_relative_and_absolute_paths`, `test_factories.py::test_file_docstore_persists_and_reloads`, `test_factories.py::test_sqlite_checkpointer`, `test_factories.py::test_lock_providers`, `test_factories.py::test_in_process_lock_registry_does_not_grow`, `test_factories.py::test_unsupported_schemes_raise`
- [x] AC9 FakeEmbedding is deterministic, has the configured dim, unit norm — `test_testing_fakes.py::test_fake_embedding_deterministic_and_dim`
- [x] AC10 FakeChatModel returns scripted responses in order, records calls, fails when exhausted, supports scripted tool calls — `test_testing_fakes.py::test_fake_chat_model_scripted`, `test_testing_fakes.py::test_fake_chat_model_structured_output`
- [x] AC11 RaisingLLM raises on every call path; tests run with it installed as `Settings.llm` — `test_testing_fakes.py::test_raising_llm_raises`, `test_testing_fakes.py::test_settings_llm_is_raising_stub_in_tests`
- [x] AC12 Markers `unit`/`integration`/`llm` are registered and a pytester run shows the default run excludes integration and llm — `test_project.py::test_pytest_markers_and_default_selection`
- [x] AC13 `.env` is git-ignored and the pre-commit hook rejects a staged `.env` — `test_project.py::test_env_file_is_git_ignored`, `test_project.py::test_forbid_env_hook_rejects_staged_env`

Also (checked by running, not by a test): `uv sync` works from the lock; `pre-commit run --all-files` passes.

## Out of scope
Real connectors, readers and steps (P0-04); ES mapping and Docker services (P0-02); embedding client, L0 cache
and cost ledger writer (P0-05); Redis/Postgres implementations of the factories (later phases); OTel setup.

## Handoff (filled at the end)
- Files: `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.env.example`, `.gitignore`, `README.md`, `kb/public/hf_docs.yaml`, `src/rag/{config/,types.py,registry.py,factories.py,llama_guard.py,testing/}`, `tests/{conftest.py,unit/}`, `docs/decisions.md`.
- Resolved versions: llama-index-core 0.14.25, langchain 1.4.5, langchain-core 1.6.11, langchain-openai 1.7.2, langgraph 1.2.15, langgraph-checkpoint-sqlite 3.1.2, mcp 2.3.0, elasticsearch 9.5.1, redis 8.1.0, pydantic 2.14.1, pydantic-settings 2.16.0, sentence-transformers 6.1.0, transformers 5.20.1, torch 2.14.1+cpu, torchvision 0.29.1+cpu, docling 2.140.0, numpy 2.5.4, scipy 1.18.1, opentelemetry-sdk/exporter-otlp 1.45.1, openinference-instrumentation-llama-index 4.6.0, openinference-instrumentation-langchain 0.1.79, filelock 4.2.0, pyyaml 6.0.3; dev: pytest 9.1.1, ruff 0.17.1, pre-commit 4.7.0. Hooks: gitleaks v8.30.1, pre-commit-hooks v6.0.0.
- Decisions (docs/decisions.md): no conflicts, docling/torch stay in main deps; CPU torch index; lock for win32+linux only; added langgraph-checkpoint-sqlite and filelock; gitleaks built with local Go; ruff/pytest hooks via `uv run --frozen`; `access.callers` defaults to `["*"]` only when the path relative to the kb root starts with `public/`, other configs must set it (load error otherwise); docstore URL is a directory with one JSON per KB.
- Open issues: gitleaks hook needs Go on PATH (first install ~7 min). The hook test runs pre-commit in a scratch repo (~7 s of the ~13 s unit run).
- Review rounds: 2 (round 1 CHANGES: URL credential masking, access default, git-based .env tests, pytester marker test, budget check, computed error_categories, lock registry growth; round 2 APPROVE with one note: one public-path rule for both loaders; all fixed). 34 unit tests pass.
