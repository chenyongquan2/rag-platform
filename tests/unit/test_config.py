from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import (
    KnowledgeBaseConfig,
    ServiceSettings,
    StepConfig,
    load_kb_config,
    load_kb_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_KEYS = [
    "ES_URL",
    "REDIS_URL",
    "DOCSTORE_URL",
    "CHECKPOINT_URL",
    "LOCK_URL",
    "OTLP_ENDPOINT",
    "KB_DIR",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "EMBED_BASE_URL",
    "EMBED_API_KEY",
    "CHAT_MODEL",
    "EMBED_MODEL",
    "EMBED_DIM",
    "RERANKER_MODEL",
    "BUDGET_CNY",
    "BUDGET_STOP_CNY",
    "COST_LEDGER_PATH",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_service_settings_defaults(clean_env: pytest.MonkeyPatch) -> None:
    s = ServiceSettings(_env_file=None)
    assert s.es_url == "http://localhost:9200"
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.docstore_url.startswith("file://")
    assert s.checkpoint_url.startswith("sqlite://")
    assert s.lock_url.startswith("file://")
    assert s.otlp_endpoint == "http://localhost:4317"
    assert s.kb_dir == "kb"
    assert s.chat_model == "deepseek-v4-flash"
    assert s.embed_model == "qwen/text-embedding-v4"
    assert s.embed_dim == 1024
    assert s.reranker_model == "BAAI/bge-reranker-base"
    assert s.budget_cny == 30.0
    assert s.budget_stop_cny == 28.0
    assert s.cost_ledger_path == "data/cost/ledger.jsonl"
    assert s.llm_base_url == ""
    assert s.llm_api_key.get_secret_value() == ""
    assert s.embed_base_url == ""
    assert s.embed_api_key.get_secret_value() == ""


def test_service_settings_env_override_and_secret_masking(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "sk-test-not-a-real-key-123456"
    clean_env.setenv("ES_URL", "http://es.example:9200")
    clean_env.setenv("EMBED_DIM", "512")
    clean_env.setenv("LLM_API_KEY", secret)
    env_file = tmp_path / ".env"
    env_file.write_text(f"EMBED_API_KEY={secret}\nREDIS_URL=redis://cache:6379/1\n", "utf-8")

    s = ServiceSettings(_env_file=env_file)

    assert s.es_url == "http://es.example:9200"
    assert s.redis_url == "redis://cache:6379/1"
    assert s.embed_dim == 512
    assert s.llm_api_key.get_secret_value() == secret
    assert s.embed_api_key.get_secret_value() == secret
    assert secret not in repr(s)
    assert secret not in str(s)
    redacted = s.redacted()
    assert secret not in str(redacted)
    assert redacted["llm_api_key"] == "***"
    assert redacted["es_url"] == "http://es.example:9200"


def test_kb_config_minimal_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "my_notes.yaml"
    path.write_text("source: {connector: local_dir, path: data/notes}\n", "utf-8")

    cfg = load_kb_config(path)

    assert cfg.kb == "my_notes"
    assert cfg.source.connector == "local_dir"
    assert cfg.source.model_extra == {"path": "data/notes"}
    assert cfg.language == "auto"
    assert ".md" in cfg.pipelines
    md = cfg.pipelines[".md"]
    assert md.reader is None
    assert [c.name for c in md.clean] == ["md_clean"]
    assert [s.name for s in md.steps] == [
        "markdown_split",
        "token_split",
        "merge_tiny",
        "assign_chunk_ids",
    ]
    assert cfg.metadata.embed == ["title_path"]
    assert cfg.metadata.llm == ["title_path", "source_rel"]
    assert cfg.es.analyzer == "standard"
    assert cfg.retrieval.top_k_per_route == 20
    assert cfg.retrieval.rerank_n == 30
    assert cfg.retrieval.final_k == 8
    assert cfg.retrieval.threshold == 0.3
    assert cfg.eval.dir == "evals/datasets/my_notes"
    assert cfg.access.callers == ["*"]


def test_kb_config_hf_docs_example() -> None:
    cfg = load_kb_config(REPO_ROOT / "kb" / "public" / "hf_docs.yaml")

    assert cfg.kb == "hf_docs"
    assert cfg.source.connector == "hf_dataset"
    assert cfg.source.model_extra["dataset"] == "m-ric/huggingface_doc"
    assert cfg.language == "en"
    steps = cfg.pipelines[".md"].steps
    assert steps[1] == StepConfig(name="token_split", params={"max_tokens": 400})
    assert steps[0] == StepConfig(name="markdown_split", params={})
    assert cfg.eval.dir == "evals/datasets/hf_docs"


def test_kb_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        KnowledgeBaseConfig.model_validate({"retreival": {"final_k": 3}})
    with pytest.raises(ValidationError):
        KnowledgeBaseConfig.model_validate({"retrieval": {"finalk": 3}})
    with pytest.raises(ValidationError):
        KnowledgeBaseConfig.model_validate({"pipelines": {".md": {"steps": [{"a": {}, "b": {}}]}}})
    with pytest.raises(ValidationError):
        KnowledgeBaseConfig.model_validate({"retrieval": {"final_k": 40, "rerank_n": 30}})


def test_load_kb_configs_rejects_duplicate_names(tmp_path: Path) -> None:
    (tmp_path / "public").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "public" / "a.yaml").write_text("kb: same\n", "utf-8")
    (tmp_path / "private" / "b.yaml").write_text("kb: other\n", "utf-8")

    loaded = load_kb_configs(tmp_path)
    assert sorted(loaded) == ["other", "same"]

    (tmp_path / "private" / "c.yaml").write_text("kb: same\n", "utf-8")
    with pytest.raises(ValueError, match="same"):
        load_kb_configs(tmp_path)
