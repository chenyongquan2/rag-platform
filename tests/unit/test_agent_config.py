"""Spec P3-01 AC13: explicit settings and KB config keys of the ask_docs graph."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rag.config import REPO_ROOT, AgentConfig, KnowledgeBaseConfig, load_kb_config
from rag.testing import observe_settings


def test_agent_settings_defaults_and_validation() -> None:
    s = observe_settings()
    assert s.agent_deadline_ms == 30_000
    assert s.agent_answer_reserve_ms == 12_000
    assert s.agent_recursion_limit == 25
    assert s.agent_token_cap == 20_000
    assert s.agent_answer_reserve_tokens == 6_000
    assert s.agent_history_turns == 6
    assert s.session_store_url == "sqlite://data/sessions/sessions.db"
    assert s.session_lock_url == "memory://"
    # The deadline is a total time limit: never above what a caller may ask for.
    with pytest.raises(ValueError):
        observe_settings(agent_deadline_ms=70_000, search_max_deadline_ms=60_000)
    # Time kept for the answer must leave room for the search before it.
    with pytest.raises(ValueError):
        observe_settings(agent_deadline_ms=10_000, agent_answer_reserve_ms=10_000)
    with pytest.raises(ValueError):
        observe_settings(agent_token_cap=5_000, agent_answer_reserve_tokens=5_000)
    with pytest.raises(ValueError):
        observe_settings(agent_recursion_limit=0)
    assert observe_settings(agent_history_turns=0).agent_history_turns == 0


def test_hf_docs_sets_agent_keys() -> None:
    assert AgentConfig().max_rounds == 3
    assert KnowledgeBaseConfig().agent == AgentConfig(max_rounds=3)
    with pytest.raises(ValidationError):
        AgentConfig(max_rounds=0)
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"max_round": 2})

    path = Path(REPO_ROOT) / "kb" / "public" / "hf_docs.yaml"
    raw = yaml.safe_load(path.read_text("utf-8"))
    assert set(raw["agent"]) == set(AgentConfig.model_fields)
    cfg = load_kb_config(path, kb_root=Path(REPO_ROOT) / "kb")
    assert cfg.agent == AgentConfig(max_rounds=3)
