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
    # 6 s since P3-06 (the answer step took at most 6.0 s in the P3-05 run; it was 12 s).
    assert s.agent_answer_reserve_ms == 6_000
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
    assert cfg.agent == AgentConfig(
        max_rounds=3,
        rewrite=True,
        declarative=False,  # P3-06: no rewrite call for a plain question
        max_sub_questions=3,
        clarify=True,
        route_model=True,
        judge=True,
        multi_queries=3,
        hyde=True,
        judge_skip_score=0.98,  # P3-06
    )


def test_rewrite_and_clarify_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec P3-02 AC19: every new parameter is a named key with a checked range."""
    s = observe_settings()
    assert s.agent_rewrite_profile == "cheap"  # the small, short-timeout chat profile
    assert s.agent_rewrite_timeout_ms == 6_000
    assert (s.agent_rewrite_history_turns, s.agent_rewrite_history_chars) == (3, 400)
    assert s.agent_resume_ttl_s == 1_800
    assert observe_settings(agent_rewrite_profile="online").agent_rewrite_profile == "online"
    assert observe_settings(agent_rewrite_history_turns=0).agent_rewrite_history_turns == 0
    for bad in (
        {"agent_rewrite_profile": "huge"},
        {"agent_rewrite_timeout_ms": 0},
        {"agent_rewrite_history_turns": -1},
        {"agent_rewrite_history_chars": 0},
        {"agent_resume_ttl_s": 0},
    ):
        with pytest.raises(ValueError):
            observe_settings(**bad)
    monkeypatch.setenv("AGENT_REWRITE_TIMEOUT_MS", "2500")
    monkeypatch.setenv("AGENT_RESUME_TTL_S", "60")
    from rag.config import ServiceSettings

    env = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert (env.agent_rewrite_timeout_ms, env.agent_resume_ttl_s) == (2_500, 60)

    # Per knowledge base: off unless the KB's file switches them on (hf_docs.yaml does), so a
    # config that says nothing keeps the P3-01 behaviour and calls no model before the answer.
    default = AgentConfig()
    assert (default.rewrite, default.clarify, default.route_model) == (False, False, False)
    assert (default.declarative, default.max_sub_questions) == (True, 3)
    on = AgentConfig(rewrite=True, clarify=True, route_model=True, max_sub_questions=1)
    assert on.max_sub_questions == 1
    for bad_kb in ({"max_sub_questions": 0}, {"max_sub_questions": 9}, {"rewrite": "maybe"}):
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(bad_kb)
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"clarifiy": True})  # a typo is an error, not a no-op


def test_judge_and_correction_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec P3-03 AC22: the judge's switch, profile, timeout, context size and the correction's
    limits are named keys with checked ranges."""
    s = observe_settings()
    assert s.agent_judge_profile == "cheap"
    assert (s.agent_judge_timeout_ms, s.agent_judge_context_tokens) == (9_000, 1_500)
    assert s.agent_round_min_ms == 8_000
    assert observe_settings(agent_judge_profile="online").agent_judge_profile == "online"
    assert observe_settings(agent_round_min_ms=0).agent_round_min_ms == 0
    for bad in (
        {"agent_judge_profile": "huge"},
        {"agent_judge_timeout_ms": 0},
        {"agent_judge_context_tokens": 0},
        {"agent_round_min_ms": -1},
    ):
        with pytest.raises(ValueError):
            observe_settings(**bad)
    monkeypatch.setenv("AGENT_JUDGE_TIMEOUT_MS", "2500")
    from rag.config import ServiceSettings

    env = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert env.agent_judge_timeout_ms == 2_500

    # Per knowledge base: the judge is off unless the KB's file switches it on.
    default = AgentConfig()
    assert (default.judge, default.multi_queries, default.hyde) == (False, 3, True)
    assert AgentConfig(judge=True, multi_queries=1, hyde=False).multi_queries == 1
    for bad_kb in ({"multi_queries": 0}, {"multi_queries": 7}, {"judge": "maybe"}):
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(bad_kb)


def test_time_fix_settings_are_named_keys_with_checked_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec P3-06 AC11: the new limits are explicit settings with defaults and ranges."""
    s = observe_settings()
    assert (s.agent_search_min_ms, s.agent_round_ratio) == (4_000, 0.9)
    assert s.search_rerank_reserve_ms == 2_000
    for bad in (
        {"agent_search_min_ms": -1},
        {"agent_round_ratio": -0.1},
        {"agent_round_ratio": 2.5},
        {"search_rerank_reserve_ms": -1},
    ):
        with pytest.raises(ValueError):
            observe_settings(**bad)
    monkeypatch.setenv("AGENT_SEARCH_MIN_MS", "3500")
    from rag.config import ServiceSettings

    assert ServiceSettings(_env_file=None).agent_search_min_ms == 3_500  # type: ignore[call-arg]

    # Per knowledge base: off unless a KB sets it (the P3-03 behaviour), a score in [0, 1].
    assert AgentConfig().judge_skip_score is None
    assert AgentConfig(judge_skip_score=0.9).judge_skip_score == 0.9
    for bad_score in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            AgentConfig(judge_skip_score=bad_score)
