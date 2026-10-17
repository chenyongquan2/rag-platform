"""Spec P0-09 AC3: caller identity from config and KB visibility (design s16-4)."""

from __future__ import annotations

import pytest

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.security.access import (
    AclResolver,
    Caller,
    ConfigAclResolver,
    Credentials,
    kb_visible,
    visible_kbs,
)
from rag.testing import observe_settings


def test_config_resolver_returns_configured_caller() -> None:
    resolver = ConfigAclResolver.from_settings(
        observe_settings(service_caller="alice", service_acl_groups=["public", "team-a"])
    )
    assert isinstance(resolver, AclResolver)
    expected = Caller(caller_id="alice", acl_groups=("public", "team-a"))
    # P0: one local user; whatever the transport says (no token checks until P8).
    for credentials in (
        Credentials(transport="stdio"),
        Credentials(transport="http"),
        Credentials(transport="streamable-http", client_id="c1", subject="bob", scopes=("x",)),
    ):
        assert resolver.resolve(credentials) == expected

    with pytest.raises(ValueError, match="acl_groups"):
        ConfigAclResolver("alice", [])
    with pytest.raises(TypeError, match="list"):
        ConfigAclResolver("alice", "public")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="caller"):
        ConfigAclResolver("", ["public"])


def _kb(name: str, callers: list[str]) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(kb=name, access=AccessConfig(callers=callers))


def test_visible_kbs_follow_access_callers() -> None:
    alice = Caller(caller_id="alice", acl_groups=("public",))
    configs = {
        "open": _kb("open", ["*"]),
        "mine": _kb("mine", ["alice", "carol"]),
        "theirs": _kb("theirs", ["bob"]),
        "nobody": _kb("nobody", []),
    }
    assert kb_visible(alice, configs["open"])
    assert kb_visible(alice, configs["mine"])
    assert not kb_visible(alice, configs["theirs"])
    assert not kb_visible(alice, configs["nobody"])
    assert [cfg.kb for cfg in visible_kbs(alice, configs)] == ["mine", "open"]
