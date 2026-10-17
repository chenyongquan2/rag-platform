"""Caller identity and knowledge-base access (design s16-4; spec P0-09).

Two levels of access, both decided by the server, never by the request:

- Knowledge base: a caller may use a KB when its config's `access.callers` contains `*` or the
  caller's id (P0-01). `list_knowledge_bases` shows only those; for any other KB, search and
  get_context answer `kb_not_found`, the same as for a KB that does not exist.
- Documents: every Elasticsearch read adds a terms clause on `acl_groups` with the caller's
  groups (`rag.retrieve.filters.build_es_filter`, used by search and get_context alike);
  caller filters can only narrow it.

Who the caller is comes from an `AclResolver`. Until P8 there is one local user:
`ConfigAclResolver` returns the caller configured in the settings whatever the transport says
(stdio is a local process; the HTTP endpoints listen on 127.0.0.1). P8 adds a resolver that maps
a verified OAuth token (the MCP SDK's `TokenVerifier`, s16-4) to a caller id and groups; the
transports already pass what they know about the token in `Credentials`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag.config import KnowledgeBaseConfig, ServiceSettings

EVERY_CALLER = "*"


@dataclass(frozen=True)
class Caller:
    """A resolved caller: its id (matched against `access.callers`) and document ACL groups."""

    caller_id: str
    acl_groups: tuple[str, ...]


@dataclass(frozen=True)
class Credentials:
    """What a transport knows about the request's sender. `transport`: `stdio`,
    `streamable-http` or `http`. The token fields are filled from a token the transport has
    already verified (P8); no raw token is passed around."""

    transport: str
    client_id: str | None = None
    subject: str | None = None
    scopes: tuple[str, ...] = ()


@runtime_checkable
class AclResolver(Protocol):
    def resolve(self, credentials: Credentials) -> Caller:
        """The caller for these credentials; raises when they grant nothing (P8)."""
        ...


class ConfigAclResolver:
    """P0: every request is the one caller from the settings (`service_caller`,
    `service_acl_groups`)."""

    def __init__(self, caller_id: str, acl_groups: Sequence[str]) -> None:
        if not caller_id:
            raise ValueError("caller id must not be empty")
        if isinstance(acl_groups, (str, bytes)):
            raise TypeError("acl_groups must be a list of group names, not one string")
        groups = tuple(acl_groups)
        if not groups or not all(isinstance(g, str) and g for g in groups):
            raise ValueError("acl_groups must name at least one non-empty group")
        self.caller = Caller(caller_id=caller_id, acl_groups=groups)

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> ConfigAclResolver:
        return cls(settings.service_caller, settings.service_acl_groups)

    def resolve(self, credentials: Credentials) -> Caller:
        return self.caller


def kb_visible(caller: Caller, cfg: KnowledgeBaseConfig) -> bool:
    """True when the KB's `access.callers` lists `*` or this caller."""
    callers = cfg.access.callers
    return EVERY_CALLER in callers or caller.caller_id in callers


def visible_kbs(
    caller: Caller, configs: Mapping[str, KnowledgeBaseConfig] | Iterable[KnowledgeBaseConfig]
) -> list[KnowledgeBaseConfig]:
    """The KBs this caller may use, sorted by name."""
    items = configs.values() if isinstance(configs, Mapping) else configs
    return sorted((cfg for cfg in items if kb_visible(caller, cfg)), key=lambda cfg: cfg.kb)
