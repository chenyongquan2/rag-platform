"""Caller identity and knowledge-base access (design s16-4; spec P0-09).

Two levels of access, both decided by the server, never by the request:

- Knowledge base: a caller may use a KB when its config's `access.callers` contains `*` or the
  caller's id (P0-01). `list_knowledge_bases` shows only those; for any other KB, search and
  get_context answer `kb_not_found`, the same as for a KB that does not exist.
- Documents: every Elasticsearch read adds a terms clause on `acl_groups` with the caller's
  groups (`rag.retrieve.filters.build_es_filter`, used by search and get_context alike);
  caller filters can only narrow it.

Who the caller is comes from an `AclResolver` (`make_acl_resolver` picks it from the settings):

- `auth_mode=local`: `ConfigAclResolver` returns the caller configured in the settings whatever
  the transport says (stdio is a local process; the HTTP server listens only on the local
  machine in this mode).
- `auth_mode=oauth` (spec P8-01): `ScopeAclResolver` maps the scopes of a token the transport
  has already verified (`rag.security.tokens`) to one configured caller and its groups. A token
  whose scopes name no configured caller, or more than one, is refused (`AccessDenied`, the
  service answers `forbidden`): guessing which of two callers was meant could hand one caller
  the other's documents. stdio keeps the configured local caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag.config import AuthCallerConfig, KnowledgeBaseConfig, ServiceSettings

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


class AccessDenied(Exception):  # noqa: N818 - reads as what happened
    """The credentials grant no caller (no or several mapped scopes, or none at all)."""


class ScopeAclResolver:
    """OAuth: the one configured caller whose scope the token carries (module doc); stdio is
    `local`'s caller."""

    def __init__(self, callers: Sequence[AuthCallerConfig], *, local: ConfigAclResolver) -> None:
        self._by_scope = {
            c.scope: Caller(caller_id=c.caller, acl_groups=tuple(c.acl_groups)) for c in callers
        }
        self._local = local

    def resolve(self, credentials: Credentials) -> Caller:
        if credentials.transport == "stdio":
            return self._local.caller
        matched = {self._by_scope[s] for s in credentials.scopes if s in self._by_scope}
        if len(matched) != 1:
            raise AccessDenied(
                f"token of client {credentials.client_id!r} maps to {len(matched)} callers"
            )
        return matched.pop()


def make_acl_resolver(settings: ServiceSettings) -> AclResolver:
    local = ConfigAclResolver.from_settings(settings)
    if settings.auth_mode == "oauth":
        return ScopeAclResolver(settings.auth_callers, local=local)
    return local


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
