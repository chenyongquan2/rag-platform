"""Who is calling and what they may read (design s16-4). Prompt-injection checks live in
`rag.ingest` (they run at ingestion); bearer tokens are in `rag.security.tokens` (P8-01)."""

from rag.security.access import (
    AccessDenied,
    AclResolver,
    Caller,
    ConfigAclResolver,
    Credentials,
    ScopeAclResolver,
    kb_visible,
    make_acl_resolver,
    visible_kbs,
)

__all__ = [
    "AccessDenied",
    "AclResolver",
    "Caller",
    "ConfigAclResolver",
    "Credentials",
    "ScopeAclResolver",
    "kb_visible",
    "make_acl_resolver",
    "visible_kbs",
]
