"""Who is calling and what they may read (design s16-4). Prompt-injection checks live in
`rag.ingest` (they run at ingestion); budgets and rate limits come in P8."""

from rag.security.access import (
    AclResolver,
    Caller,
    ConfigAclResolver,
    Credentials,
    kb_visible,
    visible_kbs,
)

__all__ = [
    "AclResolver",
    "Caller",
    "ConfigAclResolver",
    "Credentials",
    "kb_visible",
    "visible_kbs",
]
