"""The key of the exact cache (L1): what a result depends on, and nothing else (design s12-1).

    fields = make_fields("search", kb="hf_docs", question=q, filters=f, acl_groups=groups,
                         top_k=8, data_generation=gen, config_version=pipeline.config_version)
    key = cache_key(fields)      # "hf_docs:search:<sha256>"

Every field below changes the result of a request, so every field is in the key:

- `data_generation`: the Redis counter that ingestion and an alias switch add 1 to (s7-4, s7-6);
  an incremental update keeps the index version, so only this makes changed pages miss.
- `config_version`: the retrieval settings (KB config, embedding model, reranker; s12-1).
- `answer_version` (ask only): the KB's `agent` and `answer` config, the chat models and
  profiles, the `agent_*` settings and the prompt texts.
- `question`: canonical (see `canonical_question`): case, spaces and closing punctuation only.
- `filters` and `acl`: another filter or another set of ACL groups is another result.
- `top_k`, `language`, `kb`, `kind`, `index_prefix`: the rest of what the request chose.

The Redis key keeps `<kb>:<kind>:` readable in front of the digest, so one knowledge base's
entries can be listed and dropped by prefix (`rag.cache.admin`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rag.agent.rules import normalise, protected_tokens
from rag.answer.prompts import LanguageSetting, answer_language
from rag.config import KnowledgeBaseConfig, ServiceSettings

Kind = Literal["search", "ask"]
SCHEMA_VERSION = 1
VERSION_LENGTH = 16

# Closing punctuation that carries no meaning in a question (after NFKC the full-width marks
# are ASCII, except the ideographic full stop).
_CLOSING = "?!.。… "

# `agent_*` settings that only matter when a call has history (which the cache never serves) or
# that only bound how long a pending question waits.
_AGENT_SETTINGS_NOT_IN_VERSION = frozenset(
    {
        "agent_resume_ttl_s",
        "agent_history_turns",
        "agent_rewrite_history_turns",
        "agent_rewrite_history_chars",
    }
)


@dataclass(frozen=True)
class CacheKeyFields:
    kind: Kind
    kb: str
    index_prefix: str
    data_generation: int
    config_version: str
    answer_version: str
    question: str
    filters: str
    acl: tuple[str, ...]
    top_k: int | None
    language: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["acl"] = list(self.acl)
        return data


def canonical_question(text: str) -> str:
    """The question as the key sees it: NFKC, no invisible characters, one space between words
    (`rag.agent.rules.normalise`), no closing `? ! .`, and lower case - but only when the
    question holds no protected token (identifier, version, flag, path, quoted message: those
    are case sensitive, `LoRA` is not `lora`). No stemming, no stop words: another word is
    another question."""
    text = normalise(text).rstrip(_CLOSING)
    if protected_tokens(text):
        return text
    return text.lower()


def canonical_filters(filters: Mapping[str, Any] | None) -> str:
    """The caller's filters as sorted JSON; a list of values is a set (`terms`): sorted, no
    repeats."""
    clean: dict[str, Any] = {}
    for key, value in dict(filters or {}).items():
        if isinstance(value, (list, tuple)):
            value = sorted(set(value))
        clean[str(key)] = value
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def acl_groups_of(groups: Sequence[str]) -> tuple[str, ...]:
    if isinstance(groups, (str, bytes)):
        raise TypeError("acl_groups must be a list of group names, not one string")
    return tuple(sorted(set(groups)))


def acl_hash(groups: Sequence[str]) -> str:
    """A short hash of a set of ACL groups (shown in logs and used by the semantic cache)."""
    text = json.dumps(list(acl_groups_of(groups)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:VERSION_LENGTH]


def filters_hash(filters: Mapping[str, Any] | None) -> str:
    return hashlib.sha256(canonical_filters(filters).encode("utf-8")).hexdigest()[:VERSION_LENGTH]


def make_fields(
    kind: Kind,
    *,
    kb: str,
    question: str,
    filters: Mapping[str, Any] | None = None,
    acl_groups: Sequence[str] = ("public",),
    top_k: int | None = None,
    data_generation: int = 0,
    config_version: str = "",
    answer_version: str = "",
    index_prefix: str = "",
    language_setting: LanguageSetting = "auto",
) -> CacheKeyFields:
    return CacheKeyFields(
        kind=kind,
        kb=kb,
        index_prefix=index_prefix,
        data_generation=int(data_generation),
        config_version=config_version,
        answer_version=answer_version,
        question=canonical_question(question),
        filters=canonical_filters(filters),
        acl=acl_groups_of(acl_groups),
        top_k=top_k,
        language=answer_language(question, language_setting),
    )


def cache_key(fields: CacheKeyFields) -> str:
    """`<index_prefix><kb>:<kind>:<sha256 of the fields as canonical JSON>`."""
    payload = {"schema": SCHEMA_VERSION, **fields.as_dict()}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{fields.index_prefix}{fields.kb}:{fields.kind}:{digest}"


def prompt_version() -> str:
    """A hash of every prompt text an `ask_docs` answer depends on (answer prompts and the
    route, rewrite and judge prompts)."""
    from rag.agent import judge, rewrite, routing
    from rag.answer import prompts

    parts: dict[str, Any] = {
        "system": prompts.SYSTEM_PROMPTS,
        "user": getattr(prompts, "_USER_TEMPLATES", {}),
        "retry": getattr(prompts, "_RETRY_RULE", {}),
        "not_found": prompts.NOT_FOUND,
        "judge": getattr(judge, "_SYSTEM", ""),
        "rewrite": getattr(rewrite, "_SYSTEM", ""),
        "route": getattr(routing, "_SYSTEM", ""),
    }
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:VERSION_LENGTH]


def answer_config_version(cfg: KnowledgeBaseConfig, settings: ServiceSettings) -> str:
    """What decides an `ask_docs` answer besides the retrieval (that is `config_version`): the
    whole KB config (`agent`, `answer`, `retrieval`), the chat models and their profiles, the
    `agent_*` settings and the prompt texts. First 16 hex digits of a sha256."""
    agent_settings = {
        name: getattr(settings, name)
        for name in type(settings).model_fields
        if name.startswith("agent_") and name not in _AGENT_SETTINGS_NOT_IN_VERSION
    }
    payload = {
        "kb_config": cfg.model_dump(mode="json", exclude={"description"}),
        "chat_model": settings.chat_model,
        "chat_backend": settings.chat_backend,
        "chat_profiles": {
            name: getattr(settings, f"chat_{name}").model_dump(mode="json")
            for name in ("online", "cheap", "offline")
        },
        "agent": agent_settings,
        "prompts": prompt_version(),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:VERSION_LENGTH]
