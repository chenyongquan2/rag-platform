"""Metadata whitelist (design s5-4).

LlamaIndex puts every metadata key in front of the text it embeds and sends to the model, and
subtracts that length from the chunk budget. Only whitelisted, non-empty keys may go in; every
other key (ids, versions, acl groups, flags) is excluded. Steps that change metadata re-apply the
whitelist, so a key added later never leaks into the embedding text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode

DEFAULT_EMBED_KEYS = ("title_path",)
DEFAULT_LLM_KEYS = ("title_path", "source_rel")
TITLE_SEPARATOR = " > "


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def apply_whitelist(node: BaseNode, embed_keys: Sequence[str], llm_keys: Sequence[str]) -> None:
    """Exclude every metadata key outside the whitelists, and every empty value."""
    meta = node.metadata
    node.excluded_embed_metadata_keys = [k for k in meta if k not in embed_keys or _empty(meta[k])]
    node.excluded_llm_metadata_keys = [k for k in meta if k not in llm_keys or _empty(meta[k])]


def add_metadata(node: BaseNode, key: str, value: Any) -> None:
    """Set a key a rule adds; a new key is excluded from embed and LLM text until the
    whitelist is re-applied (after the clean chain, and by the chunk steps)."""
    new = key not in node.metadata
    node.metadata[key] = value
    if new:
        node.excluded_embed_metadata_keys = [*node.excluded_embed_metadata_keys, key]
        node.excluded_llm_metadata_keys = [*node.excluded_llm_metadata_keys, key]


def title_parts(title_path: str) -> list[str]:
    return title_path.split(TITLE_SEPARATOR) if title_path else []


def join_title(parts: Sequence[str]) -> str:
    return TITLE_SEPARATOR.join(parts)
