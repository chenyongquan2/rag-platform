"""Index and alias names of one knowledge base (design s7-3, s7-5).

Queries use only the alias `kb_<kb>`; each full rebuild writes a new index `kb_<kb>_v<N>` and the
alias is switched when the new index is ready. A prefix (`test1a2b_`) keeps test indices apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Same rule as KnowledgeBaseConfig.kb; also a valid Elasticsearch index name part.
_KB_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_PREFIX_RE = re.compile(r"([a-z0-9][a-z0-9_-]*)?")


@dataclass(frozen=True)
class KbIndexNames:
    kb: str
    prefix: str = ""

    def __post_init__(self) -> None:
        if not _KB_RE.fullmatch(self.kb):
            raise ValueError(f"invalid knowledge base name for an index: {self.kb!r}")
        if not _PREFIX_RE.fullmatch(self.prefix):
            raise ValueError(
                f"invalid index prefix {self.prefix!r}: lower-case letters, digits, '_' or '-', "
                "not starting with '_' or '-'"
            )

    @property
    def alias(self) -> str:
        """The only name queries use."""
        return f"{self.prefix}kb_{self.kb}"

    @property
    def pattern(self) -> str:
        """Wildcard over the versions. Also matches other KBs (`kb_hf_v*` matches `kb_hf_v2_v1`);
        filter with `version_of`."""
        return f"{self.alias}_v*"

    def index(self, version: int) -> str:
        if version < 1:
            raise ValueError(f"index version must be >= 1, got {version}")
        return f"{self.alias}_v{version}"

    def version_of(self, index: str) -> int | None:
        """Version number when `index` is a versioned index of this knowledge base, else None."""
        match = re.fullmatch(rf"{re.escape(self.alias)}_v([1-9][0-9]*)", index)
        return int(match.group(1)) if match else None
