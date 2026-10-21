"""Chunk steps for PDFs: tables as chunks, table summaries, generated-text flag (spec P6-01,
design s7-2, s5-3).

    markdown_split -> table_chunks -> token_split -> merge_tiny -> mark_generated_text
    -> table_summary -> suspicious_instruction -> assign_chunk_ids

- `table_chunks`: every Markdown table becomes its own chunk (`chunk_kind: table`, `table_id`
  = `<page doc_id>#t<k>`), with the caption paragraph above it ("Table 2. ..."). A table that
  does not fit the token cap is cut between rows into parts; every part starts with the caption
  and the header row, and all parts share the `table_id`. Text around a table stays in
  `chunk_kind: text` chunks; `merge_tiny` never merges a table chunk.
- `table_summary`: one extra chunk per table (`chunk_kind: table_summary`, same `table_id`) whose
  text is a short summary of the table, so a question that does not use the table's own words can
  still find it (the multi-vector index of design s7-2). `mode: model` asks the model through
  `ModelTextService` (cached by table content; an error falls back to the rule text),
  `rule` writes the summary from the caption, the column names and the first column without a
  model, `off` adds nothing. Summary chunks go after the page's other chunks, so the
  neighbours of ordinary chunks do not change. At search time a summary hit is replaced by its
  table chunk (`rag.retrieve.table_groups`).
- `mark_generated_text`: `has_generated_text` is true for a chunk that holds a model-written
  figure description or is a table summary (design s5-3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, ClassVar

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode, TransformComponent

from rag.ingest.chunk import TokenizerFn, _blocks, doc_id_of, metadata_tokens, trim_block
from rag.ingest.describe import IMAGE_MARKER_PREFIX, ModelTextService
from rag.ingest.markdown import split_table, table_header_lines
from rag.ingest.meta import DEFAULT_EMBED_KEYS, DEFAULT_LLM_KEYS, apply_whitelist
from rag.registry import steps

__all__ = [
    "IMAGE_MARKER_PREFIX",
    "MarkGeneratedText",
    "TableChunks",
    "TableSummary",
    "join_table_parts",
    "rule_summary",
    "split_table",
]

CAPTION = re.compile(r"^\s*(?:table|fig(?:ure)?\.?)\s*[A-Z]?\d+|^\s*[表图]\s*\d+", re.IGNORECASE)
"""A paragraph that starts like a caption: `Table 2.`, `Figure 3:`, `Fig. 4`, `表 1`."""
_WHITELIST_CONTEXT = {"embed_keys": "embed_keys", "llm_keys": "llm_keys"}
SUMMARY_MODES = ("model", "rule", "off")
TEXT, TABLE, TABLE_SUMMARY = "text", "table", "table_summary"


def _is_caption(block: str) -> bool:
    text = block.strip()
    return bool(text) and "\n\n" not in text and text.count("\n") <= 2 and bool(CAPTION.match(text))


def _copy(node: BaseNode, text: str, node_id: str, **metadata: Any) -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata={**node.metadata, **metadata},
        relationships=dict(node.relationships),
        metadata_template=node.metadata_template,
        metadata_separator=node.metadata_separator,
        text_template=node.text_template,
    )


@steps.register("table_chunks")
class TableChunks(TransformComponent):
    """Tables become chunks of their own (see the module doc). `max_tokens` is the cap of
    `token_split`: text plus the metadata LlamaIndex puts in front of it."""

    context_params: ClassVar[dict[str, str]] = {
        "tokenizer": "tokenizer",
        "max_tokens": "max_tokens",
        **_WHITELIST_CONTEXT,
    }
    max_tokens: int = Field(gt=0)
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))
    _tokenizer: TokenizerFn = PrivateAttr()

    def __init__(self, *, tokenizer: TokenizerFn | None = None, **data: Any) -> None:
        tokenizer_model = data.pop("tokenizer_model", None)
        super().__init__(**data)
        if tokenizer is None:
            if not tokenizer_model:
                raise ValueError("table_chunks needs a tokenizer (the reranker's)")
            from rag.ingest.tokens import load_tokenizer

            tokenizer = load_tokenizer(tokenizer_model)
        self._tokenizer = tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "table_chunks"

    def _count(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _table_texts(self, node: BaseNode, table: str, caption: str | None) -> list[str]:
        """The text of each chunk of one table: caption and table, or caption and header
        repeated in every part when the table does not fit the cap."""
        budget = self.max_tokens - metadata_tokens(node, self._tokenizer)
        lead = f"{caption.strip()}\n\n" if caption else ""
        whole = lead + table
        if self._count(whole) <= budget:
            return [whole]
        inner = budget - self._count(lead)
        for _ in range(8):  # joined token counts are not exactly additive: tighten and retry
            parts = [lead + part for part in split_table(table, max(inner, 1), self._count)]
            over = max(self._count(p) for p in parts) - budget
            if over <= 0 or inner <= 1:
                return parts
            inner -= over
        return parts

    def _one(self, node: BaseNode, counters: dict[str, int]) -> list[BaseNode]:
        text = node.get_content(MetadataMode.NONE)
        blocks = _blocks(text, "\n\n")
        if not any(kind == TABLE for _, kind in blocks):
            node.metadata["chunk_kind"] = TEXT
            apply_whitelist(node, self.embed_keys, self.llm_keys)
            return [node]
        doc_id = doc_id_of(node)
        out: list[BaseNode] = []
        pending: list[str] = []
        made = 0

        def flush() -> None:
            nonlocal made
            body = trim_block("".join(pending))
            pending.clear()
            if body.strip():
                out.append(_copy(node, body, f"{node.node_id}.x{made}", chunk_kind=TEXT))
                made += 1

        for block, kind in blocks:
            if kind != TABLE:
                pending.append(block)
                continue
            caption: str | None = None
            if pending and _is_caption(pending[-1]):
                caption = pending.pop()
            flush()
            k = counters.get(doc_id, 0)
            counters[doc_id] = k + 1
            table_id = f"{doc_id}#t{k}"
            for part in self._table_texts(node, block.strip("\n"), caption):
                out.append(
                    _copy(
                        node,
                        part,
                        f"{node.node_id}.x{made}",
                        chunk_kind=TABLE,
                        table_id=table_id,
                    )
                )
                made += 1
        flush()
        for item in out:
            apply_whitelist(item, self.embed_keys, self.llm_keys)
        return out

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        counters: dict[str, int] = {}
        return [item for node in nodes for item in self._one(node, counters)]


# ---- summaries --------------------------------------------------------------------------------


def join_table_parts(texts: Sequence[str]) -> tuple[str, str]:
    """`(caption, table)` of one table from the texts of its chunks: the caption of the first
    part, and the rows of all parts with the repeated header rows taken out."""
    caption = ""
    table_lines: list[str] = []
    for i, text in enumerate(texts):
        head, sep, rest = text.partition("\n\n")
        if sep and _is_caption(head):
            if i == 0:
                caption = head.strip()
            text = rest
        lines = text.split("\n")
        if i > 0:
            lines = lines[table_header_lines(lines) :]
        table_lines.extend(lines)
    return caption, "\n".join(table_lines)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def rule_summary(caption: str, table: str, *, max_rows: int = 12) -> str:
    """A summary without a model: caption, column names and the first column's values."""
    lines = [ln for ln in table.split("\n") if ln.strip()]
    head_n = table_header_lines(lines)
    columns = [c for c in _cells(lines[0]) if c] if lines else []
    rows = [_cells(ln) for ln in lines[head_n:]]
    firsts: list[str] = []
    for cells in rows:
        if cells and cells[0] and cells[0] not in firsts:
            firsts.append(cells[0])
    parts = [caption.rstrip(".") + "." if caption else "", f"Columns: {', '.join(columns)}."]
    listed = "; ".join(firsts[:max_rows]) + ("; ..." if len(firsts) > max_rows else "")
    parts.append(f"{len(rows)} rows: {listed}." if firsts else f"{len(rows)} rows.")
    return " ".join(p for p in parts if p)


@steps.register("table_summary")
class TableSummary(TransformComponent):
    """One summary chunk per table (see the module doc). Stats of the last runs: `stats`."""

    mode: str = Field(default="model")
    model: str | None = Field(default=None)
    cache_path: str = Field(default="data/cache/model_text.sqlite3")
    budget_cny: float | None = Field(default=None)
    max_rows: int = Field(default=12, gt=0)
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))
    context_params: ClassVar[dict[str, str]] = {**_WHITELIST_CONTEXT, "settings": "settings"}
    _service: ModelTextService | None = PrivateAttr(default=None)
    _settings: Any = PrivateAttr(default=None)
    _counts: dict[str, int] = PrivateAttr(default_factory=dict)

    def __init__(
        self, *, service: ModelTextService | None = None, settings: Any = None, **data: Any
    ) -> None:
        super().__init__(**data)
        self._settings = settings
        if self.mode not in SUMMARY_MODES:
            raise ValueError(
                f"table_summary mode must be one of {SUMMARY_MODES}, got {self.mode!r}"
            )
        if self.mode == "model" and service is None and not self.model:
            raise ValueError("table_summary mode `model` needs a `model` (or a service)")
        self._service = service
        self._counts = {"tables": 0, "summaries": 0, "model_errors": 0}

    @classmethod
    def class_name(cls) -> str:
        return "table_summary"

    @property
    def stats(self) -> dict[str, Any]:
        return {"mode": self.mode, **self._counts}

    def _model_service(self) -> ModelTextService:
        if self._service is None:
            from rag.ingest.describe import shared_service

            self._service = shared_service(
                self._settings,
                text_model=self.model or "",
                cache_path=self.cache_path,
                budget_cny=self.budget_cny,
            )
        return self._service

    def _summary(self, caption: str, table: str) -> str:
        fallback = rule_summary(caption, table, max_rows=self.max_rows)
        if self.mode == "rule":
            return fallback
        from rag.ingest.describe import DescribeBudgetError

        try:
            text = self._model_service().summarize_table(table, caption=caption)
        except DescribeBudgetError:
            raise
        except Exception:  # the page must still be indexed: the rule text stands in
            self._counts["model_errors"] += 1
            return fallback
        return f"{caption.rstrip()} {text}".strip() if caption else text

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        if self.mode == "off":
            return list(nodes)
        pages: dict[str, list[BaseNode]] = {}
        for node in nodes:
            pages.setdefault(doc_id_of(node), []).append(node)
        out: list[BaseNode] = []
        for page in pages.values():
            out.extend(page)
            tables: dict[str, list[BaseNode]] = {}
            for node in page:
                table_id = node.metadata.get("table_id")
                if table_id and node.metadata.get("chunk_kind") == TABLE:
                    tables.setdefault(str(table_id), []).append(node)
            for table_id, parts in tables.items():
                caption, table = join_table_parts([p.get_content(MetadataMode.NONE) for p in parts])
                self._counts["tables"] += 1
                text = self._summary(caption, table)
                self._counts["summaries"] += 1
                first = parts[0]
                summary = _copy(
                    first,
                    text,
                    f"{first.node_id}.sum",
                    chunk_kind=TABLE_SUMMARY,
                    table_id=table_id,
                    has_generated_text=True,
                )
                apply_whitelist(summary, self.embed_keys, self.llm_keys)
                out.append(summary)
        return out


@steps.register("mark_generated_text")
class MarkGeneratedText(TransformComponent):
    """`has_generated_text` = the chunk holds a model-written figure description, or is a table
    summary (design s5-3: the prompt must be able to say the text was written by a model)."""

    context_params: ClassVar[dict[str, str]] = _WHITELIST_CONTEXT
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))

    @classmethod
    def class_name(cls) -> str:
        return "mark_generated_text"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        for node in nodes:
            generated = bool(node.metadata.get("has_generated_text")) or (
                IMAGE_MARKER_PREFIX in node.get_content(MetadataMode.NONE)
            )
            node.metadata["has_generated_text"] = generated
            apply_whitelist(node, self.embed_keys, self.llm_keys)
        return list(nodes)
