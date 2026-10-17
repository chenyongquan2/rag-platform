"""Chunking steps (design s7-1, s7-2, s7-4), run inside the IngestionPipeline in this order:

1. `markdown_split`: one node per heading section, with `title_path` and `parent_id`.
2. `token_split`: LlamaIndex SentenceSplitter with the reranker's tokenizer, paragraphs split at a
   blank line, no overlap; fenced code blocks and Markdown tables are kept whole when they fit.
3. `merge_tiny`: chunks below `min_tokens` or with nothing but headings are merged into the next
   chunk (the last one into the previous), never across pages and never over the token cap.
4. `assign_chunk_ids`: ids `doc_id#<n>` numbered over the whole page, `chunk_index`, `parent_id`,
   and the final metadata whitelist. The splitters' own ids restart in every section (`id_func`
   gets the index within one section), so they are only temporary.

The token cap counts chunk text plus the metadata LlamaIndex puts in front of it (the longer of
the embed and LLM strings), measured with the reranker tokenizer, so "question + title path +
chunk" fits in the reranker's 512 tokens when `max_tokens` leaves room for the question.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser.text.sentence import CHUNKING_REGEX, _Split
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    TextNode,
    TransformComponent,
)

from rag.ingest.markdown import (
    FENCE_OPEN,
    heading_title,
    indent_of,
    is_closing_fence,
    is_table_row,
    is_title_only,
    split_fenced,
)
from rag.ingest.meta import (
    DEFAULT_EMBED_KEYS,
    DEFAULT_LLM_KEYS,
    apply_whitelist,
    join_title,
    title_parts,
)
from rag.registry import steps

TokenizerFn = Callable[[str], Sequence[Any]]
_WHITELIST_CONTEXT = {"embed_keys": "embed_keys", "llm_keys": "llm_keys"}


_LEADING_BLANK_LINES = re.compile(r"\A(?:[ \t]*\n)+")


def trim_block(text: str) -> str:
    """Drop blank lines at the start and whitespace at the end, keep the first line's indent.

    `str.strip()` would also remove the indentation of an indented opening fence ("    ```py"
    inside a list item) while its closing fence keeps it, and the fence would no longer close.
    """
    return _LEADING_BLANK_LINES.sub("", text).rstrip()


def doc_id_of(node: BaseNode) -> str:
    return node.ref_doc_id or node.metadata.get("doc_id") or node.node_id


def metadata_tokens(node: BaseNode, tokenizer: TokenizerFn) -> int:
    """Tokens of the metadata text LlamaIndex puts before the chunk (longer of embed and LLM)."""
    embed = node.get_metadata_str(MetadataMode.EMBED)
    llm = node.get_metadata_str(MetadataMode.LLM)
    return len(tokenizer(embed if len(embed) > len(llm) else llm))


# A sentence ends at . ! ? followed by whitespace, or right after a Chinese full stop, ! or ?.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])")


def split_sentences(text: str) -> list[str]:
    """Sentences, each with the whitespace after it; `"".join(...) == text`.

    Used instead of LlamaIndex's default (NLTK punkt): same job for English, also ends sentences
    at Chinese full stops, and does not import NLTK (about 10 s on this machine).
    """
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if start < end < len(text):
            out.append(text[start:end])
            start = end
    if start < len(text):
        out.append(text[start:])
    return out


def _child_id(i: int, node: BaseNode) -> str:
    return f"{node.node_id}.{i}"


# --- markdown_split -------------------------------------------------------------------------


@steps.register("markdown_split")
class MarkdownSplit(MarkdownNodeParser):
    """Split a page at ATX headings; `#` lines inside fenced code are not headings.

    Each section node gets `title_path` (the page's own title path, then the heading chain,
    joined with " > ") and `parent_id` = its section id `doc_id#s<k>` (k counts sections over the
    page). Text before the first heading is a section with the page's title path.
    """

    context_params: ClassVar[dict[str, str]] = _WHITELIST_CONTEXT
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))

    def __init__(self, **data: Any) -> None:
        data.setdefault("include_metadata", True)
        data.setdefault("include_prev_next_rel", False)
        super().__init__(**data)

    @classmethod
    def class_name(cls) -> str:
        return "markdown_split"

    def get_nodes_from_node(self, node: BaseNode) -> list[TextNode]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        base = title_parts(node.metadata.get("title_path") or "")
        sections: list[tuple[str, list[str]]] = []
        stack: list[tuple[int, str]] = []
        lines: list[str] = []
        parts = list(base)

        def flush() -> None:
            body = trim_block("".join(lines))
            if body:
                sections.append((body, parts))

        for seg in split_fenced(text):
            if seg.code:
                lines.append(seg.text)
                continue
            for line in seg.text.splitlines(keepends=True):
                heading = heading_title(line.rstrip("\r\n"))
                if heading is None:
                    lines.append(line)
                    continue
                flush()
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                parts = base + [t for _, t in stack]
                lines = [line]
        flush()

        page = doc_id_of(node)
        nodes: list[TextNode] = []
        for k, (body, path) in enumerate(sections):
            section_id = f"{page}#s{k}"
            (section,) = build_nodes_from_splits(
                [body], node, id_func=lambda i, _doc, sid=section_id: sid
            )
            section.metadata = {
                **node.metadata,
                "title_path": join_title(path),
                "parent_id": section_id,
            }
            apply_whitelist(section, self.embed_keys, self.llm_keys)
            nodes.append(section)
        return nodes


# --- token_split ----------------------------------------------------------------------------


def _blocks(text: str, separator: str) -> list[tuple[str, str]]:
    """Paragraphs, fenced code blocks and tables; `"".join(texts) == text`.

    Separators stay at the start of the following piece (as SentenceSplitter does), and pieces
    that are only whitespace are glued to the next one.
    """
    pieces: list[tuple[str, str]] = []
    for seg in split_fenced(text):
        if seg.code:
            pieces.append((seg.text, "code"))
            continue
        for i, part in enumerate(seg.text.split(separator)):
            piece = part if i == 0 else separator + part
            if not piece:
                continue
            lines = [ln for ln in piece.split("\n") if ln.strip()]
            table = len(lines) >= 2 and all(is_table_row(ln) for ln in lines)
            pieces.append((piece, "table" if table else "text"))
    out: list[tuple[str, str]] = []
    carry = ""
    for piece, kind in pieces:
        if not piece.strip():
            carry += piece
            continue
        out.append((carry + piece, kind))
        carry = ""
    if carry:
        if out:
            last, kind = out[-1]
            out[-1] = (last + carry, kind)
        else:
            out.append((carry, "text"))
    return out


@steps.register("token_split")
class TokenSplit(SentenceSplitter):
    """SentenceSplitter with every parameter set by us (design s7-1).

    `max_tokens` is the cap for text plus metadata; `tokenizer` is the reranker's tokenizer
    (`tokenizer_model`, loaded from the local HF cache when no tokenizer is passed).
    Fenced code blocks and Markdown tables are atomic when they fit; a code block larger than the
    cap is cut at line breaks into parts that each reopen and close the fence; a table larger than
    the cap is cut between rows.
    """

    context_params: ClassVar[dict[str, str]] = {
        "tokenizer": "tokenizer",
        "tokenizer_model": "tokenizer_model",
    }
    tokenizer_model: str | None = Field(default=None)

    def __init__(
        self,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: TokenizerFn | None = None,
        tokenizer_model: str | None = None,
        paragraph_separator: str = "\n\n",
    ) -> None:
        if tokenizer is None:
            if not tokenizer_model:
                raise ValueError(
                    "token_split needs `tokenizer_model` (the reranker) or a tokenizer"
                )
            from rag.ingest.tokens import load_tokenizer

            tokenizer = load_tokenizer(tokenizer_model)
        super().__init__(
            separator=" ",
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            tokenizer=tokenizer,
            paragraph_separator=paragraph_separator,
            chunking_tokenizer_fn=split_sentences,
            secondary_chunking_regex=CHUNKING_REGEX,
            include_metadata=True,
            include_prev_next_rel=False,
            id_func=_child_id,
        )
        self.tokenizer_model = tokenizer_model

    @classmethod
    def class_name(cls) -> str:
        return "token_split"

    @property
    def max_tokens(self) -> int:
        return self.chunk_size

    def _postprocess_chunks(self, chunks: list[str]) -> list[str]:
        # SentenceSplitter strips each chunk; keep the first line's indentation (see trim_block).
        return [t for t in (trim_block(c) for c in chunks) if t.strip()]

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> list[str]:
        # Same as SentenceSplitter, without its print() when the budget gets small.
        budget = self.chunk_size - len(self._tokenizer(metadata_str))
        if budget <= 0:
            raise ValueError(
                f"metadata ({self.chunk_size - budget} tokens) does not fit in max_tokens "
                f"({self.chunk_size}); shorten the title path or raise max_tokens"
            )
        return self._split_text(text, chunk_size=budget)

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        chunks = super()._split_text(text, chunk_size)
        out: list[str] = []
        for chunk in chunks:
            size = self._token_size(chunk)
            if size <= chunk_size or chunk_size <= 8:
                out.append(chunk)
            else:  # token counts of joined pieces are not always additive: split once more
                out.extend(self._split_text(chunk, chunk_size - (size - chunk_size)))
        return out

    def _split(self, text: str, chunk_size: int) -> list[_Split]:
        size = self._token_size(text)
        if size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=size)]
        blocks = _blocks(text, self.paragraph_separator)
        if len(blocks) == 1 and blocks[0][1] == "text":
            return super()._split(text, chunk_size)
        splits: list[_Split] = []
        for block, kind in blocks:
            size = self._token_size(block)
            if size <= chunk_size:
                splits.append(_Split(block, is_sentence=True, token_size=size))
            elif kind == "code":
                splits.extend(self._split_code(block, chunk_size))
            elif kind == "table":
                splits.extend(self._split_lines(block, chunk_size))
            else:
                splits.extend(super()._split(block, chunk_size))
        return splits

    def _split_lines(self, block: str, chunk_size: int) -> list[_Split]:
        splits: list[_Split] = []
        for i, line in enumerate(block.split("\n")):
            piece = line if i == 0 else "\n" + line
            size = self._token_size(piece)
            if size <= chunk_size:
                splits.append(_Split(piece, is_sentence=True, token_size=size))
            else:
                splits.extend(super()._split(piece, chunk_size))
        return splits

    def _split_code(self, block: str, chunk_size: int) -> list[_Split]:
        lead = block[: len(block) - len(block.lstrip("\n"))]
        # The line break after the closing fence stays, or text right after the block would be
        # glued to the fence ("```The loop above ...") and the fence would never close.
        trail = block[len(block.rstrip("\n")) :]
        lines = block[len(lead) : len(block) - len(trail)].split("\n")
        opener, body = lines[0], lines[1:]
        fence = FENCE_OPEN.match(opener)
        fence_chars = fence.group(1) if fence else "```"
        if body and is_closing_fence(body[-1], fence_chars, opener_indent=indent_of(opener)):
            closer, body = body[-1], body[:-1]
        else:
            closer = opener[: len(opener) - len(opener.lstrip())] + fence_chars
        overhead = self._token_size(f"{opener}\n\n{closer}")
        parts: list[list[str]] = []
        current: list[str] = []
        current_size = overhead
        for line in body:
            size = self._token_size(line)
            if overhead + size > chunk_size:  # one line longer than the cap: cut inside it
                if current:
                    parts.append(current)
                    current, current_size = [], overhead
                parts.extend([s.text] for s in super()._split(line, chunk_size - overhead))
                continue
            if current and current_size + size > chunk_size:
                parts.append(current)
                current, current_size = [], overhead
            current.append(line)
            current_size += size
        if current or not parts:
            parts.append(current)

        def wrap(part: list[str], first: bool, last: bool) -> str:
            head = lead if first else "\n\n"
            return head + opener + "\n" + "\n".join(part) + "\n" + closer + (trail if last else "")

        def fit(part: list[str], first: bool, last: bool) -> list[_Split]:
            # Line token counts are an estimate (joining lines can add a token); check the
            # real size and halve a part that is still over the cap.
            text = wrap(part, first, last)
            size = self._token_size(text)
            if size <= chunk_size:
                return [_Split(text, is_sentence=True, token_size=size)]
            if len(part) > 1:
                half = len(part) // 2
                return fit(part[:half], first, False) + fit(part[half:], False, last)
            return super(TokenSplit, self)._split(text, chunk_size)

        last = len(parts) - 1
        return [s for i, part in enumerate(parts) for s in fit(part, i == 0, i == last)]


# --- merge_tiny -----------------------------------------------------------------------------


@steps.register("merge_tiny")
class MergeTiny(TransformComponent):
    """Merge chunks shorter than `min_tokens`, and chunks that are only headings, into the next
    chunk of the same page (the last one into the previous), only when text + metadata stays
    within `max_tokens`.

    Heading-only sections ("# Title" right before "## Part") are the usual case. The merged
    chunk keeps the next chunk's id and parent; its title path is the next chunk's when the tiny
    chunk was only the parent heading, else the common prefix of both title paths.
    """

    context_params: ClassVar[dict[str, str]] = {
        "tokenizer": "tokenizer",
        "max_tokens": "max_tokens",
        **_WHITELIST_CONTEXT,
    }
    min_tokens: int = Field(default=50, ge=0)
    max_tokens: int = Field(gt=0)
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))
    _tokenizer: TokenizerFn = PrivateAttr()

    def __init__(self, *, tokenizer: TokenizerFn | None = None, **data: Any) -> None:
        tokenizer_model = data.pop("tokenizer_model", None)
        super().__init__(**data)
        if tokenizer is None:
            if not tokenizer_model:
                raise ValueError("merge_tiny needs a tokenizer (the reranker's)")
            from rag.ingest.tokens import load_tokenizer

            tokenizer = load_tokenizer(tokenizer_model)
        self._tokenizer = tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "merge_tiny"

    def _tokens(self, node: BaseNode) -> int:
        return len(self._tokenizer(node.get_content(MetadataMode.NONE)))

    def _fits(self, node: BaseNode) -> bool:
        return self._tokens(node) + metadata_tokens(node, self._tokenizer) <= self.max_tokens

    def _merge(self, first: BaseNode, second: BaseNode, *, into_next: bool) -> TextNode:
        base = second if into_next else first
        p1 = title_parts(first.metadata.get("title_path") or "")
        p2 = title_parts(second.metadata.get("title_path") or "")
        first_text = first.get_content(MetadataMode.NONE)
        if into_next and p2[: len(p1)] == p1 and is_title_only(first_text):
            path = p2
        else:
            path = []
            for a, b in zip(p1, p2, strict=False):
                if a != b:
                    break
                path.append(a)
        merged = TextNode(
            id_=base.node_id,
            text=first_text + "\n\n" + second.get_content(MetadataMode.NONE),
            metadata={**base.metadata, "title_path": join_title(path)},
            relationships=dict(base.relationships),
            metadata_template=base.metadata_template,
            metadata_separator=base.metadata_separator,
            text_template=base.text_template,
        )
        apply_whitelist(merged, self.embed_keys, self.llm_keys)
        return merged

    def _merge_page(self, nodes: list[BaseNode]) -> list[BaseNode]:
        out: list[BaseNode] = []
        carry: BaseNode | None = None
        for node in nodes:
            if carry is not None:
                merged = self._merge(carry, node, into_next=True)
                if self._fits(merged):
                    node = merged
                else:
                    out.append(carry)
                carry = None
            if self._tokens(node) < self.min_tokens or is_title_only(
                node.get_content(MetadataMode.NONE)
            ):
                carry = node
            else:
                out.append(node)
        if carry is not None:
            merged = self._merge(out[-1], carry, into_next=False) if out else None
            if merged is not None and self._fits(merged):
                out[-1] = merged
            else:
                out.append(carry)
        return out

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        pages: dict[str, list[BaseNode]] = {}
        for node in nodes:
            pages.setdefault(doc_id_of(node), []).append(node)
        return [n for page in pages.values() for n in self._merge_page(page)]


# --- assign_chunk_ids -----------------------------------------------------------------------


@steps.register("assign_chunk_ids")
class AssignChunkIds(TransformComponent):
    """Final step: ids `doc_id#<n>` in page order over the whole page (n from 0),
    `chunk_index` = n, `doc_id`, `parent_id` (the heading section, else the page), and the
    metadata whitelist. Only the link to the source page is kept: LlamaIndex prev/next links
    would point at the temporary ids (neighbours are found by doc_id + chunk_index, s7-4).
    """

    context_params: ClassVar[dict[str, str]] = _WHITELIST_CONTEXT
    embed_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_EMBED_KEYS))
    llm_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_LLM_KEYS))

    @classmethod
    def class_name(cls) -> str:
        return "assign_chunk_ids"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        pages: dict[str, list[BaseNode]] = {}
        for node in nodes:
            pages.setdefault(doc_id_of(node), []).append(node)
        out: list[BaseNode] = []
        for doc_id, page in pages.items():
            for index, node in enumerate(page):
                node.id_ = f"{doc_id}#{index}"
                node.metadata["doc_id"] = doc_id
                node.metadata["chunk_index"] = index
                node.metadata["parent_id"] = node.metadata.get("parent_id") or doc_id
                source = node.relationships.get(NodeRelationship.SOURCE)
                node.relationships = {NodeRelationship.SOURCE: source} if source else {}
                apply_whitelist(node, self.embed_keys, self.llm_keys)
                out.append(node)
        return out
