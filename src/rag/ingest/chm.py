"""Reader `chm`: a page of an unpacked CHM -> one Markdown Document (spec P4-01, s5-2, s5-4).

The connector (`rag.data.chm`, `chm_dir`) hands over the page's raw bytes and what it knows
about it (`SourceDocument.extra`: language, page key, title path from the contents, charset
hint). The reader decodes the bytes (`decode_html`, which detects lying charsets), converts the
page (`rag.ingest.chm_html`) and fills the s5-4 contract:

- `doc_id` / `source_rel`: `<kb>:<language>/<page file>` / `<language>/<page file>`; `unit`:
  the page file;
- `title_path`: the names from the `.hhc` hierarchy down to the page; for a page the contents do
  not list, its first heading, else its `<title>`, else its file name (each case is counted);
- `language`: from the volume (the connector), else detected from the text;
- `page_key`: the path inside the CHM without extension, lower case; the zh and en version of a
  page share it (design s8-4);
- `images`: the page's image files (`<language>/<path>`; the images themselves are P6);
- `links_to`: page keys of the pages it links to.
The reader keeps counts of what it saw (`ChmReader.stats`: encodings, fallbacks, decode errors,
pages without title, noise removed while reading) for the quality report (design s6-4); they
hold no page text.
"""

from __future__ import annotations

import posixpath
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from rag.data import SourceDocument
from rag.data.chm.encoding import decode_html
from rag.data.chm.toc import page_key_of
from rag.ingest.chm_html import Converted, html_to_markdown
from rag.ingest.language import UNDETERMINED, detect_language
from rag.ingest.meta import DEFAULT_EMBED_KEYS, DEFAULT_LLM_KEYS, apply_whitelist, join_title
from rag.ingest.readers import METADATA_SEPARATOR, METADATA_TEMPLATE, TEXT_TEMPLATE
from rag.registry import readers

__all__ = ["ChmReader", "HtmlRuleStats", "ReadStats", "html_to_markdown"]


@dataclass
class HtmlRuleStats:
    """What one read-time noise rule dropped: pages it hit, elements, visible characters."""

    pages_hit: int = 0
    elements: int = 0
    chars_removed: int = 0


@dataclass
class ReadStats:
    pages: int = 0
    by_language: Counter[str] = field(default_factory=Counter)
    encodings: Counter[str] = field(default_factory=Counter)
    """`<language>:<codec that read the page>`."""
    declared: Counter[str] = field(default_factory=Counter)
    """The charsets pages declare (`none` for a page that declares none)."""
    fallbacks: int = 0
    """Pages not read with the family of their declared charset (aliases are not counted)."""
    decode_errors: int = 0
    """Pages that no codec decoded strictly (read with replacement characters)."""
    replacement_chars: int = 0
    not_in_toc: int = 0
    no_title: int = 0
    no_container: int = 0
    """Pages without `div#help`: the whole body was converted."""
    empty_pages: int = 0
    duplicate_page_keys: list[str] = field(default_factory=list)
    html_rules: dict[str, HtmlRuleStats] = field(default_factory=dict)
    visible_chars: int = 0
    """Letters and digits in the visible text of all pages (fidelity, `lost_characters`)."""
    lost_chars: int = 0
    """Of those, the ones that did not reach the Markdown; should be 0."""
    lossy_pages: list[str] = field(default_factory=list)
    """Pages that lost any (ids; the documented noise is not text and never counts)."""
    images: int = 0
    internal_links: int = 0
    external_links: int = 0
    tables: int = 0
    code_blocks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "by_language": dict(self.by_language),
            "encodings": dict(self.encodings),
            "declared_charsets": dict(self.declared),
            "fallbacks": self.fallbacks,
            "decode_errors": self.decode_errors,
            "replacement_chars": self.replacement_chars,
            "not_in_toc": self.not_in_toc,
            "no_title": self.no_title,
            "no_container": self.no_container,
            "empty_pages": self.empty_pages,
            "duplicate_page_keys": list(self.duplicate_page_keys),
            "html_rules": {k: vars(v).copy() for k, v in self.html_rules.items()},
            "visible_chars": self.visible_chars,
            "lost_chars": self.lost_chars,
            "lossy_pages": list(self.lossy_pages),
            "images": self.images,
            "internal_links": self.internal_links,
            "external_links": self.external_links,
            "tables": self.tables,
            "code_blocks": self.code_blocks,
        }


def _volume_path(source_rel: str) -> tuple[str, str]:
    """`(language folder, path inside the CHM)` of `en/sub/page.htm`: `("en", "sub/page.htm")`."""
    head, sep, rest = source_rel.partition("/")
    return (head, rest) if sep else ("", source_rel)


@readers.register("chm")
class ChmReader(BaseReader):
    """One CHM page -> one Document (see the module doc). `language` is the knowledge base's
    (`und` for `auto`): used only when neither the connector nor the text says more."""

    def __init__(
        self,
        *,
        kb: str,
        pipeline_version: str,
        language: str = UNDETERMINED,
        acl_groups: Sequence[str] = ("public",),
        embed_keys: Sequence[str] = DEFAULT_EMBED_KEYS,
        llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
    ) -> None:
        self.kb = kb
        self.pipeline_version = pipeline_version
        self.language = language
        self.acl_groups = list(acl_groups)
        self.embed_keys = tuple(embed_keys)
        self.llm_keys = tuple(llm_keys)
        self.stats = ReadStats()
        self._seen: set[tuple[str, str]] = set()

    def load_data(self, source: SourceDocument, **_: Any) -> list[Document]:  # type: ignore[override]
        folder, inner = _volume_path(source.source_rel)
        extra = source.extra
        page_key = extra.get("page_key") or page_key_of(inner)
        decoded = decode_html(source.content, hint=extra.get("charset_hint"))
        converted = html_to_markdown(
            decoded.text, page_dir=posixpath.dirname(inner), self_key=page_key
        )
        language = extra.get("language") or self._detect(converted)
        title_path = self._title_path(extra.get("toc_path"), converted, inner)
        self._count(source, language, page_key, decoded, converted, extra.get("toc_path"))
        metadata: dict[str, Any] = {
            "kb": self.kb,
            "doc_id": source.doc_id,
            "title_path": title_path,
            "source_rel": source.source_rel,
            "unit": posixpath.basename(inner),
            "source_url": source.source_url,
            "doc_type": "chm",
            "language": language,
            "acl_groups": list(self.acl_groups),
            "images": [f"{folder}/{p}" if folder else p for p in converted.images],
            "pipeline_version": self.pipeline_version,
            "page_key": page_key,
            "links_to": list(converted.links_to),
        }
        doc = Document(
            id_=source.doc_id,
            text=converted.markdown,
            metadata=metadata,
            metadata_template=METADATA_TEMPLATE,
            metadata_separator=METADATA_SEPARATOR,
            text_template=TEXT_TEMPLATE,
        )
        apply_whitelist(doc, self.embed_keys, self.llm_keys)
        return [doc]

    def _detect(self, converted: Converted) -> str:
        if self.language != UNDETERMINED:
            return self.language
        return detect_language(converted.markdown)

    def _title_path(self, toc_path: Sequence[str] | None, converted: Converted, inner: str) -> str:
        if toc_path:
            return join_title(list(toc_path))
        return converted.heading or converted.html_title or posixpath.splitext(inner)[0]

    def _count(
        self,
        source: SourceDocument,
        language: str,
        page_key: str | None,
        decoded: Any,
        converted: Converted,
        toc_path: Sequence[str] | None,
    ) -> None:
        stats = self.stats
        stats.pages += 1
        stats.by_language[language] += 1
        stats.encodings[f"{language}:{decoded.encoding}"] += 1
        stats.declared[decoded.declared or "none"] += 1
        stats.fallbacks += decoded.fallback
        stats.decode_errors += decoded.replaced > 0
        stats.replacement_chars += decoded.replaced
        stats.not_in_toc += not toc_path
        stats.no_title += not (toc_path or converted.heading or converted.html_title)
        stats.no_container += not converted.has_container
        stats.empty_pages += not converted.markdown
        key = (language, page_key or source.doc_id)
        if key in self._seen:
            stats.duplicate_page_keys.append(source.doc_id)
        self._seen.add(key)
        for name, (elements, chars) in converted.removed.items():
            rule = stats.html_rules.setdefault(name, HtmlRuleStats())
            rule.pages_hit += 1
            rule.elements += elements
            rule.chars_removed += chars
        stats.visible_chars += converted.visible_chars
        stats.lost_chars += converted.lost_chars
        if converted.lost_chars:
            stats.lossy_pages.append(source.doc_id)
        stats.images += converted.image_markers
        stats.internal_links += converted.internal_links
        stats.external_links += converted.external_links
        stats.tables += converted.tables
        stats.code_blocks += converted.code_blocks
