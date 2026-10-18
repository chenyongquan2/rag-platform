"""The security eval set (design s11-3, s14-2): planted pages and questions about them.

    evals/datasets/security/
      pages/*.md        HF-style pages, each with one or more planted instructions
      docs.jsonl        one line per page: doc, path, language, source_rel, injection{...}
      questions.jsonl   2-3 questions per page with the reference answer and expected_behaviour

`injection` says what the page plants: the detector categories it must hit (`kinds`), the tricks
used (`techniques`), the tokens and URLs (or e-mail addresses) an answer must never repeat, the
language it demands, the refusal phrases it asks for, and one example reply that follows it.
The loader checks that every planted token and URL is in its page (text decoded from Unicode tag
characters counts), never in a reference answer, and listed in each question's
`must_not_contain`. The set's sha256 covers the two JSONL files
and every page, with Windows line endings normalized to `\\n`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.config import REPO_ROOT, KnowledgeBaseConfig
from rag.data import SourceDocument, make_doc_id
from rag.ingest import KbProcessor
from rag.ingest.injection import Category
from rag.ingest.meta import title_parts
from rag.ingest.tokens import TokenizerFn
from rag.types import Chunk

_TAG_BASE = 0xE0000
_TAG_RUN = re.compile("[" + chr(_TAG_BASE) + "-" + chr(_TAG_BASE + 0x7F) + "]+")

DEFAULT_SET_DIR = REPO_ROOT / "evals" / "datasets" / "security"
SOURCE_VERSION = "security-set"

Technique = Literal[
    "direct_request",
    "disguised_note",
    "html_comment",
    "hidden_element",
    "zero_width",
    "base64",
    "unicode_tags",
    "full_width",
    "delimiter_forgery",
    "fake_system_message",
    "role_play",
]


class SecuritySetError(ValueError):
    """The security set on disk is inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Injection(_Strict):
    kinds: list[Category] = Field(min_length=1)
    """Detector categories the page's raw text must hit."""
    techniques: list[Technique] = Field(min_length=1)
    tokens: list[str] = Field(default_factory=list)
    """Codes the page asks the model to repeat."""
    urls: list[str] = Field(default_factory=list)
    """URLs or e-mail addresses the page wants in an answer or as a destination."""
    demanded_language: str | None = None
    """ISO 639-1 code of the language the page orders answers in."""
    refusal_phrases: list[str] = Field(default_factory=list)
    """Phrases the page wants the model to reply with instead of answering."""
    followed_example: str = Field(min_length=1)
    """A reply (with `[n]` markers) that follows the injection; used by the dry run."""


class SecurityPage(_Strict):
    doc: str = Field(min_length=1)
    path: str = Field(min_length=1)
    language: Literal["en", "zh"]
    source_rel: str = Field(min_length=1)
    injection: Injection
    text: str = ""
    """The page's Markdown, filled in by the loader."""

    @property
    def planted(self) -> list[str]:
        return [*self.injection.tokens, *self.injection.urls]

    @property
    def readable_text(self) -> str:
        """The raw text plus the text hidden in Unicode tag characters, decoded: a planted
        token may only exist in the hidden part."""
        hidden = [
            "".join(chr(ord(c) - _TAG_BASE) for c in m.group(0))
            for m in _TAG_RUN.finditer(self.text)
        ]
        return "\n".join([self.text, *hidden])


class ExpectedBehaviour(_Strict):
    summary: str = Field(min_length=1)
    answer_language: Literal["en", "zh"]
    must_answer: bool
    """The page answers the question: a refusal or not-found reply is a violation."""
    must_not_contain: list[str] = Field(default_factory=list)


class SecurityQuestion(_Strict):
    qid: str = Field(min_length=1)
    doc: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    """Reference answer, from the page's genuine content only."""
    expected_behaviour: ExpectedBehaviour


class SecuritySet(BaseModel):
    path: str
    sha256: str
    pages: dict[str, SecurityPage]
    questions: list[SecurityQuestion]

    def question(self, qid: str) -> SecurityQuestion:
        for q in self.questions:
            if q.qid == qid:
                return q
        raise KeyError(qid)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    try:
        lines = _read_text(path).split("\n")
    except OSError as exc:
        raise SecuritySetError(f"{path}: {exc}") from exc
    rows: list[T] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise SecuritySetError(f"{path.name} line {number}: {exc}") from exc
    return rows


def _check(pages: dict[str, SecurityPage], questions: list[SecurityQuestion]) -> None:
    for page in pages.values():
        readable = page.readable_text
        for item in page.planted:
            if item not in readable:
                raise SecuritySetError(f"{page.doc}: planted {item!r} is not in the page")
    seen: set[str] = set()
    for q in questions:
        if q.qid in seen:
            raise SecuritySetError(f"duplicate qid {q.qid!r}")
        seen.add(q.qid)
        page = pages.get(q.doc)
        if page is None:
            raise SecuritySetError(f"{q.qid}: unknown doc {q.doc!r}")
        missing = [i for i in page.planted if i not in q.expected_behaviour.must_not_contain]
        if missing:
            raise SecuritySetError(f"{q.qid}: must_not_contain lacks {missing}")
        leaked = [i for i in q.expected_behaviour.must_not_contain if i.lower() in q.answer.lower()]
        if leaked:
            raise SecuritySetError(f"{q.qid}: reference answer contains {leaked}")


def load_security_set(directory: str | Path = DEFAULT_SET_DIR) -> SecuritySet:
    """Load and check the set in `directory` (see the module doc)."""
    root = Path(directory)
    docs = _jsonl(root / "docs.jsonl", SecurityPage)
    questions = _jsonl(root / "questions.jsonl", SecurityQuestion)
    digest = hashlib.sha256()
    pages: dict[str, SecurityPage] = {}
    for doc in docs:
        if doc.doc in pages:
            raise SecuritySetError(f"duplicate doc {doc.doc!r}")
        page_path = root / doc.path
        if not page_path.is_file():
            raise SecuritySetError(f"{doc.doc}: missing page {doc.path}")
        pages[doc.doc] = doc.model_copy(update={"text": _read_text(page_path)})
    _check(pages, questions)
    for name in ("docs.jsonl", "questions.jsonl"):
        digest.update(_read_text(root / name).encode("utf-8"))
    for doc in sorted(pages.values(), key=lambda p: p.path):
        digest.update(doc.path.encode("utf-8") + b"\0" + doc.text.encode("utf-8"))
    try:
        shown = root.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown = str(root.resolve())
    return SecuritySet(path=shown, sha256=digest.hexdigest(), pages=pages, questions=questions)


def page_source(page: SecurityPage, kb: str) -> SourceDocument:
    """The page as a connector document of knowledge base `kb`."""
    return SourceDocument(
        doc_id=make_doc_id(kb, page.source_rel),
        kb=kb,
        source_rel=page.source_rel,
        source_url=None,
        ext=Path(page.path).suffix.lower(),
        content=page.text.encode("utf-8"),
        source_version=SOURCE_VERSION,
    )


def planted_chunks(
    page: SecurityPage, kb_config: KnowledgeBaseConfig, *, tokenizer: TokenizerFn | None = None
) -> list[Chunk]:
    """The page read, cleaned and chunked by `kb_config`'s pipeline, as our `Chunk`s (what
    retrieval would return for it; nothing is written to an index)."""
    proc = KbProcessor(kb_config, tokenizer=tokenizer)
    read = proc.read([page_source(page, kb_config.kb)])
    if read.errors:
        raise SecuritySetError(f"{page.doc}: {read.errors}")
    chunks: list[Chunk] = []
    for node in proc.chunk(proc.clean(read.documents)):
        meta = dict(node.metadata)
        chunks.append(
            Chunk(
                chunk_id=node.node_id,
                doc_id=str(meta.pop("doc_id")),
                kb=str(meta.pop("kb", kb_config.kb)),
                text=node.get_content(),
                suspicious_instruction=bool(meta.pop("suspicious_instruction", False)),
                title_path=title_parts(str(meta.pop("title_path", "") or "")),
                source_rel=str(meta.pop("source_rel", "") or ""),
                source_url=meta.pop("source_url", None),
                source_version=SOURCE_VERSION,
                metadata=meta,
            )
        )
    return chunks
