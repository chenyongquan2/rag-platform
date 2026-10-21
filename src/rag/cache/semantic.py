# ruff: noqa: SIM905, RUF001 (word lists are written as one string; curly apostrophes are matched on purpose)
"""L2 semantic cache in shadow mode (design s12-1, s12-2, s12-3; spec P5-01).

What it does, for a `search_docs` request that missed L1 and was computed (complete result):

1. takes the question's vector from the L0 cache (the retrieval just computed it: no second
   model call);
2. looks for the nearest stored questions of the same partition (knowledge base, filters hash,
   ACL groups hash, config version, data generation, top_k, language: all exact matches, the
   rules of L1 with the question left out) within `threshold` cosine distance;
3. accepts a candidate only if the entity guard passes (identifiers, names, numbers and negations
   equal, `entity_guard`); a question without any entity needs a distance within
   `threshold * strict_factor` (design: "tighten the threshold for questions with no proper noun");
4. writes one JSONL record: what would have hit (candidate, distance, guard verdict) and whether
   the stored question's top chunks equal the fresh result's top chunks;
5. stores the fresh question and its top chunk ids for later questions (not when it would have
   hit: a live L2 would have served the stored entry and written nothing).

It never changes the returned result. Every failure here is logged and ignored by the caller
(`ResultCache._shadow`).

Stores: `MemorySemanticStore` (numpy matrix per partition, this process) and `EsSemanticStore`
(index `cache_semantic_<kb>`, design s12-1). Distance is cosine distance, 1 - cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rag.agent.rules import normalise, protected_tokens
from rag.answer.prompts import answer_language
from rag.cache.keys import acl_hash, canonical_question, filters_hash
from rag.observe import attributes as attr
from rag.observe import get_observability
from rag.observe.rotating import RotatingJsonlWriter
from rag.types import RetrievalResult

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from rag.config import ServiceSettings

logger = logging.getLogger(__name__)

TOP_IDS = 10
"""How many chunk ids of a result are kept and compared."""
STRICT_FACTOR = 0.6
"""A question with no entity of its own must be this much closer (times the threshold)."""
QUESTION_CHARS = 120
"""Longest question text written to a shadow record (the hash identifies the rest)."""

# ---- the entity guard ------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w])")
_NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve twenty thirty fifty hundred "
    "thousand first second third fourth fifth".split()
)
_ZH_NUMBER_CHARS = frozenset("二三四五六七八九十百千万两")
_NEGATION_WORDS = frozenset(
    "not no never none nothing without cannot neither nor unsupported disable disabled except "
    "excluding".split()
)
_NEGATION_N_T = re.compile(r"\b\w+n['’]t\b", re.IGNORECASE)
_ZH_NEGATIONS = frozenset("不没无非未别勿否")
_NAME = re.compile(r"[A-Z][A-Za-z0-9]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
# Words that start a question and are capitalised only because they start it.
_STARTERS = frozenset(
    "how what why when where which who whom whose can could does do did is are was were should "
    "will would shall may might must have has had i we you please tell explain show give list "
    "describe the a an in on for to my our if there any it this that these those with from "
    "and or but so then also let let's".split()
)


@dataclass(frozen=True)
class Signature:
    """What two questions must have in common for a stored one to stand in for a new one."""

    tokens: frozenset[str]
    names: frozenset[str]
    numbers: tuple[str, ...]
    negations: frozenset[str]

    @property
    def has_entities(self) -> bool:
        return bool(self.tokens or self.names or self.numbers)


def _names(text: str) -> frozenset[str]:
    names: set[str] = set()
    words = text.split()
    for position, word in enumerate(words):
        word = word.strip(".,;:!?()[]{}\"'")
        if not _NAME.fullmatch(word):
            continue
        if position == 0 and word.lower() in _STARTERS:
            continue
        names.add(word)
    return frozenset(names)


def entity_signature(question: str) -> Signature:
    """Identifiers (`rag.agent.rules.protected_tokens`), capitalised names, numbers (digits,
    number words, Chinese numerals) and negation words of a question."""
    text = normalise(question)
    tokens = frozenset(str(t) for t in protected_tokens(text))
    names = _names(text)
    # Numbers inside a protected token (`4.36.0`, `T5`) are covered by the token itself.
    outside = text
    for token in sorted(tokens, key=len, reverse=True):
        outside = outside.replace(token, " ")
    numbers = sorted(
        [*_NUMBER.findall(outside)]
        + [w.lower() for w in _WORD.findall(outside) if w.lower() in _NUMBER_WORDS]
        + [c for c in outside if c in _ZH_NUMBER_CHARS]
    )
    lowered = [w.lower() for w in _WORD.findall(text)]
    negations = {w for w in lowered if w in _NEGATION_WORDS}
    negations.update(m.group(0).lower() for m in _NEGATION_N_T.finditer(text))
    negations.update(c for c in text if c in _ZH_NEGATIONS)
    return Signature(tokens, names, tuple(numbers), frozenset(negations))


@dataclass(frozen=True)
class GuardVerdict:
    ok: bool
    reason: str = ""
    entities: bool = True
    """False: neither question has an identifier, name or number (the distance rule is stricter)."""


_WORD_FORM = re.compile(r"[\w.\-/]+")


def _words_of(question: str) -> frozenset[str]:
    """The lower-cased words of a question as written (identifiers stay whole, a closing full
    stop is dropped): what an entity of the other question has to be found among."""
    text = normalise(question).casefold()
    return frozenset(w.rstrip(".") for w in _WORD_FORM.findall(text))


def entity_guard(new_question: str, cached_question: str) -> GuardVerdict:
    """Accept only when identifiers, names, numbers and negations are equal (design s12-3).
    Identifiers and names are compared without regard to case (`LoRA` is `lora`, a casual
    rewrite writes names in lower case): each one of a question has to be a whole word of the
    other. Numbers and negations have to be equal."""
    a, b = entity_signature(new_question), entity_signature(cached_question)
    entities = a.has_entities or b.has_entities
    words_a, words_b = _words_of(new_question), _words_of(cached_question)
    text_a, text_b = normalise(new_question).casefold(), normalise(cached_question).casefold()

    def found_in(entities_of: frozenset[str], words: frozenset[str], text: str) -> bool:
        # A quoted message has spaces in it: it has to be found as a piece of the text.
        return all(
            (e.casefold() in text) if " " in e else (e.casefold().rstrip(".") in words)
            for e in entities_of
        )

    if not (found_in(a.tokens, words_b, text_b) and found_in(b.tokens, words_a, text_a)):
        return GuardVerdict(False, "identifiers", entities)
    if not (found_in(a.names, words_b, text_b) and found_in(b.names, words_a, text_a)):
        return GuardVerdict(False, "names", entities)
    if a.numbers != b.numbers:
        return GuardVerdict(False, "numbers", entities)
    if a.negations != b.negations:
        return GuardVerdict(False, "negation", entities)
    return GuardVerdict(True, "", entities)


# ---- the lexical guard (my addition to the design's rule; see the report) --------------------

LEXICAL_MAX_DIFF = 2
"""Two questions whose content words differ in at most this many words (one word replaced = 2,
one word added or removed = 1) are different questions: a single-word swap (`push` and `pull`,
a class name, `2` and `3`) moves a question's vector very little but changes what it asks."""
_CONTENT_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_STOPWORDS = frozenset(
    "the a an is are was were be been do does did how what which who whom whose when where why "
    "to of in on for with by and or can could should would will you your i me my we our us it "
    "its this that these those as at from into than then there if so also please tell give show "
    "explain way".split()
)
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def content_words(question: str) -> frozenset[str] | None:
    """The lower-cased words of an English question without the stop words; None for a question
    with Han characters (no spaces between its words: not measured here)."""
    text = normalise(question)
    if _HAN.search(text):
        return None
    return frozenset(w for w in _CONTENT_WORD.findall(text.lower()) if w not in _STOPWORDS)


def lexical_guard(new_question: str, cached_question: str) -> GuardVerdict:
    """Reject a stored question whose content words differ from the new question's by one or two
    words (`LEXICAL_MAX_DIFF`); the same words in another order or with other stop words pass."""
    a, b = content_words(new_question), content_words(cached_question)
    if a is None or b is None:
        return GuardVerdict(True)
    if 0 < len(a ^ b) <= LEXICAL_MAX_DIFF:
        return GuardVerdict(False, "lexical")
    return GuardVerdict(True)


def question_guard(
    new_question: str, cached_question: str, *, lexical: bool = True
) -> GuardVerdict:
    """The entity guard of the design, then (when `lexical`) the lexical guard."""
    verdict = entity_guard(new_question, cached_question)
    if verdict.ok and lexical:
        second = lexical_guard(new_question, cached_question)
        if not second.ok:
            return GuardVerdict(False, second.reason, verdict.entities)
    return verdict


# ---- stores ----------------------------------------------------------------------------------


def question_hash(question: str) -> str:
    """16 hex digits of the sha256 of the canonical question: names a question in records."""
    return hashlib.sha256(canonical_question(question).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Partition:
    """What must match exactly for a stored question to be a candidate at all."""

    kb: str
    filters_hash: str
    acl_hash: str
    config_version: str
    data_generation: int
    top_k: int
    language: str

    def key(self) -> tuple[Any, ...]:
        return tuple(asdict(self).values())


@dataclass(frozen=True)
class StoredQuery:
    entry_id: str
    question: str
    top_chunk_ids: tuple[str, ...]
    top_doc_ids: tuple[str, ...]
    created_at: float


@dataclass(frozen=True)
class Candidate:
    entry: StoredQuery
    distance: float


class SemanticStore(Protocol):
    def search(
        self, partition: Partition, vector: Sequence[float], *, k: int, now: float
    ) -> list[Candidate]:
        """The `k` nearest stored questions of the partition, nearest first."""
        ...

    def add(
        self, partition: Partition, vector: Sequence[float], entry: StoredQuery, *, ttl_s: int
    ) -> None: ...

    def close(self) -> None: ...


def _unit(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class _Block:
    """The unit vectors of one partition in one growing matrix (capacity doubles), so a lookup
    is one matrix-vector product without copying the rows."""

    def __init__(self, dim: int) -> None:
        self.matrix = np.zeros((64, dim), dtype=np.float32)
        self.entries: list[tuple[StoredQuery, float]] = []

    def append(self, vector: np.ndarray, entry: StoredQuery, expires: float) -> None:
        n = len(self.entries)
        if n == self.matrix.shape[0]:
            grown = np.zeros((n * 2, self.matrix.shape[1]), dtype=np.float32)
            grown[:n] = self.matrix
            self.matrix = grown
        self.matrix[n] = vector
        self.entries.append((entry, expires))

    def drop_oldest(self, count: int) -> None:
        n = len(self.entries)
        self.matrix[: n - count] = self.matrix[count:n].copy()
        del self.entries[:count]


class MemorySemanticStore:
    """Stored questions in process memory: one matrix of unit vectors per partition and a dot
    product per lookup (exact). `max_entries` bounds one partition: the oldest are dropped."""

    def __init__(self, *, max_entries: int = 20_000) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._blocks: dict[tuple[Any, ...], _Block] = {}

    def search(
        self, partition: Partition, vector: Sequence[float], *, k: int, now: float
    ) -> list[Candidate]:
        query = _unit(vector)
        with self._lock:
            block = self._blocks.get(partition.key())
            if block is None or not block.entries:
                return []
            n = len(block.entries)
            similarity = block.matrix[:n] @ query
            entries = list(block.entries)
        out: list[Candidate] = []
        for i in np.argsort(-similarity):
            entry, expires = entries[int(i)]
            if expires <= now:
                continue
            out.append(Candidate(entry, float(1.0 - similarity[int(i)])))
            if len(out) >= k:
                break
        return out

    def add(
        self, partition: Partition, vector: Sequence[float], entry: StoredQuery, *, ttl_s: int
    ) -> None:
        unit = _unit(vector)
        with self._lock:
            block = self._blocks.setdefault(partition.key(), _Block(len(unit)))
            block.append(unit, entry, entry.created_at + ttl_s)
            if len(block.entries) > self.max_entries:
                block.drop_oldest(len(block.entries) - self.max_entries)

    def size(self) -> int:
        with self._lock:
            return sum(len(b.entries) for b in self._blocks.values())

    def close(self) -> None:
        pass


class EsSemanticStore:
    """Stored questions in Elasticsearch, index `<index_prefix>cache_semantic_<kb>` (design
    s12-1): the partition is a `filter` of the kNN search, expiry a range on `expires_at`.
    `flat` vector index: exact distances (the index holds some thousand questions). A new entry
    is searchable after the index refresh (one second; `refresh=True` waits for it, for tests)."""

    def __init__(
        self,
        es: Elasticsearch,
        *,
        dim: int,
        index_prefix: str = "",
        refresh: bool = False,
        request_timeout_s: float = 0.5,
    ) -> None:
        self.es = es.options(request_timeout=request_timeout_s)
        self._admin = es
        self.dim = dim
        self.index_prefix = index_prefix
        self.refresh = refresh
        self._ready: set[str] = set()

    def index_name(self, kb: str) -> str:
        return f"{self.index_prefix}cache_semantic_{kb}"

    def mapping(self) -> dict[str, Any]:
        keyword = {"type": "keyword"}
        return {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "kb": keyword,
                    "filters_hash": keyword,
                    "acl_hash": keyword,
                    "config_version": keyword,
                    "language": keyword,
                    "data_generation": {"type": "long"},
                    "top_k": {"type": "integer"},
                    "question": {"type": "keyword", "index": False, "ignore_above": 1024},
                    "top_chunk_ids": {"type": "keyword", "index": False},
                    "top_doc_ids": {"type": "keyword", "index": False},
                    "created_at": {"type": "double"},
                    "expires_at": {"type": "double"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.dim,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {"type": "flat"},
                    },
                },
            },
        }

    def _ensure(self, kb: str) -> str:
        index = self.index_name(kb)
        if index in self._ready:
            return index
        if not self._admin.indices.exists(index=index):
            from elasticsearch import BadRequestError

            try:
                self._admin.indices.create(index=index, **self.mapping())
            except BadRequestError as exc:  # another process made it first
                if "resource_already_exists" not in str(exc):
                    raise
        self._ready.add(index)
        return index

    @staticmethod
    def _filters(partition: Partition, now: float) -> list[dict[str, Any]]:
        return [
            {"term": {"filters_hash": partition.filters_hash}},
            {"term": {"acl_hash": partition.acl_hash}},
            {"term": {"config_version": partition.config_version}},
            {"term": {"language": partition.language}},
            {"term": {"data_generation": partition.data_generation}},
            {"term": {"top_k": partition.top_k}},
            {"range": {"expires_at": {"gt": now}}},
        ]

    def search(
        self, partition: Partition, vector: Sequence[float], *, k: int, now: float
    ) -> list[Candidate]:
        from elasticsearch import NotFoundError

        index = self.index_name(partition.kb)
        try:
            response = self.es.search(
                index=index,
                knn={
                    "field": "vector",
                    "query_vector": [float(x) for x in vector],
                    "k": k,
                    "num_candidates": max(50, k),
                    "filter": self._filters(partition, now),
                },
                size=k,
                source=["question", "top_chunk_ids", "top_doc_ids", "created_at"],
            )
        except NotFoundError:  # nothing stored yet
            return []
        out: list[Candidate] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            entry = StoredQuery(
                entry_id=str(hit["_id"]),
                question=source["question"],
                top_chunk_ids=tuple(source["top_chunk_ids"]),
                top_doc_ids=tuple(source["top_doc_ids"]),
                created_at=float(source["created_at"]),
            )
            # ES scores a cosine kNN hit (1 + cosine) / 2.
            out.append(Candidate(entry, float(1.0 - (2.0 * float(hit["_score"]) - 1.0))))
        return out

    def add(
        self, partition: Partition, vector: Sequence[float], entry: StoredQuery, *, ttl_s: int
    ) -> None:
        index = self._ensure(partition.kb)
        self.es.index(
            index=index,
            id=entry.entry_id,
            document={
                "kb": partition.kb,
                "filters_hash": partition.filters_hash,
                "acl_hash": partition.acl_hash,
                "config_version": partition.config_version,
                "language": partition.language,
                "data_generation": partition.data_generation,
                "top_k": partition.top_k,
                "question": entry.question,
                "top_chunk_ids": list(entry.top_chunk_ids),
                "top_doc_ids": list(entry.top_doc_ids),
                "created_at": entry.created_at,
                "expires_at": entry.created_at + ttl_s,
                "vector": [float(x) for x in _unit(vector)],
            },
            refresh="true" if self.refresh else "false",
        )

    def drop(self, kb: str) -> None:
        """Delete the index of `kb` (admin; entries of an old data generation are never
        matched but stay until this or the expiry cleanup)."""
        index = self.index_name(kb)
        self._admin.indices.delete(index=index, ignore_unavailable=True)
        self._ready.discard(index)

    def close(self) -> None:
        self._admin.close()


# ---- the recorder ----------------------------------------------------------------------------


@dataclass
class ShadowRecord:
    """One line of the shadow file."""

    time: float
    kb: str
    question_hash: str
    question: str
    status: str
    """no_vector | empty | too_far | guard_rejected | would_hit"""
    threshold: float
    strict_threshold: float
    generation: int
    config_version: str
    partition_size_hint: int | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    candidate_question: str | None = None
    distance: float | None = None
    guard_ok: bool | None = None
    guard_reason: str | None = None
    entities: bool | None = None
    nearest_distance: float | None = None
    nearest_guard_ok: bool | None = None
    nearest_guard_reason: str | None = None
    cached_top: list[str] = field(default_factory=list)
    fresh_top: list[str] = field(default_factory=list)
    cached_top_docs: list[str] = field(default_factory=list)
    fresh_top_docs: list[str] = field(default_factory=list)
    same_top1: bool | None = None
    same_top_docs: bool | None = None
    overlap: float | None = None
    lookup_ms: float = 0.0


def _ids(result: RetrievalResult) -> tuple[list[str], list[str]]:
    chunks = [c.chunk_id for c in result.chunks[:TOP_IDS]]
    docs: list[str] = []
    for c in result.chunks:
        if c.doc_id not in docs:
            docs.append(c.doc_id)
    return chunks, docs[:TOP_IDS]


class ShadowRecorder:
    """See the module doc. `observe` returns the record it wrote (also for tests)."""

    def __init__(
        self,
        store: SemanticStore,
        writer: RotatingJsonlWriter | None,
        *,
        threshold: float,
        ttl_s: int,
        strict_factor: float = STRICT_FACTOR,
        clock: Callable[[], float] = time.time,
        k: int = 3,
        lexical: bool = True,
    ) -> None:
        self.lexical = lexical
        self.store = store
        self.writer = writer
        self.threshold = threshold
        self.strict_factor = strict_factor
        self.ttl_s = ttl_s
        self._clock = clock
        self.k = k

    def observe(
        self,
        *,
        kb: str,
        question: str,
        filters: Mapping[str, Any] | None,
        acl_groups: Sequence[str],
        top_k: int,
        config_version: str,
        data_generation: int,
        fresh: RetrievalResult,
        vector: Sequence[float] | None,
    ) -> ShadowRecord:
        began = time.perf_counter()
        now = self._clock()
        fresh_ids, fresh_docs = _ids(fresh)
        record = ShadowRecord(
            time=now,
            kb=kb,
            question_hash=question_hash(question),
            question=question[:QUESTION_CHARS],
            status="no_vector",
            threshold=self.threshold,
            strict_threshold=self.threshold * self.strict_factor,
            generation=data_generation,
            config_version=config_version,
            fresh_top=fresh_ids,
            fresh_top_docs=fresh_docs,
        )
        if vector is None:
            return self._finish(record, began)
        partition = Partition(
            kb=kb,
            filters_hash=filters_hash(filters),
            acl_hash=acl_hash(acl_groups),
            config_version=config_version,
            data_generation=data_generation,
            top_k=top_k,
            language=answer_language(question, "auto"),
        )
        candidates = self.store.search(partition, vector, k=self.k, now=now)
        record.status = "empty"
        if candidates:
            self._judge(record, candidates, question, fresh_ids, fresh_docs)
        if record.status != "would_hit":
            self.store.add(
                partition,
                vector,
                StoredQuery(
                    entry_id=uuid.uuid4().hex,
                    question=question[:QUESTION_CHARS],
                    top_chunk_ids=tuple(fresh_ids),
                    top_doc_ids=tuple(fresh_docs),
                    created_at=now,
                ),
                ttl_s=self.ttl_s,
            )
        return self._finish(record, began)

    def _judge(
        self,
        record: ShadowRecord,
        candidates: list[Candidate],
        question: str,
        fresh_ids: list[str],
        fresh_docs: list[str],
    ) -> None:
        nearest = candidates[0]
        first_guard = question_guard(question, nearest.entry.question, lexical=self.lexical)
        record.nearest_distance = round(nearest.distance, 5)
        record.nearest_guard_ok = first_guard.ok
        record.nearest_guard_reason = first_guard.reason or None
        record.status = "too_far"
        for candidate in candidates:
            guard = question_guard(question, candidate.entry.question, lexical=self.lexical)
            limit = self.threshold if guard.entities else self.threshold * self.strict_factor
            if candidate.distance > limit:
                continue
            if not guard.ok:
                record.status = "guard_rejected"
                record.guard_ok, record.guard_reason = False, guard.reason
                record.distance = round(candidate.distance, 5)
                record.candidate_id = candidate.entry.entry_id
                record.candidate_hash = question_hash(candidate.entry.question)
                continue
            record.status = "would_hit"
            record.candidate_id = candidate.entry.entry_id
            record.candidate_hash = question_hash(candidate.entry.question)
            record.candidate_question = candidate.entry.question
            record.distance = round(candidate.distance, 5)
            record.guard_ok, record.guard_reason, record.entities = True, None, guard.entities
            cached = list(candidate.entry.top_chunk_ids)
            record.cached_top = cached
            record.cached_top_docs = list(candidate.entry.top_doc_ids)
            record.same_top1 = bool(cached and fresh_ids and cached[0] == fresh_ids[0])
            record.same_top_docs = record.cached_top_docs == fresh_docs
            union = set(cached) | set(fresh_ids)
            record.overlap = (
                round(len(set(cached) & set(fresh_ids)) / len(union), 4) if union else 1.0
            )
            return

    def _finish(self, record: ShadowRecord, began: float) -> ShadowRecord:
        record.lookup_ms = round((time.perf_counter() - began) * 1000.0, 3)
        if self.writer is not None:
            self.writer.write(asdict(record))
        get_observability().metrics.record_cache_shadow(
            kb=record.kb, tool="search_docs", verdict=record.status
        )
        try:
            from opentelemetry import trace

            trace.get_current_span().set_attribute(attr.CACHE_SHADOW, record.status)
        except Exception:  # a span attribute is never worth a failure
            logger.debug("could not set the shadow span attribute", exc_info=True)
        return record


def make_shadow_recorder(settings: ServiceSettings) -> ShadowRecorder:
    """The recorder the settings describe: store `cache_semantic_store`, records in
    `<cache_shadow_dir>/shadow.jsonl`."""
    from rag.config import repo_path

    store: SemanticStore
    if settings.cache_semantic_store == "es":
        from rag.factories import make_es_client

        store = EsSemanticStore(
            make_es_client(settings.es_url, request_timeout=0.5, max_retries=0),
            dim=settings.embed_dim,
            index_prefix=settings.search_index_prefix,
        )
    else:
        store = MemorySemanticStore()
    writer = (
        RotatingJsonlWriter(
            repo_path(settings.cache_shadow_dir) / "shadow.jsonl", max_bytes=20_000_000, backups=3
        )
        if settings.cache_shadow_dir
        else None
    )
    return ShadowRecorder(
        store,
        writer,
        threshold=settings.cache_semantic_threshold,
        ttl_s=settings.cache_semantic_ttl_s,
    )
