"""Spec P5-01 AC11: the semantic cache's Elasticsearch store against the compose ES.

The index is `<test prefix>cache_semantic_<kb>` with a knowledge base name made for each test:
under the prefix unique to this run, deleted with the test; the real indices are never touched.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from elasticsearch import Elasticsearch

from rag.cache.semantic import (
    EsSemanticStore,
    MemorySemanticStore,
    Partition,
    StoredQuery,
    question_hash,
)

pytestmark = pytest.mark.integration

DIM = 16
FOREVER = 10_000_000_000


def _vec(angle: float) -> list[float]:
    v = np.zeros(DIM)
    v[0], v[1] = np.cos(angle), np.sin(angle)
    return list(v)


def _entry(question: str, at: float) -> StoredQuery:
    return StoredQuery(question_hash(question), question, ("c1", "c2"), ("d1", "d2"), at)


@pytest.fixture
def part() -> Partition:
    return Partition(f"k{uuid.uuid4().hex[:8]}", "f1", "a1", "cfg", 7, 8, "en")


@pytest.fixture
def store(es: Elasticsearch, index_prefix: str, part: Partition) -> Iterator[EsSemanticStore]:
    made = EsSemanticStore(
        es, dim=DIM, index_prefix=index_prefix, refresh=True, request_timeout_s=10.0
    )
    yield made
    made.drop(part.kb)


def test_the_index_is_created_with_the_strict_mapping(
    store: EsSemanticStore, es: Elasticsearch, part: Partition
) -> None:
    store.add(part, _vec(0.0), _entry("q", 1000.0), ttl_s=FOREVER)
    name = store.index_name(part.kb)
    mapping = es.indices.get_mapping(index=name)[name]["mappings"]
    assert mapping["dynamic"] == "strict"
    assert mapping["properties"]["vector"]["type"] == "dense_vector"
    assert mapping["properties"]["vector"]["similarity"] == "cosine"
    store.drop(part.kb)
    assert not es.indices.exists(index=name)


def test_nearest_first_with_the_same_distances_as_the_memory_store(
    store: EsSemanticStore, part: Partition
) -> None:
    memory = MemorySemanticStore()
    base = _vec(0.0)
    for angle, question in ((0.5, "far"), (0.1, "close"), (0.9, "farthest")):
        entry = _entry(question, 1000.0)
        store.add(part, _vec(angle), entry, ttl_s=FOREVER)
        memory.add(part, _vec(angle), entry, ttl_s=FOREVER)
    found = store.search(part, base, k=3, now=1001.0)
    reference = memory.search(part, base, k=3, now=1001.0)
    assert [c.entry.question for c in found] == ["close", "far", "farthest"]
    for got, want in zip(found, reference, strict=True):
        assert got.entry.question == want.entry.question
        assert got.distance == pytest.approx(want.distance, abs=1e-4)
        assert got.entry.top_chunk_ids == ("c1", "c2")
        assert got.entry.top_doc_ids == ("d1", "d2")


@pytest.mark.parametrize(
    "change",
    [
        {"filters_hash": "f2"},
        {"acl_hash": "a2"},
        {"config_version": "cfg2"},
        {"data_generation": 8},
        {"top_k": 9},
        {"language": "zh"},
    ],
    ids=["filters", "acl", "config", "generation", "top_k", "language"],
)
def test_other_partitions_never_match(
    store: EsSemanticStore, part: Partition, change: dict[str, object]
) -> None:
    store.add(part, _vec(0.0), _entry("q", 1000.0), ttl_s=FOREVER)
    assert store.search(replace(part, **change), _vec(0.0), k=3, now=1001.0) == []  # type: ignore[arg-type]
    assert len(store.search(part, _vec(0.0), k=3, now=1001.0)) == 1


def test_an_expired_entry_is_not_a_candidate(store: EsSemanticStore, part: Partition) -> None:
    store.add(part, _vec(0.0), _entry("short", 1000.0), ttl_s=10)
    assert store.search(part, _vec(0.0), k=1, now=1005.0)
    assert store.search(part, _vec(0.0), k=1, now=1011.0) == []


def test_searching_a_knowledge_base_that_has_no_index_yet_finds_nothing(
    store: EsSemanticStore, part: Partition
) -> None:
    assert store.search(part, _vec(0.0), k=3, now=1000.0) == []


def test_a_second_writer_does_not_fail_when_the_index_already_exists(
    es: Elasticsearch, index_prefix: str, store: EsSemanticStore, part: Partition
) -> None:
    other = EsSemanticStore(
        es, dim=DIM, index_prefix=index_prefix, refresh=True, request_timeout_s=10.0
    )
    store.add(part, _vec(0.0), _entry("a", 1000.0), ttl_s=FOREVER)
    other.add(part, _vec(0.1), _entry("b", 1000.0), ttl_s=FOREVER)  # makes no new index
    assert len(store.search(part, _vec(0.0), k=5, now=1001.0)) == 2
