from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from elasticsearch import BadRequestError, Elasticsearch
from redis import Redis

from rag.config import EsConfig, KnowledgeBaseConfig, ServiceSettings
from rag.index import cli
from rag.index.generation import get_data_generation
from rag.index.manage import (
    AliasConflictError,
    AliasedIndexDeletionError,
    IndexExistsError,
    aliased_indices,
    create_kb_index,
    delete_kb_index,
    list_versions,
    switch_alias,
)
from rag.index.mapping import KEYWORD_FIELDS, build_index_body, index_body_for_kb
from rag.index.names import KbIndexNames

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
NamesFactory = Callable[[], KbIndexNames]


def _small_body(analyzer: str = "standard") -> dict:
    return build_index_body(analyzer=analyzer, dims=4)


def test_es_matches_compose_pin(es: Elasticsearch) -> None:
    compose = (REPO_ROOT / "docker-compose.yml").read_text("utf-8")
    pinned = re.search(r"elasticsearch/elasticsearch:(\d+\.\d+\.\d+)", compose)
    assert pinned, "docker-compose.yml must pin an exact Elasticsearch version"
    assert es.info()["version"]["number"] == pinned.group(1)
    assert es.license.get()["license"]["type"] == "basic"
    nodes = es.nodes.stats(metric="jvm")["nodes"]
    heaps = [node["jvm"]["mem"]["heap_max_in_bytes"] for node in nodes.values()]
    assert heaps == [2 * 1024**3]


def test_create_kb_index_applies_mapping(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    cfg = KnowledgeBaseConfig(kb=names.kb, es=EsConfig(analyzer="standard"))
    body = index_body_for_kb(cfg, dims=1024)

    index = create_kb_index(es, names, 1, body)

    assert index == names.index(1)
    mapping = es.indices.get_mapping(index=index)[index]["mappings"]
    assert mapping["dynamic"] == "strict"
    props = mapping["properties"]
    assert props["content"] == {"type": "text", "analyzer": "standard"}
    assert props["title_path"] == {"type": "text", "analyzer": "standard"}
    embedding = props["embedding"]
    assert embedding["dims"] == 1024
    assert (embedding["similarity"], embedding["index"]) == ("cosine", True)
    assert embedding["index_options"] == {
        "type": "int8_hnsw",
        "m": 16,
        "ef_construction": 100,
        "rescore_vector": {"oversample": 3.0},
    }
    for field in KEYWORD_FIELDS:
        assert props[field] == {"type": "keyword"}
    settings = es.indices.get_settings(index=index)[index]["settings"]["index"]
    assert (settings["number_of_shards"], settings["number_of_replicas"]) == ("1", "0")
    assert settings["refresh_interval"] == "1s"
    assert settings["mapping"]["exclude_source_vectors"] == "true"

    with pytest.raises(IndexExistsError, match=index):
        create_kb_index(es, names, 1, body)
    with pytest.raises(BadRequestError, match="strict"):
        es.index(index=index, document={"content": "x", "not_in_mapping": 1})

    vector = [1.0] + [0.0] * 1023
    es.index(
        index=index,
        id="c1",
        document={"kb": names.kb, "chunk_id": "c1", "content": "hello", "embedding": vector},
        refresh=True,
    )
    assert "embedding" not in es.get(index=index, id="c1")["_source"]
    hits = es.search(
        index=index,
        knn={"field": "embedding", "query_vector": vector, "k": 1, "num_candidates": 10},
    )["hits"]["hits"]
    assert [h["_id"] for h in hits] == ["c1"]


def test_analyzer_applied_to_text_fields(es: Elasticsearch, make_names: NamesFactory) -> None:
    cjk_index = create_kb_index(es, make_names(), 1, _small_body("cjk"))
    std_index = create_kb_index(es, make_names(), 1, _small_body("standard"))
    text = "中文检索 is not supported"

    def tokens(index: str, field: str) -> list[str]:
        resp = es.indices.analyze(index=index, field=field, text=text)
        return [t["token"] for t in resp["tokens"]]

    # cjk: Han characters become overlapping bigrams; English stop words (is, not) are dropped.
    assert tokens(cjk_index, "content") == ["中文", "文检", "检索", "supported"]
    assert tokens(cjk_index, "title_path") == ["中文", "文检", "检索", "supported"]
    assert tokens(std_index, "content") == ["中", "文", "检", "索", "is", "not", "supported"]
    with pytest.raises(BadRequestError, match="no_such_analyzer"):
        create_kb_index(es, make_names(), 1, _small_body("no_such_analyzer"))


def test_alias_switch_is_atomic_for_readers(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    v1 = create_kb_index(es, names, 1, _small_body())
    v2 = create_kb_index(es, names, 2, _small_body())
    for index in (v1, v2):
        es.index(index=index, id="same", document={"kb": names.kb, "content": "x"}, refresh=True)
    assert switch_alias(es, names, v1) is None  # first switch: there was no previous index

    stop = threading.Event()
    seen: list[tuple[str, ...]] = []
    errors: list[Exception] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                resp = es.search(index=names.alias, query={"match_all": {}}, size=10)
            except Exception as exc:  # any failure during a switch is what this test looks for
                errors.append(exc)
                continue
            seen.append(tuple(sorted(hit["_index"] for hit in resp["hits"]["hits"])))

    readers = [threading.Thread(target=reader) for _ in range(2)]
    for t in readers:
        t.start()
    previous = []
    try:
        for i in range(20):
            previous.append(switch_alias(es, names, v2 if i % 2 == 0 else v1))
    finally:
        stop.set()
        for t in readers:
            t.join()

    assert errors == []
    assert previous == [v1, v2] * 10
    # Every search saw exactly one index: never none (alias missing), never both.
    assert set(seen) == {(v1,), (v2,)}
    assert aliased_indices(es, names) == [v1]


def test_alias_switch_conflict_changes_nothing(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    v1, v2, v3 = (create_kb_index(es, names, n, _small_body()) for n in (1, 2, 3))
    switch_alias(es, names, v1)
    switch_alias(es, names, v2)

    # The caller validated v1, but someone switched to v2 meanwhile: nothing changes.
    with pytest.raises(AliasConflictError, match=v1):
        switch_alias(es, names, v3, expected_current=v1)
    assert aliased_indices(es, names) == [v2]
    with pytest.raises(AliasConflictError):
        switch_alias(es, names, v3, expected_current=None)
    assert aliased_indices(es, names) == [v2]

    assert switch_alias(es, names, v3, expected_current=v2) == v2
    assert aliased_indices(es, names) == [v3]

    # A stale caller expects the alias already on its target: the request must still check
    # where the alias is, or it would add v2 next to v3.
    with pytest.raises(AliasConflictError, match=v2):
        switch_alias(es, names, v2, expected_current=v2)
    assert aliased_indices(es, names) == [v3]
    # The expected index does not exist (deleted meanwhile): also a conflict, nothing changes.
    with pytest.raises(AliasConflictError, match="does not exist"):
        switch_alias(es, names, v1, expected_current=names.index(7))
    assert aliased_indices(es, names) == [v3]
    # Switching to where the alias already is stays a no-op, with or without expected_current.
    assert switch_alias(es, names, v3, expected_current=v3) == v3
    assert switch_alias(es, names, v3) == v3
    assert aliased_indices(es, names) == [v3]

    with pytest.raises(ValueError, match="does not exist"):
        switch_alias(es, names, names.index(9))
    with pytest.raises(ValueError, match="not a versioned index"):
        switch_alias(es, names, make_names().index(1))
    assert aliased_indices(es, names) == [v3]


def test_delete_guard(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    other = make_names()
    v1 = create_kb_index(es, names, 1, _small_body())
    v2 = create_kb_index(es, names, 2, _small_body())
    other_v1 = create_kb_index(es, other, 1, _small_body())
    switch_alias(es, names, v2)

    with pytest.raises(AliasedIndexDeletionError, match=v2):
        delete_kb_index(es, names, v2)
    for bad in (names.alias, names.pattern, other_v1, "_all", "*"):
        with pytest.raises(ValueError, match="not a versioned index"):
            delete_kb_index(es, names, bad)
    assert es.indices.exists(index=v2)
    assert es.indices.exists(index=other_v1)

    delete_kb_index(es, names, v1)

    assert not es.indices.exists(index=v1)
    assert list_versions(es, names) == {2: v2}
    assert aliased_indices(es, names) == [v2]


def test_cli_init_and_status(
    es: Elasticsearch,
    redis_client: Redis,
    service_settings: ServiceSettings,
    index_prefix: str,
    cli_redis_key_prefix: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(REPO_ROOT)  # kb_dir "kb" is relative to the working directory
    names = KbIndexNames("hf_docs", prefix=index_prefix)
    common = ["--kb", "hf_docs", "--index-prefix", index_prefix]

    assert cli.main(["init", *common]) == 0
    out = capsys.readouterr().out
    assert f"created {names.index(1)}" in out
    assert aliased_indices(es, names) == [names.index(1)]
    props = es.indices.get_mapping(index=names.index(1))[names.index(1)]["mappings"]["properties"]
    assert props["content"]["analyzer"] == "standard"  # es.analyzer of kb/public/hf_docs.yaml
    assert props["embedding"]["dims"] == service_settings.embed_dim

    assert cli.main(["init", *common]) == 0  # idempotent
    assert "already initialized" in capsys.readouterr().out
    assert list_versions(es, names) == {1: names.index(1)}

    assert cli.main(["status", *common]) == 0
    out = capsys.readouterr().out
    assert f"{names.alias} -> {names.index(1)}" in out
    assert f"v1 {names.index(1)} docs=0 (aliased)" in out
    assert "data generation" in out

    create_kb_index(es, names, 2, build_index_body(analyzer="standard", dims=4))
    before = get_data_generation(redis_client, "hf_docs", key_prefix=cli_redis_key_prefix)
    assert cli.main(["switch", *common, "--version", "2"]) == 0
    assert f"{names.alias}: {names.index(1)} -> {names.index(2)}" in capsys.readouterr().out
    assert aliased_indices(es, names) == [names.index(2)]
    after = get_data_generation(redis_client, "hf_docs", key_prefix=cli_redis_key_prefix)
    assert after == before + 1

    assert cli.main(["init", "--kb", "no_such_kb", "--index-prefix", index_prefix]) == 2
    assert "no_such_kb" in capsys.readouterr().err
