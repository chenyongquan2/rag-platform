"""Spec P8-01 AC7 (unit part): export format, file copies and docstore rebinding.

The round trip through a real Elasticsearch is tests/integration/test_ops_backup_es.py.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from rag.factories import JsonFileDocumentStore
from rag.ingest.job import INDEX_UUID_KEY
from rag.ops.backup import _kept_settings, copy_files, export_index, rebind_docstore


class FakeIndices:
    def __init__(self, uuid: str) -> None:
        self.uuid = uuid

    def get_settings(self, index: str) -> dict[str, Any]:
        return {
            index: {
                "settings": {
                    "index": {
                        "uuid": self.uuid,
                        "number_of_shards": "1",
                        "number_of_replicas": "0",
                        "creation_date": "1",
                        "provided_name": index,
                        "mapping": {"exclude_source_vectors": "true"},
                    }
                }
            }
        }

    def get_mapping(self, index: str) -> dict[str, Any]:
        return {index: {"mappings": {"dynamic": "strict", "properties": {"x": {"type": "long"}}}}}

    def get_alias(self, index: str) -> dict[str, Any]:
        return {index: {"aliases": {"kb_x": {}}}}


class FakeEs:
    """PIT paging over `n` documents, two per page."""

    def __init__(self, n: int) -> None:
        self.docs = [{"_id": f"d{i}", "_source": {"x": i, "embedding": [0.5, i]}} for i in range(n)]
        self.indices = FakeIndices("uuid-1")
        self.searches: list[dict[str, Any]] = []
        self.closed: list[str] = []

    def open_point_in_time(self, index: str, keep_alive: str) -> dict[str, str]:
        return {"id": "pit-1"}

    def close_point_in_time(self, id: str) -> None:
        self.closed.append(id)

    def info(self) -> dict[str, Any]:
        return {"version": {"number": "9.5.5"}}

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        start = kwargs.get("search_after", [-1])[0] + 1
        page = self.docs[start : start + 2]
        hits = [{**d, "sort": [start + i]} for i, d in enumerate(page)]
        return {"pit_id": "pit-1", "hits": {"hits": hits}}


def test_export_writes_meta_and_every_document_with_its_vector(tmp_path: Path) -> None:
    es = FakeEs(5)
    result = export_index(es, "kb_x_v1", tmp_path)
    assert (result.docs, result.uuid) == (5, "uuid-1")
    meta = json.loads((tmp_path / "kb_x_v1" / "meta.json").read_text("utf-8"))
    assert meta["aliases"] == ["kb_x"] and meta["count"] == 5
    assert meta["settings"] == {
        "number_of_shards": "1",
        "number_of_replicas": "0",
        "mapping": {"exclude_source_vectors": "true"},
    }
    with gzip.open(tmp_path / "kb_x_v1" / meta["files"][0], "rt", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    assert docs == es.docs
    # Vectors are asked for explicitly (the index leaves them out of _source), the PIT is closed.
    assert all(s["source_exclude_vectors"] is False for s in es.searches)
    assert es.closed == ["pit-1"]


def test_export_refuses_a_folder_that_is_not_empty(tmp_path: Path) -> None:
    (tmp_path / "kb_x_v1").mkdir()
    (tmp_path / "kb_x_v1" / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_index(FakeEs(1), "kb_x_v1", tmp_path)


def test_only_restorable_settings_are_kept() -> None:
    kept = _kept_settings({"uuid": "u", "version": {}, "number_of_shards": "1", "analysis": {}})
    assert kept == {"number_of_shards": "1", "analysis": {}}


def test_copy_files_copies_sqlite_consistently_and_writes_a_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    with sqlite3.connect(src / "cache.sqlite3") as db:
        db.execute("create table t (k text)")
        db.execute("insert into t values ('v')")
    (src / "kb_x_v1.json").write_text('{"a": 1}', encoding="utf-8")
    (src / "kb_x_v1.json.tmp").write_text("partial", encoding="utf-8")
    (src / "sub").mkdir()
    (src / "sub" / "kb_x_v1.json").write_text('{"b": 2}', encoding="utf-8")  # same name, deeper
    manifest = copy_files([src], tmp_path / "out")
    copies = sorted(Path(m["copy"]).relative_to(tmp_path / "out").as_posix() for m in manifest)
    assert copies == ["src/cache.sqlite3", "src/kb_x_v1.json", "src/sub/kb_x_v1.json"]
    assert (tmp_path / "out" / "src" / "sub" / "kb_x_v1.json").read_text("utf-8") == '{"b": 2}'
    with sqlite3.connect(tmp_path / "out" / "src" / "cache.sqlite3") as db:
        assert db.execute("select k from t").fetchall() == [("v",)]
    assert all(len(m["sha256"]) == 64 for m in manifest)


def _backup_dir(tmp_path: Path, uuid: str) -> Path:
    folder = tmp_path / "backup" / "kb_x_v1"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"uuid": uuid}), encoding="utf-8")
    return folder


def test_rebind_writes_a_copy_bound_to_the_new_index(tmp_path: Path) -> None:
    # Review P8-01 round 1, issue 3: the backup folder is given and its uuid is checked.
    old = tmp_path / "kb_x_v1.json"
    store = JsonFileDocumentStore(old)
    store.set_meta(INDEX_UUID_KEY, "old-uuid")
    store.persist()
    es = FakeEs(0)
    es.indices.uuid = "new-uuid"
    backup = _backup_dir(tmp_path, "old-uuid")
    out = rebind_docstore(es, old, "kb_x_v2", tmp_path / "kb_x_v2.json", backup=backup)
    assert (out["was"], out["now"]) == ("old-uuid", "new-uuid")
    assert JsonFileDocumentStore(tmp_path / "kb_x_v2.json").get_meta(INDEX_UUID_KEY) == "new-uuid"
    assert JsonFileDocumentStore(old).get_meta(INDEX_UUID_KEY) == "old-uuid"  # untouched
    with pytest.raises(ValueError):
        rebind_docstore(es, old, "kb_x_v2", old, backup=backup)


def test_rebind_refuses_a_docstore_of_another_index_than_the_backup(tmp_path: Path) -> None:
    old = tmp_path / "kb_x_v1.json"
    store = JsonFileDocumentStore(old)
    store.set_meta(INDEX_UUID_KEY, "uuid-of-some-other-index")
    store.persist()
    with pytest.raises(ValueError, match="belongs to index"):
        rebind_docstore(
            FakeEs(0), old, "kb_x_v2", tmp_path / "out.json", backup=_backup_dir(tmp_path, "u1")
        )
    assert not (tmp_path / "out.json").exists()


def test_verify_is_not_ok_when_a_knn_ranking_differs() -> None:
    from rag.ops.backup import VerifyResult

    base = {
        "source": "a",
        "target": "b",
        "source_count": 1,
        "target_count": 1,
        "ids_equal": True,
        "missing": 0,
        "extra": 0,
        "sampled": 1,
        "sources_equal": 1,
    }
    good = [{"kind": "bm25", "same_top10": True}, {"kind": "knn", "same_top10": True}]
    bad = [{"kind": "bm25", "same_top10": True}, {"kind": "knn", "same_top10": False}]
    assert VerifyResult(**base, searches=good).ok
    assert not VerifyResult(**base, searches=bad).ok
