"""Spec P8-01 AC7: export an index, restore it into a new one, verify, rebind a docstore.

The source is `fake_kb_hf_docs_v1` (438 chunks with fake vectors), read only. The restored index
is `test<run>_restore_v1`, deleted at the end.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch

from rag.factories import JsonFileDocumentStore
from rag.ingest.job import INDEX_UUID_KEY
from rag.ops.backup import export_index, rebind_docstore, restore_index, verify_indexes

pytestmark = pytest.mark.integration

SOURCE = "fake_kb_hf_docs_v1"


@pytest.fixture
def target(es: Elasticsearch, index_prefix: str) -> Iterator[str]:
    if not es.indices.exists(index=SOURCE):
        pytest.skip(f"{SOURCE} is missing (run the ingest job with EMBED_BACKEND=fake)")
    name = f"{index_prefix}restore_v1"
    yield name
    es.indices.delete(index=name, ignore_unavailable=True)


def test_round_trip_keeps_mapping_ids_sources_and_rankings(
    es: Elasticsearch, target: str, tmp_path: Path
) -> None:
    exported = export_index(es, SOURCE, tmp_path)
    source_count = int(es.count(index=SOURCE)["count"])
    assert exported.docs == source_count > 0
    meta = json.loads((tmp_path / SOURCE / "meta.json").read_text("utf-8"))
    assert meta["mappings"]["dynamic"] == "strict" and "embedding" in meta["mappings"]["properties"]

    restored = restore_index(es, tmp_path / SOURCE, target)
    assert restored.docs == source_count and restored.uuid != exported.uuid
    assert es.indices.get_mapping(index=target)[target]["mappings"] == meta["mappings"]

    verified = verify_indexes(es, SOURCE, target, sample=50)
    assert verified.ok, verified
    assert verified.sources_equal == verified.sampled == 50  # vectors included
    knn = [s for s in verified.searches if s["kind"] == "knn"]
    assert knn and all(s["self_first"] for s in knn)

    with pytest.raises(FileExistsError):
        restore_index(es, tmp_path / SOURCE, target)  # never over an index that exists


def test_rebind_docstore_to_the_restored_index(
    es: Elasticsearch, target: str, tmp_path: Path
) -> None:
    export_index(es, SOURCE, tmp_path)
    restore_index(es, tmp_path / SOURCE, target)
    old = tmp_path / f"{SOURCE}.json"
    store = JsonFileDocumentStore(old)
    store.set_meta(INDEX_UUID_KEY, "uuid-of-the-lost-index")
    store.persist()
    store.set_meta(
        INDEX_UUID_KEY, es.indices.get_settings(index=SOURCE)[SOURCE]["settings"]["index"]["uuid"]
    )
    store.persist()
    out = rebind_docstore(es, old, target, tmp_path / f"{target}.json", backup=tmp_path / SOURCE)
    uuid = es.indices.get_settings(index=target)[target]["settings"]["index"]["uuid"]
    assert out["now"] == uuid
    assert JsonFileDocumentStore(tmp_path / f"{target}.json").get_meta(INDEX_UUID_KEY) == uuid
