from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import EsConfig, KnowledgeBaseConfig
from rag.index.mapping import (
    BOOLEAN_FIELDS,
    INTEGER_FIELDS,
    KEYWORD_FIELDS,
    TEXT_FIELDS,
    VectorIndexOptions,
    build_index_body,
    index_body_for_kb,
)

DESIGN_KEYWORD_FIELDS = {
    "kb",
    "doc_id",
    "chunk_id",
    "parent_id",
    "source_rel",
    "unit",
    "source_url",
    "source_version",
    "doc_type",
    "language",
    "acl_groups",
    "content_hash",
    "pipeline_version",
    "ingest_run_id",
}


def test_mapping_matches_design() -> None:
    body = build_index_body(analyzer="standard", dims=1024)

    index_settings = body["settings"]["index"]
    assert index_settings["number_of_shards"] == 1
    assert index_settings["number_of_replicas"] == 0
    assert index_settings["refresh_interval"] == "1s"
    # Vectors are not returned in _source (ES 9.x default; explicit so an upgrade cannot flip it).
    assert index_settings["mapping"]["exclude_source_vectors"] is True

    mappings = body["mappings"]
    assert mappings["dynamic"] == "strict"
    props = mappings["properties"]
    for field in ("content", "title_path"):
        assert props[field] == {"type": "text", "analyzer": "standard"}
    assert props["embedding"] == {
        "type": "dense_vector",
        "element_type": "float",
        "dims": 1024,
        "similarity": "cosine",
        "index": True,
        "index_options": {
            "type": "int8_hnsw",
            "m": 16,
            "ef_construction": 100,
            "rescore_vector": {"oversample": 3.0},
        },
    }
    assert set(KEYWORD_FIELDS) == DESIGN_KEYWORD_FIELDS
    for field in DESIGN_KEYWORD_FIELDS:
        assert props[field] == {"type": "keyword"}
    assert set(INTEGER_FIELDS) == {"chunk_index"}
    assert props["chunk_index"] == {"type": "integer"}
    assert set(BOOLEAN_FIELDS) == {"has_generated_text", "suspicious_instruction"}
    for field in BOOLEAN_FIELDS:
        assert props[field] == {"type": "boolean"}
    assert set(TEXT_FIELDS) == {"content", "title_path"}
    expected = {*TEXT_FIELDS, *KEYWORD_FIELDS, *INTEGER_FIELDS, *BOOLEAN_FIELDS, "embedding"}
    assert set(props) == expected


def test_mapping_uses_kb_analyzer() -> None:
    cfg = KnowledgeBaseConfig(kb="mt5", es=EsConfig(analyzer="cjk"))

    body = index_body_for_kb(cfg, dims=768)

    props = body["mappings"]["properties"]
    assert props["content"]["analyzer"] == "cjk"
    assert props["title_path"]["analyzer"] == "cjk"  # same analyzer as content (design s7-3)
    assert props["embedding"]["dims"] == 768
    # Each call returns a fresh dict; callers may change it without touching the next index.
    props["content"]["analyzer"] = "changed"
    fresh = index_body_for_kb(cfg, dims=768)["mappings"]["properties"]
    assert fresh["content"]["analyzer"] == "cjk"


def test_vector_options_validation() -> None:
    # Defaults are filled per type: HNSW parameters only for *hnsw, rescore only when quantized.
    assert VectorIndexOptions(type="bbq_hnsw").to_es() == {
        "type": "bbq_hnsw",
        "m": 16,
        "ef_construction": 100,
        "rescore_vector": {"oversample": 3.0},
    }
    assert VectorIndexOptions(type="flat").to_es() == {"type": "flat"}
    assert VectorIndexOptions(type="hnsw").to_es() == {
        "type": "hnsw",
        "m": 16,
        "ef_construction": 100,
    }
    assert VectorIndexOptions(oversample=5).to_es()["rescore_vector"] == {"oversample": 5.0}

    with pytest.raises(ValidationError, match="rescore"):
        VectorIndexOptions(type="flat", oversample=3)
    with pytest.raises(ValidationError, match="rescore"):
        VectorIndexOptions(type="hnsw", oversample=2)
    with pytest.raises(ValidationError, match="HNSW"):
        VectorIndexOptions(type="flat", m=16)
    with pytest.raises(ValidationError):
        VectorIndexOptions(type="bbq_disk")  # needs an Enterprise license
    with pytest.raises(ValidationError):
        VectorIndexOptions(oversample=0.5)
    with pytest.raises(ValidationError):
        VectorIndexOptions(m=1)

    with pytest.raises(ValueError, match="dims"):
        build_index_body(analyzer="standard", dims=0)
    with pytest.raises(ValueError, match="analyzer"):
        build_index_body(analyzer="", dims=1024)
    flat = build_index_body(analyzer="standard", dims=8, vector=VectorIndexOptions(type="flat"))
    assert flat["mappings"]["properties"]["embedding"]["index_options"] == {"type": "flat"}
