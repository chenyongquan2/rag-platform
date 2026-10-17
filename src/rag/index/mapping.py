"""Elasticsearch index body of one knowledge base (design s7-3), checked against ES 9.5.5.

Text and vector live in the same document, so BM25 and kNN read the same record and one filter
works for both routes. Every setting that changed between ES versions is written out:
- vector quantization: 9.0 defaults to int8_hnsw, 9.1+ to bbq_hnsw for >= 384 dims, 9.4+ to
  bbq_disk (Enterprise) - so the type, HNSW parameters and rescore oversampling are explicit;
- `index.mapping.exclude_source_vectors`: true by default in 9.5 (vectors are not returned in
  `_source`); written out so an upgrade cannot change what searches return.
`dynamic: strict` rejects fields that are not in the mapping instead of guessing their type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from rag.config import KnowledgeBaseConfig

TEXT_FIELDS = ("content", "title_path")
KEYWORD_FIELDS = (
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
    "images",  # s5-4: image paths of the unit, returned with the chunk
)
INTEGER_FIELDS = ("chunk_index",)
BOOLEAN_FIELDS = ("has_generated_text", "suspicious_instruction")
STORED_OBJECT_FIELDS = ("front_matter",)
"""Kept in `_source` only (`enabled: false`): any keys, nothing indexed. `front_matter` holds a
page's YAML header (cleaning rule `front_matter`, s5-2), whose keys differ from page to page."""
VECTOR_FIELD = "embedding"

VectorIndexType = Literal[
    "hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw", "flat", "int8_flat", "int4_flat", "bbq_flat"
]
_HNSW_TYPES = {"hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"}
_QUANTIZED_TYPES = {"int8_hnsw", "int4_hnsw", "bbq_hnsw", "int8_flat", "int4_flat", "bbq_flat"}

DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 100
DEFAULT_OVERSAMPLE = 3.0


class VectorIndexOptions(BaseModel):
    """`index_options` of the dense_vector field.

    Unset values get our defaults for the type: `m` and `ef_construction` only for HNSW types,
    `oversample` (rescore with full-precision vectors) only for quantized types. Setting a value
    the type does not support is an error. `bbq_disk` is not offered: it needs Enterprise.
    """

    model_config = ConfigDict(extra="forbid")

    type: VectorIndexType = "int8_hnsw"
    m: int | None = Field(default=None, ge=2, le=512)
    ef_construction: int | None = Field(default=None, ge=1, le=3200)
    oversample: float | None = Field(default=None, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def _fill_per_type(self) -> Self:
        if self.type in _HNSW_TYPES:
            if self.m is None:
                self.m = DEFAULT_M
            if self.ef_construction is None:
                self.ef_construction = DEFAULT_EF_CONSTRUCTION
        elif self.m is not None or self.ef_construction is not None:
            raise ValueError(f"m and ef_construction are HNSW parameters; {self.type} is not HNSW")
        if self.type in _QUANTIZED_TYPES:
            if self.oversample is None:
                self.oversample = DEFAULT_OVERSAMPLE
        elif self.oversample is not None:
            raise ValueError(
                f"rescore oversample needs a quantized type; {self.type} stores floats"
            )
        return self

    def to_es(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.type in _HNSW_TYPES:
            out["m"] = self.m
            out["ef_construction"] = self.ef_construction
        if self.type in _QUANTIZED_TYPES:
            out["rescore_vector"] = {"oversample": float(self.oversample or DEFAULT_OVERSAMPLE)}
        return out


def build_index_body(
    *,
    analyzer: str,
    dims: int,
    vector: VectorIndexOptions | None = None,
    refresh_interval: str = "1s",
) -> dict[str, Any]:
    """`{"settings": ..., "mappings": ...}` for `indices.create`; a new dict on every call.

    `analyzer` is used by both text fields; it must exist in ES (built-in or plugin), otherwise
    index creation fails. One shard: BM25 IDF is per shard (design s7-3). No replicas: one node.
    """
    if dims <= 0:
        raise ValueError(f"dims must be > 0, got {dims}")
    if not analyzer:
        raise ValueError("analyzer must be a non-empty analyzer name")
    vector = vector or VectorIndexOptions()
    properties: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        properties[field] = {"type": "text", "analyzer": analyzer}
    properties[VECTOR_FIELD] = {
        "type": "dense_vector",
        "element_type": "float",
        "dims": dims,
        "similarity": "cosine",
        "index": True,
        "index_options": vector.to_es(),
    }
    for field in KEYWORD_FIELDS:
        properties[field] = {"type": "keyword"}
    for field in INTEGER_FIELDS:
        properties[field] = {"type": "integer"}
    for field in BOOLEAN_FIELDS:
        properties[field] = {"type": "boolean"}
    for field in STORED_OBJECT_FIELDS:
        properties[field] = {"type": "object", "enabled": False}
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": refresh_interval,
                "mapping": {"exclude_source_vectors": True},
            }
        },
        "mappings": {"dynamic": "strict", "properties": properties},
    }


def index_body_for_kb(
    cfg: KnowledgeBaseConfig, *, dims: int, vector: VectorIndexOptions | None = None
) -> dict[str, Any]:
    """Index body with the knowledge base's analyzer (`es.analyzer` in its YAML)."""
    return build_index_body(analyzer=cfg.es.analyzer, dims=dims, vector=vector)
