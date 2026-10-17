"""Retrieval layer (design s9): hybrid search over Elasticsearch and its postprocessors.

`HybridSearch.retrieve(question, kb=..., top_k=...)` -> `RetrievalResult`; inside, the LlamaIndex
chain `EsHybridRetriever` (BM25 + kNN in one `_msearch`, RRF) -> `TextDedupe`.
"""

from rag.retrieve.deadline import Deadline
from rag.retrieve.dedupe import TextDedupe, dedupe_key
from rag.retrieve.filters import CALLER_FILTER_KEYS, FilterError, build_es_filter
from rag.retrieve.fusion import FusedHit, rrf_fuse
from rag.retrieve.hybrid import BM25, VECTOR, EsHybridRetriever, RetrievalError
from rag.retrieve.query import HybridQuery, RetrievalTrace
from rag.retrieve.search import (
    HybridSearch,
    IndexEmbedModelError,
    make_hybrid_search,
    node_to_chunk,
)

__all__ = [
    "BM25",
    "CALLER_FILTER_KEYS",
    "VECTOR",
    "Deadline",
    "EsHybridRetriever",
    "FilterError",
    "FusedHit",
    "HybridQuery",
    "HybridSearch",
    "IndexEmbedModelError",
    "RetrievalError",
    "RetrievalTrace",
    "TextDedupe",
    "build_es_filter",
    "dedupe_key",
    "make_hybrid_search",
    "node_to_chunk",
    "rrf_fuse",
]
