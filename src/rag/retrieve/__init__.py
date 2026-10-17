"""Retrieval layer (design s9, s10): hybrid search over Elasticsearch and its postprocessors.

`SearchPipeline.retrieve(question, kb=..., top_k=...)` -> `RetrievalResult`; inside, the
LlamaIndex chain `EsHybridRetriever` (BM25 + kNN in one `_msearch`, RRF) -> `TextDedupe` ->
`BatchedTimeoutRerank` (local cross-encoder) -> `RerankThreshold` -> top_k -> `ContextExpander`.
`HybridSearch` is the same chain without rerank (fusion order; the rerank latency curve uses it).
"""

from rag.retrieve.context import ContextExpander, ContextReader
from rag.retrieve.deadline import Deadline
from rag.retrieve.dedupe import TextDedupe, dedupe_key
from rag.retrieve.filters import CALLER_FILTER_KEYS, FilterError, build_es_filter
from rag.retrieve.fusion import FusedHit, rrf_fuse
from rag.retrieve.hybrid import BM25, VECTOR, EsHybridRetriever, RetrievalError
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline, search_config_version
from rag.retrieve.query import HybridQuery, RetrievalTrace
from rag.retrieve.rerank import (
    BatchedTimeoutRerank,
    ChunkReranker,
    RerankRunner,
    RerankScaleError,
    get_rerank_runner,
    load_cross_encoder,
)
from rag.retrieve.search import (
    HybridSearch,
    IndexEmbedModelError,
    make_hybrid_search,
    node_to_chunk,
)
from rag.retrieve.threshold import RerankThreshold

__all__ = [
    "BM25",
    "CALLER_FILTER_KEYS",
    "VECTOR",
    "BatchedTimeoutRerank",
    "ChunkReranker",
    "ContextExpander",
    "ContextReader",
    "Deadline",
    "EsHybridRetriever",
    "FilterError",
    "FusedHit",
    "HybridQuery",
    "HybridSearch",
    "IndexEmbedModelError",
    "RerankRunner",
    "RerankScaleError",
    "RerankThreshold",
    "RetrievalError",
    "RetrievalTrace",
    "SearchPipeline",
    "TextDedupe",
    "build_es_filter",
    "dedupe_key",
    "get_rerank_runner",
    "load_cross_encoder",
    "make_hybrid_search",
    "make_search_pipeline",
    "node_to_chunk",
    "rrf_fuse",
    "search_config_version",
]
