"""`SearchPipeline`: the whole retrieval chain of design s9-1, as search_docs and the eval use it.

    EsHybridRetriever -> TextDedupe -> BatchedTimeoutRerank -> RerankThreshold
    -> cut to top_k -> ContextExpander (optional)

The retriever sends BM25 and kNN in one `_msearch` and fuses them (P0-07); dedupe drops equal
bodies; the reranker scores the first `rerank_n` candidates with the process's cross-encoder
and stops by the request deadline (`rerank_timeout`); the threshold drops chunks whose rerank
score is too low (`not_found` with the best 3 when none is left); the list is cut to `top_k`;
the expander optionally adds neighbour or section chunks within a token budget. The result is
our `RetrievalResult`: fusion score in `Chunk.score`, rerank score in `Chunk.rerank_score`, each
route's own list, the reranker's list, the threshold decision, flags, step timings, the index
version, the data generation and a config version. `get_context` reads one chunk and its context
with the same caller identity.

`make_search_pipeline` builds it from the KB config and the service settings (s16-7); the
reranker model is loaded once per process (`get_rerank_runner`), so P0-09 calls the factory at
service start.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Literal

from elasticsearch import Elasticsearch
from llama_index.core.base.embeddings.base import BaseEmbedding
from redis import Redis

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.ingest.tokens import load_tokenizer
from rag.retrieve.context import ContextExpander, ContextMode, ContextReader, TokenCounter
from rag.retrieve.deadline import Deadline
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.hybrid import ROUTES, EsHybridRetriever
from rag.retrieve.page_groups import PageKeyDedupe
from rag.retrieve.rerank import BatchedTimeoutRerank, RerankRunner, get_rerank_runner
from rag.retrieve.search import HybridSearch, acl_group_tuple, hybrid_parts
from rag.retrieve.table_groups import EsTableFetch, TableSummaryReplace
from rag.retrieve.threshold import RerankThreshold
from rag.types import ContextResult

CONFIG_VERSION_LENGTH = 16


def search_config_version(
    cfg: KnowledgeBaseConfig, settings: ServiceSettings, embed_model_id: str
) -> str:
    """First 16 hex digits of the sha256 of what decides a search result besides the data: the
    whole KB config, the embedding model and the reranker (model and input length)."""
    kb_config = cfg.model_dump(mode="json", exclude={"description", "agent"})
    if not cfg.retrieval.dedupe_by_page_key:
        # A setting that is off changes no result: the version of a knowledge base that never
        # heard of it (the Hugging Face one) stays what it was before the setting existed.
        kb_config["retrieval"].pop("dedupe_by_page_key")
    if not cfg.retrieval.replace_table_summaries:
        kb_config["retrieval"].pop("replace_table_summaries")
    if not (
        cfg.retrieval.replace_table_summaries and cfg.retrieval.replace_table_summaries_after_rerank
    ):
        kb_config["retrieval"].pop("replace_table_summaries_after_rerank")
    payload = {
        # The description only tells callers what the KB holds (spec P0-09); `agent` (rounds of
        # the ask_docs graph, spec P3-01) is not part of a search. Neither changes a result.
        "kb_config": kb_config,
        "embed_model": embed_model_id,
        "reranker": {
            "model": settings.reranker_model,
            "max_length": settings.reranker_max_length,
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONFIG_VERSION_LENGTH]


def token_counter(model: str) -> TokenCounter:
    """Tokens of a text by the reranker's tokenizer (the unit chunk sizes are measured in)."""
    encode = load_tokenizer(model)

    def count(text: str) -> int:
        return len(encode(text))

    return count


class SearchPipeline(HybridSearch):
    """`HybridSearch` with rerank, threshold and optional expansion (see the module doc).

    Implements the P0-10 `Retriever` protocol: `retrieve(question, kb=..., top_k=...)`.
    `get_context` defaults to the expander's mode, window and budget (`chunk` without one).
    """

    def __init__(
        self,
        retriever: EsHybridRetriever,
        *,
        reranker: BatchedTimeoutRerank,
        threshold: RerankThreshold,
        acl_groups: Sequence[str],
        deadline_ms: int,
        expander: ContextExpander | None = None,
        reader: ContextReader | None = None,
        dedupe: TextDedupe | None = None,
        page_groups: PageKeyDedupe | None = None,
        table_groups: TableSummaryReplace | None = None,
        table_groups_after_rerank: bool = False,
        generation: Callable[[], int] | None = None,
        config_version: str = "",
        context_window: int = 1,
        context_max_tokens: int = 4000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            retriever,
            acl_groups=acl_groups,
            deadline_ms=deadline_ms,
            postprocessors=[
                dedupe or TextDedupe(),
                *([page_groups] if page_groups is not None else []),
                *(
                    [table_groups]
                    if table_groups is not None and not table_groups_after_rerank
                    else []
                ),
                reranker,
                threshold,
                *([table_groups] if table_groups is not None and table_groups_after_rerank else []),
            ],
            expander=expander,
            generation=generation,
            config_version=config_version,
            clock=clock,
        )
        self.reader = reader if reader is not None else (expander.reader if expander else None)
        self.context_mode: ContextMode = (
            expander.mode if expander is not None and expander.mode != "none" else "chunk"
        )
        self.context_window = expander.window if expander is not None else context_window
        self.context_max_tokens = (
            expander.max_tokens if expander is not None else context_max_tokens
        )

    def get_context(
        self,
        chunk_id: str,
        mode: ContextMode | None = None,
        *,
        deadline_ms: int | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> ContextResult:
        """`ContextReader.get_context` with this chain's KB settings, read with `acl_groups`
        (the request's caller, spec P0-09) or else the groups the chain was built with."""
        if self.reader is None:
            raise RuntimeError("this pipeline has no ContextReader")
        return self.reader.get_context(
            chunk_id,
            mode or self.context_mode,
            acl_groups=self.acl_groups if acl_groups is None else acl_group_tuple(acl_groups),
            window=self.context_window,
            max_tokens=self.context_max_tokens,
            deadline=Deadline.after_ms(deadline_ms or self.deadline_ms, clock=self.clock),
        )


def make_search_pipeline(
    cfg: KnowledgeBaseConfig,
    settings: ServiceSettings,
    *,
    acl_groups: Sequence[str],
    index_prefix: str = "",
    es: Elasticsearch | None = None,
    embed_model: BaseEmbedding | None = None,
    redis: Redis | None = None,
    runner: RerankRunner | None = None,
    count_tokens: TokenCounter | None = None,
    routes: Sequence[str] = ROUTES,
) -> SearchPipeline:
    """The chain over alias `<index_prefix>kb_<kb>` (checks as `make_hybrid_search`).

    `runner` defaults to the process's runner (loads the reranker on the first call);
    `count_tokens` to the reranker's tokenizer. Rerank and expansion parameters come from
    `cfg.retrieval` (`rerank_n`, `threshold`, `expand`, `expand_window`, `expand_max_tokens`).
    """
    retriever, generation, es, embed_id = hybrid_parts(
        cfg,
        settings,
        index_prefix=index_prefix,
        es=es,
        embed_model=embed_model,
        redis=redis,
        routes=routes,
    )
    retrieval = cfg.retrieval
    reader = ContextReader(
        es,
        kb=cfg.kb,
        index=retriever.index,
        count_tokens=count_tokens or token_counter(settings.reranker_model),
    )
    mode: Literal["none", "neighbors", "parent"] = retrieval.expand
    expander = (
        ContextExpander(
            reader=reader,
            mode=mode,
            window=retrieval.expand_window,
            max_tokens=retrieval.expand_max_tokens,
        )
        if mode != "none"
        else None
    )
    return SearchPipeline(
        retriever,
        reranker=BatchedTimeoutRerank(
            runner=runner if runner is not None else get_rerank_runner(settings),
            top_n=retrieval.rerank_n,
            include_title=retrieval.rerank_with_title,
        ),
        threshold=RerankThreshold(threshold=retrieval.threshold),
        page_groups=PageKeyDedupe() if retrieval.dedupe_by_page_key else None,
        table_groups=(
            TableSummaryReplace(EsTableFetch(es, kb=cfg.kb, index=retriever.index))
            if retrieval.replace_table_summaries
            else None
        ),
        table_groups_after_rerank=retrieval.replace_table_summaries_after_rerank,
        expander=expander,
        reader=reader,
        acl_groups=acl_groups,
        deadline_ms=settings.search_deadline_ms,
        generation=generation,
        config_version=search_config_version(cfg, settings, embed_id),
        context_window=retrieval.expand_window,
        context_max_tokens=retrieval.expand_max_tokens,
    )
