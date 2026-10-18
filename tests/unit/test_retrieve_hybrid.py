"""Spec P0-07 AC5-AC10: EsHybridRetriever over a scripted `_msearch` (design s9-1, s16-8)."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from elastic_transport import ApiResponseMeta, ConnectionError, ConnectionTimeout, HttpHeaders
from elasticsearch import ApiError
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

from rag.config import RetrievalConfig
from rag.cost import BudgetExceeded
from rag.embed import CachedEmbedding, EmbeddingError, InMemoryEmbeddingCache, text_sha256
from rag.retrieve.deadline import Deadline
from rag.retrieve.hybrid import BM25, VECTOR, EsHybridRetriever, HybridQuery, RetrievalError
from rag.testing import FakeEmbedding
from rag.testing.retrieval import FakeMsearchClient, es_hit, search_error, search_response
from rag.types import ErrorCategory, RouteHit

KB = "hf_docs"
ALIAS = "kb_hf_docs"
INDEX = "kb_hf_docs_v1"
DIM = 4


class _FailingEmbedding(BaseEmbedding):
    """Raises the given error on every query embedding."""

    model_name: str = "failing"
    error: Exception

    def _get_query_embedding(self, query: str) -> Embedding:
        raise self.error

    async def _aget_query_embedding(self, query: str) -> Embedding:
        raise self.error

    def _get_text_embedding(self, text: str) -> Embedding:
        raise self.error


class _SlowEmbedding(FakeEmbedding):
    delay_s: float = 1.0

    def _get_query_embedding(self, query: str) -> Embedding:
        time.sleep(self.delay_s)
        return super()._get_query_embedding(query)


def _retriever(
    es: FakeMsearchClient,
    embed: BaseEmbedding | None = None,
    *,
    config: RetrievalConfig | None = None,
    es_reserve_ms: int = 100,
    es_reserve_ratio: float = 0.2,
    es_margin_ms: int = 200,
) -> EsHybridRetriever:
    return EsHybridRetriever(
        es,
        embed or FakeEmbedding(dim=DIM),
        kb=KB,
        index=ALIAS,
        config=config or RetrievalConfig(),
        es_reserve_ms=es_reserve_ms,
        es_reserve_ratio=es_reserve_ratio,
        es_margin_ms=es_margin_ms,
    )


def _query(
    text: str = "where are the logs?", *, deadline_ms: int = 5000, **kw: object
) -> HybridQuery:
    return HybridQuery(text, acl_groups=("public",), deadline=Deadline.after_ms(deadline_ms), **kw)  # type: ignore[arg-type]


def _frozen_clock(now: float = 100.0) -> Callable[[], float]:
    return lambda: now


BM25_HITS = search_response(
    [
        es_hit("p#0", 12.5, content="Logs are in the Logs tab.", title_path="Guide > Logs"),
        es_hit("q#0", 9.0, content="Other text."),
    ]
)
VECTOR_HITS = search_response(
    [
        es_hit("q#0", 0.91, content="Other text."),
        es_hit("r#2", 0.88, content="Vector only."),
    ]
)


def test_one_msearch_with_both_routes_and_same_filter() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    config = RetrievalConfig(top_k_per_route=15, num_candidates=120, title_boost=3.0)
    embed = FakeEmbedding(dim=DIM)
    query = _query("how to log in", filters={"doc_type": "mdx"})
    _retriever(es, embed, config=config).retrieve(query)

    (call,) = es.calls  # one round trip for both routes
    assert call["index"] == ALIAS
    headers, bodies = call["searches"][0::2], call["searches"][1::2]
    assert headers == [{}, {}]
    bm25, knn = bodies
    expected_filter = [
        {"term": {"kb": KB}},
        {"terms": {"acl_groups": ["public"]}},
        {"term": {"doc_type": "mdx"}},
    ]
    assert bm25["query"] == {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": "how to log in",
                        "fields": ["content", "title_path^3"],
                        "type": "best_fields",
                    }
                }
            ],
            "filter": expected_filter,
        }
    }
    assert bm25["size"] == 15
    # The filter sits inside knn (applied during the search, not after it); no post-filter.
    assert "query" not in knn and "post_filter" not in knn
    assert knn["knn"] == {
        "field": "embedding",
        "query_vector": embed.get_query_embedding("how to log in"),
        "k": 15,
        "num_candidates": 120,
        "filter": expected_filter,
    }
    assert knn["size"] == 15
    for body in bodies:
        assert body["track_total_hits"] is False
        assert "embedding" not in body["_source"]["includes"]
        assert {"content", "title_path", "doc_id", "chunk_id"} <= set(body["_source"]["includes"])


def test_retrieve_fuses_and_keeps_route_lists() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    query = _query()
    nodes = _retriever(es).retrieve(query)

    # q: 1/62 + 1/61; p: 1/61; r: 1/62.
    assert [n.node.node_id for n in nodes] == ["q#0", "p#0", "r#2"]
    assert [n.score for n in nodes] == pytest.approx([1 / 62 + 1 / 61, 1 / 61, 1 / 62])
    p = nodes[1].node
    assert p.get_content() == "Logs are in the Logs tab."
    assert p.metadata["title_path"] == "Guide > Logs"
    assert p.metadata["doc_id"] == "p"
    assert p.metadata["route_ranks"] == {BM25: 1}
    assert p.metadata["route_scores"] == {BM25: 12.5}
    assert nodes[0].node.metadata["route_ranks"] == {BM25: 2, VECTOR: 1}
    assert nodes[0].node.metadata["route_scores"] == {BM25: 9.0, VECTOR: 0.91}

    trace = query.trace
    assert trace.routes == {
        BM25: [
            RouteHit(chunk_id="p#0", doc_id="p", rank=1, score=12.5),
            RouteHit(chunk_id="q#0", doc_id="q", rank=2, score=9.0),
        ],
        VECTOR: [
            RouteHit(chunk_id="q#0", doc_id="q", rank=1, score=0.91),
            RouteHit(chunk_id="r#2", doc_id="r", rank=2, score=0.88),
        ],
    }
    assert trace.degraded is False
    assert trace.failed_routes == {}
    assert trace.index_version == INDEX
    assert trace.candidates == 3
    assert {"embed", "search", "fuse"} <= set(trace.timings_ms)


@pytest.mark.parametrize(("failed", "kept"), [(VECTOR, BM25), (BM25, VECTOR)])
def test_one_route_error_degrades(failed: str, kept: str) -> None:
    error = search_error(400, "search_phase_execution_exception", "failed to create query")
    scripted = {BM25: BM25_HITS, VECTOR: VECTOR_HITS, failed: error}
    es = FakeMsearchClient(bm25=scripted[BM25], vector=scripted[VECTOR])
    query = _query()
    nodes = _retriever(es).retrieve(query)

    expected = {BM25: ["p#0", "q#0"], VECTOR: ["q#0", "r#2"]}[kept]
    assert [n.node.node_id for n in nodes] == expected
    assert query.trace.degraded is True
    assert list(query.trace.routes) == [kept]
    assert set(query.trace.failed_routes) == {failed}
    assert "400" in query.trace.failed_routes[failed]
    assert "search_phase_execution_exception" in query.trace.failed_routes[failed]


def test_one_route_timed_out_degrades() -> None:
    timed_out = search_response([es_hit("p#0", 1.0)], timed_out=True)  # partial: not used
    es = FakeMsearchClient(bm25=timed_out, vector=VECTOR_HITS)
    query = _query()
    nodes = _retriever(es).retrieve(query)
    assert [n.node.node_id for n in nodes] == ["q#0", "r#2"]
    assert query.trace.degraded is True
    assert query.trace.failed_routes == {BM25: "timeout"}


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingError("gateway 503", retryable=True, status_code=503),
        BudgetExceeded("budget stop reached"),
        RuntimeError("anything else"),
    ],
)
def test_embedding_failure_runs_bm25_only(error: Exception) -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    query = _query()
    nodes = _retriever(es, _FailingEmbedding(error=error)).retrieve(query)

    (call,) = es.calls
    assert len(call["searches"]) == 2  # header + BM25 body only
    assert "knn" not in call["searches"][1]
    assert [n.node.node_id for n in nodes] == ["p#0", "q#0"]
    assert query.trace.degraded is True
    assert query.trace.failed_routes[VECTOR].startswith(f"embedding: {type(error).__name__}")
    assert list(query.trace.routes) == [BM25]


def test_slow_embedding_runs_bm25_only_within_deadline() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    query = _query(deadline_ms=400)
    retriever = _retriever(es, _SlowEmbedding(dim=DIM, delay_s=1.5), es_reserve_ms=150)
    start = time.monotonic()
    nodes = retriever.retrieve(query)
    elapsed = time.monotonic() - start

    # Waited at most deadline - reserve (0.25 s) for the vector, then searched BM25 alone.
    assert elapsed < 1.0
    assert [n.node.node_id for n in nodes] == ["p#0", "q#0"]
    assert query.trace.degraded is True
    assert query.trace.failed_routes[VECTOR] == "embedding: timeout"


def test_both_routes_failed_raises() -> None:
    error = search_error(500, "search_phase_execution_exception", "all shards failed")
    es = FakeMsearchClient(bm25=error, vector=search_error(400, "x_content_parse_exception", "bad"))
    with pytest.raises(RetrievalError) as caught:
        _retriever(es).retrieve(_query())
    assert caught.value.category is ErrorCategory.RETRIEVAL_FAILED
    assert set(caught.value.route_errors) == {BM25, VECTOR}

    timed_out = search_response([], timed_out=True)
    es = FakeMsearchClient(bm25=timed_out, vector=timed_out)
    with pytest.raises(RetrievalError) as caught:
        _retriever(es).retrieve(_query())
    assert caught.value.category is ErrorCategory.RETRIEVAL_TIMEOUT

    # Embedding failed and BM25 failed: both routes are gone.
    es = FakeMsearchClient(bm25=error)
    failing = _FailingEmbedding(error=EmbeddingError("down", retryable=True))
    with pytest.raises(RetrievalError) as caught:
        _retriever(es, failing).retrieve(_query())
    assert caught.value.category is ErrorCategory.RETRIEVAL_FAILED


def _api_error(status: int) -> ApiError:
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=None,  # type: ignore[arg-type]
    )
    return ApiError("service unavailable", meta=meta, body={"error": {"type": "unavailable"}})


@pytest.mark.parametrize(
    ("raised", "category"),
    [
        (ConnectionTimeout("timed out"), ErrorCategory.RETRIEVAL_TIMEOUT),
        (ConnectionError("refused"), ErrorCategory.RETRIEVAL_FAILED),
        (_api_error(503), ErrorCategory.RETRIEVAL_FAILED),
    ],
)
def test_transport_errors_raise_categorized(raised: Exception, category: ErrorCategory) -> None:
    es = FakeMsearchClient(raises=raised)
    with pytest.raises(RetrievalError) as caught:
        _retriever(es).retrieve(_query())
    assert caught.value.category is category
    assert caught.value.__cause__ is raised


def test_expired_deadline_raises_timeout() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    ticks = iter([0.0, 10.0])  # created at 0 s, checked at 10 s: a 5 s deadline has passed
    deadline = Deadline.after_ms(5000, clock=lambda: next(ticks, 10.0))
    query = HybridQuery("q", acl_groups=("public",), deadline=deadline)
    with pytest.raises(RetrievalError) as caught:
        _retriever(es).retrieve(query)
    assert caught.value.category is ErrorCategory.RETRIEVAL_TIMEOUT
    assert es.calls == []


def test_deadline_passed_to_es() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    deadline = Deadline.after_ms(3000, clock=_frozen_clock())
    query = HybridQuery("q", acl_groups=("public",), deadline=deadline)
    _retriever(es, es_margin_ms=200).retrieve(query)

    (call,) = es.calls
    assert call["options"]["request_timeout"] == pytest.approx(3.0)
    # Each search gets its own ES time-out a margin earlier, so a slow route comes back as
    # timed_out (the other route still counts) before the HTTP request itself is cut.
    assert [b["timeout"] for b in call["searches"][1::2]] == ["2800ms", "2800ms"]

    # Without a deadline in the query, the retriever refuses: every query has a time limit.
    with pytest.raises(ValueError, match="deadline"):
        _retriever(es).retrieve(HybridQuery("q", acl_groups=("public",)))


def test_plain_query_bundle_is_refused() -> None:
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    with pytest.raises(TypeError, match="HybridQuery"):
        _retriever(es).retrieve("where are the logs?")
    # A HybridQuery without ACL groups is refused by the filter builder, before any request.
    with pytest.raises(ValueError, match="acl_groups"):
        _retriever(es).retrieve(HybridQuery("q", deadline=Deadline.after_ms(1000)))
    assert es.calls == []


def test_reserve_for_es_is_capped_and_proportional() -> None:
    retriever = _retriever(FakeMsearchClient(), es_reserve_ms=200, es_reserve_ratio=0.2)
    # min(200 ms, 20% of the request's total time limit)
    assert retriever.es_reserve_s(Deadline.after_ms(10_000)) == pytest.approx(0.2)
    assert retriever.es_reserve_s(Deadline.after_ms(1_000)) == pytest.approx(0.2)
    assert retriever.es_reserve_s(Deadline.after_ms(300)) == pytest.approx(0.06)
    with pytest.raises(ValueError, match="es_reserve_ratio"):
        _retriever(FakeMsearchClient(), es_reserve_ratio=1.0)

    # A short deadline still leaves time for the query vector: both routes run.
    for deadline_ms in (300, 800, 1000):
        es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
        query = _query(deadline_ms=deadline_ms)
        _retriever(es, es_reserve_ms=200).retrieve(query)
        assert query.trace.degraded is False, (deadline_ms, query.trace.failed_routes)
        assert set(query.trace.routes) == {BM25, VECTOR}


def _cached(vectors: dict[str, list[float]]) -> CachedEmbedding:
    """L0 cache in front of a gateway stand-in that must not be called."""
    cache = InMemoryEmbeddingCache()
    cache.put_many("m@4", {text_sha256(text): v for text, v in vectors.items()})
    gateway = _FailingEmbedding(error=AssertionError("the gateway must not be called"))
    return CachedEmbedding(gateway, cache, cache_model="m@4")


def test_cache_hit_is_used_when_the_network_budget_is_spent() -> None:
    now = [0.0]
    deadline = Deadline.after_ms(1000, clock=lambda: now[0])
    now[0] = 0.9  # 100 ms left, less than the 200 ms kept for the search: no gateway call

    vector = [0.5, 0.5, 0.5, 0.5]
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    query = HybridQuery("cached question", acl_groups=("public",), deadline=deadline)
    _retriever(es, _cached({"cached question": vector}), es_reserve_ms=200).retrieve(query)
    assert query.trace.degraded is False
    knn = es.calls[0]["searches"][3]["knn"]
    assert knn["query_vector"] == vector

    # A miss with no time for the gateway: BM25 only, and the gateway was not called.
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    query = HybridQuery("other question", acl_groups=("public",), deadline=deadline)
    _retriever(es, _cached({"cached question": vector}), es_reserve_ms=200).retrieve(query)
    assert query.trace.failed_routes == {VECTOR: "embedding: timeout"}


def test_index_version_without_hits() -> None:
    # No route returns a hit, so no hit names the index: the alias is resolved instead.
    empty = search_response([])
    es = FakeMsearchClient(bm25=empty, vector=empty, aliases={ALIAS: "kb_hf_docs_v3"})
    query = _query()
    assert _retriever(es).retrieve(query) == []
    assert query.trace.index_version == "kb_hf_docs_v3"
    assert query.trace.degraded is False
    assert es.alias_calls == [ALIAS]

    # With hits the index comes from them; no extra request.
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS, aliases={ALIAS: "other"})
    query = _query()
    _retriever(es).retrieve(query)
    assert query.trace.index_version == INDEX
    assert es.alias_calls == []

    # An unknown alias leaves it empty rather than failing the search.
    es = FakeMsearchClient(bm25=empty, vector=empty)
    query = _query()
    _retriever(es).retrieve(query)
    assert query.trace.index_version == ""


class _SlowGateway(FakeEmbedding):
    """A gateway stand-in that answers after `delay_s` (CachedEmbedding calls the batch method)."""

    delay_s: float = 2.0

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        time.sleep(self.delay_s)
        return super()._get_text_embeddings(texts)


def test_cache_miss_and_slow_gateway_still_leave_time_for_bm25() -> None:
    # 800 ms limit: 160 ms are kept for the search (20%). The L0 cache misses and the gateway
    # does not answer, so the wait ends with 160 ms left. The per-search ES time-out must still
    # be long enough for a search (here 17 ms, as measured): BM25 answers, the result is
    # degraded, not a retrieval_timeout.
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS, search_ms=17)
    embed = CachedEmbedding(
        _SlowGateway(dim=DIM, delay_s=2.0), InMemoryEmbeddingCache(), cache_model="m@4"
    )
    query = _query(deadline_ms=800)
    nodes = _retriever(es, embed, es_reserve_ms=200, es_margin_ms=200).retrieve(query)

    assert [n.node.node_id for n in nodes] == ["p#0", "q#0"]
    assert query.trace.degraded is True
    assert query.trace.failed_routes == {VECTOR: "embedding: timeout"}
    (call,) = es.calls
    per_search_ms = float(call["searches"][1]["timeout"].removesuffix("ms"))
    remaining_ms = call["options"]["request_timeout"] * 1000.0
    # At most half of what is left goes to the margin; ES gets the other half.
    assert per_search_ms >= remaining_ms / 2 - 1
    assert per_search_ms < remaining_ms


class _CountingEmbedding(FakeEmbedding):
    calls: int = 0

    def _get_query_embedding(self, query: str) -> Embedding:
        self.calls += 1
        return super()._get_query_embedding(query)


@pytest.mark.parametrize("route", [BM25, VECTOR])
def test_single_route_sends_one_search(route: str) -> None:
    """Spec P0-12 AC3: a single-route retriever sends only that search, skips the query
    embedding when the vector route is off, and is not degraded."""
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    embed = _CountingEmbedding(dim=DIM)
    retriever = EsHybridRetriever(
        es,
        embed,
        kb=KB,
        index=ALIAS,
        config=RetrievalConfig(),
        es_reserve_ms=100,
        es_margin_ms=200,
        routes=(route,),
    )
    query = _query()
    nodes = retriever.retrieve(query)

    (call,) = es.calls
    bodies = call["searches"][1::2]
    assert len(bodies) == 1
    assert ("knn" in bodies[0]) is (route == VECTOR)
    assert embed.calls == (1 if route == VECTOR else 0)
    assert set(query.trace.routes) == {route}
    assert query.trace.degraded is False
    assert query.trace.failed_routes == {}
    expected = [h.chunk_id for h in query.trace.routes[route]]
    assert [n.node.node_id for n in nodes] == expected


def test_unknown_route_is_refused() -> None:
    with pytest.raises(ValueError, match="route"):
        EsHybridRetriever(
            FakeMsearchClient(),
            FakeEmbedding(dim=DIM),
            kb=KB,
            index=ALIAS,
            config=RetrievalConfig(),
            es_reserve_ms=100,
            es_margin_ms=200,
            routes=("sparse",),
        )


def test_retriever_uses_rrf_weights() -> None:
    """Spec P0-13 AC3: the KB's `rrf_weights` reach the fusion."""
    es = FakeMsearchClient(bm25=BM25_HITS, vector=VECTOR_HITS)
    config = RetrievalConfig(rrf_weights={BM25: 3.0, VECTOR: 1.0})
    nodes = _retriever(es, config=config).retrieve(_query())

    # p: 3/61; q: 3/62 + 1/61; r: 1/62.
    assert [n.node.node_id for n in nodes] == ["q#0", "p#0", "r#2"]
    assert [n.score for n in nodes] == pytest.approx([3 / 62 + 1 / 61, 3 / 61, 1 / 62])
    config = RetrievalConfig(rrf_weights={BM25: 0.0, VECTOR: 1.0})
    nodes = _retriever(es, config=config).retrieve(_query())
    assert [n.node.node_id for n in nodes] == ["q#0", "r#2", "p#0"]
    from rag.config.kb import RRF_ROUTES
    from rag.retrieve.hybrid import ROUTES

    assert RRF_ROUTES == ROUTES  # the config's copy of the route names stays in step
