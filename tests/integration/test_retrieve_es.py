"""Spec P0-07 AC13-AC14: hybrid retrieval against the compose Elasticsearch (design s9-1).

Small test indices with fake vectors check that the shared filter really restricts both routes
and that a real per-search error degrades. The known-question test reads the fake-vector index
`fake_kb_hf_docs` that `python -m rag.ingest.job --kb hf_docs --limit 50 --fake-embed` builds
(read only; skipped when it is missing).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from elasticsearch import Elasticsearch

from rag.config import RetrievalConfig, ServiceSettings, load_kb_configs, repo_path
from rag.eval.questions import load_question_set
from rag.index.manage import aliased_indices, index_embed_model, init_kb
from rag.index.mapping import build_index_body
from rag.index.names import KbIndexNames
from rag.retrieve import (
    BM25,
    VECTOR,
    Deadline,
    EsHybridRetriever,
    HybridQuery,
    make_hybrid_search,
)
from rag.testing import FakeEmbedding

pytestmark = pytest.mark.integration

NamesFactory = Callable[[], KbIndexNames]
DIM = 8
QUESTION_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"
KNOWN_QID = "huggingface_doc_qa_eval:025"
KNOWN_PAGE = "hf_docs:huggingface/hf-endpoints-documentation/blob/main/docs/source/guides/logs.mdx"


def _write_docs(es: Elasticsearch, names: KbIndexNames) -> str:
    """Index v1 behind the alias with five chunks; returns the concrete index name."""
    index, _ = init_kb(es, names, build_index_body(analyzer="standard", dims=DIM))
    embed = FakeEmbedding(dim=DIM)
    kb, other = names.kb, f"{names.kb}x"
    docs = [
        # chunk id, kb, doc_type, language, acl groups, source_rel, content
        ("d1", kb, "markdown", "en", ["public"], "guide/d1.md", "install the library with pip"),
        ("d2", kb, "mdx", "en", ["public"], "api/d2.mdx", "install the library from source pip"),
        ("d3", kb, "mdx", "zh", ["public"], "api/d3.mdx", "pip install the library zh"),
        ("d4", kb, "mdx", "en", ["team-a"], "team/d4.mdx", "install pip team notes"),
        ("d5", other, "mdx", "en", ["public"], "api/d5.mdx", "install pip other knowledge base"),
    ]
    operations: list[dict] = []
    for chunk, doc_kb, doc_type, language, groups, rel, content in docs:
        doc_id = f"{doc_kb}:{rel}"
        operations.append({"index": {"_index": index, "_id": f"{doc_id}#0"}})
        operations.append(
            {
                "kb": doc_kb,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#0",
                "content": content,
                "title_path": chunk,
                "source_rel": rel,
                "doc_type": doc_type,
                "language": language,
                "acl_groups": groups,
                "embedding": embed.get_text_embedding(content),
            }
        )
    resp = es.bulk(operations=operations, refresh=True)
    assert not resp["errors"], resp
    return index


def _retriever(es: Elasticsearch, names: KbIndexNames, dim: int = DIM) -> EsHybridRetriever:
    return EsHybridRetriever(
        es,
        FakeEmbedding(dim=dim),
        kb=names.kb,
        index=names.alias,
        config=RetrievalConfig(),
        es_reserve_ms=500,
        es_margin_ms=200,
    )


def _routes(
    retriever: EsHybridRetriever, *, acl: tuple[str, ...] = ("public",), **filters: object
) -> dict[str, set[str]]:
    query = HybridQuery(
        "install pip", acl_groups=acl, filters=filters, deadline=Deadline.after_ms(10_000)
    )
    retriever.retrieve(query)
    assert query.trace.degraded is False, query.trace.failed_routes
    return {
        name: {h.chunk_id.split(":", 1)[1].split("/")[-1].split(".")[0] for h in hits}
        for name, hits in query.trace.routes.items()
    }


def test_filter_restricts_both_routes(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    _write_docs(es, names)
    retriever = _retriever(es, names)

    # Without caller filters both routes see the three public chunks of this kb (the kNN
    # route returns all of them: k = 20 > 3).
    assert _routes(retriever) == {BM25: {"d1", "d2", "d3"}, VECTOR: {"d1", "d2", "d3"}}
    assert _routes(retriever, doc_type="mdx") == {BM25: {"d2", "d3"}, VECTOR: {"d2", "d3"}}
    assert _routes(retriever, doc_type="mdx", language="en") == {BM25: {"d2"}, VECTOR: {"d2"}}
    assert _routes(retriever, language=["zh", "en"], source_rel_prefix="guide/") == {
        BM25: {"d1"},
        VECTOR: {"d1"},
    }


def test_acl_and_kb_enforced_on_both_routes(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    _write_docs(es, names)
    retriever = _retriever(es, names)

    # d4 (group team-a) and d5 (another kb in the same index) never show up for public.
    public = _routes(retriever)
    assert all("d4" not in ids and "d5" not in ids for ids in public.values())
    assert _routes(retriever, acl=("team-a",)) == {BM25: {"d4"}, VECTOR: {"d4"}}
    assert _routes(retriever, acl=("team-a", "public"), doc_type="mdx", language="en") == {
        BM25: {"d2", "d4"},
        VECTOR: {"d2", "d4"},
    }


def test_real_route_error_degrades(es: Elasticsearch, make_names: NamesFactory) -> None:
    names = make_names()
    index = _write_docs(es, names)
    # A 4-dim query vector against 8-dim documents: Elasticsearch fails the kNN search inside
    # the _msearch (HTTP 200 for the request); the BM25 search still answers.
    retriever = _retriever(es, names, dim=4)
    query = HybridQuery("install pip", acl_groups=("public",), deadline=Deadline.after_ms(10_000))
    nodes = retriever.retrieve(query)

    assert query.trace.degraded is True
    assert list(query.trace.routes) == [BM25]
    assert query.trace.failed_routes[VECTOR].startswith("400 ")
    assert {n.node.metadata["doc_id"].rsplit("/", 1)[-1] for n in nodes} == {
        "d1.md",
        "d2.mdx",
        "d3.mdx",
    }
    assert query.trace.index_version == index


def test_known_question_fake_index(es: Elasticsearch) -> None:
    settings = ServiceSettings(embed_backend="fake")
    names = KbIndexNames("hf_docs", prefix="fake_")
    indices = aliased_indices(es, names)
    if not indices:
        pytest.skip(f"{names.alias} not built; run the ingest job with --limit 50 --fake-embed")
    recorded = index_embed_model(es, indices[0])
    if recorded != f"fake-embedding@{settings.embed_dim}":
        pytest.skip(f"{names.alias} holds vectors of {recorded!r}")
    cfg = load_kb_configs(repo_path(settings.kb_dir))["hf_docs"]
    questions = {
        q.qid: q for q in load_question_set(repo_path(QUESTION_SET), kb="hf_docs").questions
    }
    question = questions[KNOWN_QID]
    assert question.relevant_pages == [KNOWN_PAGE]

    search = make_hybrid_search(cfg, settings, acl_groups=["public"], index_prefix="fake_", es=es)
    try:
        result = search.retrieve(question.question, kb="hf_docs", top_k=10)
    finally:
        search.close()
    if KNOWN_PAGE not in {h.doc_id for hits in result.routes.values() for h in hits}:
        pytest.skip(f"{KNOWN_PAGE} is not in {names.alias}")

    assert result.degraded is False
    assert set(result.routes) == {BM25, VECTOR}
    assert KNOWN_PAGE in [c.doc_id for c in result.chunks]
    assert KNOWN_PAGE in [h.doc_id for h in result.routes[BM25][:10]]
    assert result.index_version == indices[0]
