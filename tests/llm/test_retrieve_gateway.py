"""Spec P0-07 AC14: a known HF question with a real query embedding (one call, < 0.0001 CNY).

Run only when a spec asks for it: `uv run pytest -m llm tests/llm/test_retrieve_gateway.py`.
Needs the compose Elasticsearch with `kb_hf_docs` built from real vectors
(`python -m rag.ingest.job --kb hf_docs --limit 50`); reads it only. The query vector goes
through the query-profile client (Redis L0 cache, cost ledger, budget check).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from elastic_transport import TransportError
from elasticsearch import ApiError, Elasticsearch

from rag.config import ServiceSettings, load_kb_configs, repo_path
from rag.cost import BudgetExceeded, CostLedger
from rag.eval.questions import load_question_set
from rag.factories import make_es_client, make_redis_client
from rag.index.manage import aliased_indices, index_embed_model
from rag.index.names import KbIndexNames
from rag.retrieve import BM25, VECTOR, make_hybrid_search

pytestmark = pytest.mark.llm

QUESTION_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"
KNOWN_QID = "huggingface_doc_qa_eval:025"
KNOWN_PAGE = "hf_docs:huggingface/hf-endpoints-documentation/blob/main/docs/source/guides/logs.mdx"


@pytest.fixture
def es() -> Iterator[Elasticsearch]:
    client = make_es_client(ServiceSettings().es_url, request_timeout=30.0, max_retries=0)
    try:
        client.info()
    except (TransportError, ApiError) as exc:
        pytest.skip(f"Elasticsearch not reachable ({exc}); start it with scripts/dev_up.sh")
    yield client
    client.close()


def test_known_question_real_embedding(es: Elasticsearch) -> None:
    settings = ServiceSettings()
    if settings.embed_backend != "gateway" or not settings.embed_base_url:
        pytest.skip("the embedding gateway is not configured (EMBED_BASE_URL, EMBED_BACKEND)")
    names = KbIndexNames("hf_docs")
    indices = aliased_indices(es, names)
    if (
        not indices
        or index_embed_model(es, indices[0]) != f"{settings.embed_model}@{settings.embed_dim}"
    ):
        pytest.skip(f"{names.alias} is not built with real {settings.embed_model} vectors")
    questions = {
        q.qid: q for q in load_question_set(repo_path(QUESTION_SET), kb="hf_docs").questions
    }
    question = questions[KNOWN_QID]
    ledger = CostLedger.from_settings(settings)
    try:  # characters as tokens: an over-estimate for English
        ledger.check_budget(ledger.estimate(settings.embed_model, len(question.question)))
    except BudgetExceeded as exc:
        pytest.skip(str(exc))

    cfg = load_kb_configs(repo_path(settings.kb_dir))["hf_docs"]
    redis = make_redis_client(settings.redis_url)
    search = make_hybrid_search(cfg, settings, acl_groups=["public"], es=es, redis=redis)
    try:
        result = search.retrieve(question.question, kb="hf_docs", top_k=10)
    finally:
        search.close()
        redis.close()

    assert result.degraded is False, result.timings_ms
    assert set(result.routes) == {BM25, VECTOR}
    assert KNOWN_PAGE in [c.doc_id for c in result.chunks]
    assert KNOWN_PAGE in [h.doc_id for h in result.routes[VECTOR][:10]]
    assert result.index_version == indices[0]
    assert result.data_generation > 0
