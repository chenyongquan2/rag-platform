"""Spec P8-01 AC3 against Elasticsearch: document ACL groups and the result cache, through the
HTTP API in OAuth mode.

A copy of `fake_kb_hf_docs_v1` (fake vectors, so no model is called) is restored into
`test<run>_kb_hf_docs_v1`; the Inference Endpoints pages in it are moved to the ACL group
`internal`. The local caller (groups public + internal) must find them, the guest (public) must
never get one of their chunks: not from search_docs, not through get_context, not from a cache
entry the local caller filled. Needs the reranker in the local Hugging Face cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from redis import Redis
from starlette.testclient import TestClient

from rag.config import AuthCallerConfig, ServiceSettings, load_kb_configs, repo_path
from rag.eval.access import AccessCase, CallerSpec
from rag.eval.access.leaks import LeakChecker
from rag.eval.questions import load_question_set
from rag.observe import shutdown_observability
from rag.ops.backup import export_index, restore_index
from rag.retrieve import rerank as rerank_module
from rag.security.tokens import make_token_verifier
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService
from rag.testing.auth import AUDIENCE, ISSUER, FakeIssuer

pytestmark = pytest.mark.integration

SOURCE = "fake_kb_hf_docs_v1"
INTERNAL_PREFIX = "huggingface/hf-endpoints-documentation/"
KNOWN_QID = "huggingface_doc_qa_eval:025"  # answered by an Inference Endpoints page
LOAD_TIMEOUT_S = 300


def known_question() -> str:
    path = repo_path("evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl")
    return next(q.question for q in load_question_set(path).questions if q.qid == KNOWN_QID)


@pytest.fixture(scope="module")
def restored(
    es: Elasticsearch, index_prefix: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[set[str]]:
    if not es.indices.exists(index=SOURCE):
        pytest.skip(f"{SOURCE} is missing (run the ingest job with EMBED_BACKEND=fake)")
    index, alias = f"{index_prefix}kb_hf_docs_v1", f"{index_prefix}kb_hf_docs"
    tmp = tmp_path_factory.mktemp("backup")
    export_index(es, SOURCE, tmp)
    restore_index(es, tmp / SOURCE, index)
    try:
        es.indices.put_alias(index=index, name=alias)
        es.update_by_query(
            index=index,
            query={"prefix": {"source_rel": INTERNAL_PREFIX}},
            script={"source": "ctx._source.acl_groups = ['internal']", "lang": "painless"},
            refresh=True,
            conflicts="abort",
        )
        hits = es.search(
            index=index, query={"term": {"acl_groups": "internal"}}, size=1000, source=False
        )["hits"]["hits"]
        internal = {h["_id"] for h in hits}
        assert internal, "no internal pages: the fake index changed?"
        yield internal
    finally:
        es.indices.delete(index=index, ignore_unavailable=True)


@pytest.fixture(scope="module")
def client(
    restored: set[str],
    index_prefix: str,
    redis_key_prefix: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[TestClient, list[CallerSpec]]]:
    from huggingface_hub import try_to_load_from_cache

    issuer = FakeIssuer()
    tmp: Path = tmp_path_factory.mktemp("auth")
    settings = ServiceSettings(  # type: ignore[call-arg]
        embed_backend="fake",
        search_index_prefix=index_prefix,
        otel_mode="memory",
        otel_auto_instrument=False,
        business_record_path=str(tmp / "business.jsonl"),
        log_dir="",
        log_stderr_level="OFF",
        service_allowed_hosts=["testserver"],
        cache_mode="l1",
        cache_key_prefix=f"{redis_key_prefix}l1:",
        rate_limit_mode="off",
        budget_daily_cny=0,
        budget_daily_caller_cny=0,
        auth_mode="oauth",
        auth_issuer=ISSUER,
        auth_audience=AUDIENCE,
        auth_jwks_path=str(issuer.write_jwks(tmp / "jwks.json")),
        auth_callers=[
            AuthCallerConfig(scope="rag.local", caller="local", acl_groups=["public", "internal"]),
            AuthCallerConfig(scope="rag.guest", caller="guest", acl_groups=["public"]),
        ],
    )
    if not isinstance(try_to_load_from_cache(settings.reranker_model, "config.json"), str):
        pytest.skip(f"{settings.reranker_model} is not in the local Hugging Face cache")
    configs = {"hf_docs": load_kb_configs(repo_path(settings.kb_dir))["hf_docs"]}
    shutdown_observability()
    rerank_module.reset_rerank_runners()
    service = SearchService.from_settings(settings, configs=configs)
    callers = [
        CallerSpec(
            "local",
            {"Authorization": f"Bearer {issuer.token(scopes=('rag.local',), expires_in_s=3600)}"},
            frozenset({"hf_docs"}),
        ),
        CallerSpec(
            "guest",
            {"Authorization": f"Bearer {issuer.token(scopes=('rag.guest',), expires_in_s=3600)}"},
            frozenset({"hf_docs"}),
        ),
    ]
    try:
        with TestClient(build_http_app(service, token_verifier=make_token_verifier(settings))) as c:
            assert service.wait_ready(LOAD_TIMEOUT_S), service.status()
            yield c, callers
    finally:
        service.close()
        rerank_module.reset_rerank_runners()
        shutdown_observability()
        redis = Redis.from_url(settings.redis_url, socket_timeout=2)
        keys = list(redis.scan_iter(match=f"{redis_key_prefix}*", count=500))
        if keys:
            redis.delete(*keys)
        redis.close()


def _doc_ids(body: Any) -> set[str]:
    chunks = (body.get("chunks") or []) + (body.get("below_threshold") or [])
    return {c["chunk_id"] for c in chunks}


def test_internal_chunks_reach_the_local_caller_and_never_the_guest(
    client: tuple[TestClient, list[CallerSpec]], restored: set[str]
) -> None:
    http, (local, guest) = client
    body = {"kb": "hf_docs", "query": known_question(), "top_k": 10, "deadline_ms": 60_000}
    seen_local = http.post("/v1/search_docs", json=body, headers=dict(local.headers)).json()
    local_ids = _doc_ids(seen_local)
    assert local_ids & restored, "the local caller should find the internal pages"

    seen_guest = http.post("/v1/search_docs", json=body, headers=dict(guest.headers)).json()
    assert _doc_ids(seen_guest), seen_guest
    assert not _doc_ids(seen_guest) & restored  # 0 leaked chunks

    for chunk_id in sorted(local_ids & restored)[:5]:
        for mode in ("chunk", "neighbors", "page"):
            r = http.post(
                "/v1/get_context",
                json={"chunk_id": chunk_id, "mode": mode},
                headers=dict(guest.headers),
            )
            assert r.status_code == 200, r.text
            got = {c["chunk_id"] for c in r.json()["chunks"]}
            assert r.json()["found"] is False and not got & restored, (mode, r.json())


def test_the_cache_does_not_hand_one_groups_results_to_another(
    client: tuple[TestClient, list[CallerSpec]], restored: set[str]
) -> None:
    http, (local, guest) = client
    checker = LeakChecker(http, [local, guest])
    case = AccessCase("hf_docs", known_question() + " (cache check)")
    assert checker.cache_isolation(local, guest, case)
    guest_check = checker.checks[-1]
    assert guest_check.cache_outcome == "miss"
