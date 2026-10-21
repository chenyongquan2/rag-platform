"""Spec P8-01 AC3: the permission isolation check over the HTTP API in OAuth mode (fakes).

Two callers, three knowledge bases (one open, two private), every tool including the stream, the
context of passages from hidden KBs and the result cache. The service must leak nothing; the
last tests break the setup on purpose and check that the checker notices.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, AuthCallerConfig, KnowledgeBaseConfig, ServiceSettings
from rag.eval.access import AccessCase, CallerSpec, chunk_ids, run_leak_check, sse_events
from rag.observe import shutdown_observability
from rag.security.tokens import make_token_verifier
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.auth import AUDIENCE, ISSUER, FakeIssuer
from rag.testing.serve import FakePipelineBuilder

KBS = ("hf_docs", "mt5", "pdf_demo")
CASES = [AccessCase(kb, f"question {i} about {kb}") for kb in KBS for i in range(2)]


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture(scope="module")
def issuer() -> FakeIssuer:
    return FakeIssuer()


def _configs(open_kbs: tuple[str, ...] = ("hf_docs",)) -> dict[str, KnowledgeBaseConfig]:
    return {
        kb: KnowledgeBaseConfig(
            kb=kb,
            description=f"{kb} pages",
            language="en",
            access=AccessConfig(callers=["*"] if kb in open_kbs else ["local"]),
        )
        for kb in KBS
    }


def _settings(tmp_path: Path, issuer: FakeIssuer, guest_groups: list[str]) -> ServiceSettings:
    return agent_settings(
        auth_mode="oauth",
        auth_issuer=ISSUER,
        auth_audience=AUDIENCE,
        auth_jwks_path=str(issuer.write_jwks(tmp_path / "jwks.json")),
        auth_callers=[
            AuthCallerConfig(scope="rag.local", caller="local", acl_groups=["public", "internal"]),
            AuthCallerConfig(scope="rag.guest", caller="guest", acl_groups=guest_groups),
        ],
    )


def _report(
    tmp_path: Path,
    issuer: FakeIssuer,
    *,
    open_kbs: tuple[str, ...] = ("hf_docs",),
    guest_groups: list[str] | None = None,
    guest_visible: frozenset[str] = frozenset({"hf_docs"}),
) -> Any:
    settings = _settings(tmp_path, issuer, guest_groups or ["public"])
    cache = ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings)
    service = SearchService(
        settings,
        configs=_configs(open_kbs),
        builder=FakePipelineBuilder({kb: ScriptedSearchBackend(kb) for kb in KBS}),
        answer_model=StreamingFakeChatModel(default_response="It is so [1]."),
        cache=cache,
    )
    callers = [
        CallerSpec(
            "local",
            {"Authorization": f"Bearer {issuer.token(scopes=('rag.local',))}"},
            frozenset(KBS),
        ),
        CallerSpec(
            "guest",
            {"Authorization": f"Bearer {issuer.token(scopes=('rag.guest',))}"},
            guest_visible,
        ),
    ]
    with TestClient(build_http_app(service, token_verifier=make_token_verifier(settings))) as c:
        assert service.wait_ready(5)
        return run_leak_check(c, callers, CASES, cache_case=CASES[0])


def test_nothing_leaks_across_callers_tools_contexts_and_the_cache(
    tmp_path: Path, issuer: FakeIssuer
) -> None:
    report = _report(tmp_path, issuer)
    summary = report.summary()
    assert report.leaked_chunks == 0 and report.exposures == 0, report.markdown()
    assert report.wrong_denials == 0
    assert report.cache_works is True
    tools = {c.tool for c in report.checks}
    assert tools == {
        "search_docs",
        "ask_docs",
        "ask_docs_stream",
        "get_context",
        "list_knowledge_bases",
    }
    # The guest really was refused: 4 private questions x 3 tools, and every private passage the
    # local caller saw was asked for by the guest through get_context.
    denied = [c for c in report.checks if c.expected == "denied"]
    assert {c.error for c in denied} == {"kb_not_found"}
    assert sum(c.tool == "get_context" for c in denied) >= 2
    assert summary["chunks_returned"] > 0


def test_the_checker_sees_a_kb_opened_to_the_wrong_caller(
    tmp_path: Path, issuer: FakeIssuer
) -> None:
    # mt5 is misconfigured as open to every caller, but the guest must not see it.
    report = _report(tmp_path, issuer, open_kbs=("hf_docs", "mt5"))
    assert report.exposures > 0 and report.leaked_chunks > 0
    # Passages themselves are counted, not only the KB name in the listing.
    for tool in ("search_docs", "ask_docs", "ask_docs_stream"):
        assert any(c.leaked for c in report.checks if c.tool == tool), tool


def test_the_checker_sees_a_cache_entry_shared_across_groups(
    tmp_path: Path, issuer: FakeIssuer
) -> None:
    # Same groups for both callers stand in for a cache key without the groups' hash: the
    # guest is then served the local caller's entry, which the checker must flag.
    report = _report(tmp_path, issuer, guest_groups=["public", "internal"])
    assert report.cache_works is False
    assert any(c.note.startswith("served from another") for c in report.checks)


def test_chunk_ids_and_sse_parsing() -> None:
    body = {"chunks": [{"chunk_id": "a:1#0"}], "x": {"nearest": [{"chunk_id": "b:2#1"}]}}
    assert list(chunk_ids(body)) == ["a:1#0", "b:2#1"]
    text = 'event: progress\ndata: {"a": 1}\n\n: heartbeat\n\nevent: answer\ndata: {"b": 2}\n\n'
    assert sse_events(text) == [("progress", {"a": 1}), ("answer", {"b": 2})]
