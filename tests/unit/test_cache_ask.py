"""Spec P5-01 AC7: the exact cache in `ask_docs` and in the stream (design s12-1, s16-8, s16-9):
a hit skips the graph and the model, a call with history or a session never reads the cache, an
incomplete answer is never written, the stream shows a hit as progress, answer and done."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.layer import ResultCache
from rag.cache.store import CacheUnavailable, MemoryCacheBackend
from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.security.access import Caller, Credentials
from rag.serve.http_app import build_http_app
from rag.serve.schema import AskResponse
from rag.serve.service import SearchService
from rag.testing import FakeChatModel, StreamingFakeChatModel, memory_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings, pages_result
from rag.testing.serve import FakePipelineBuilder
from rag.types import AnswerStatus, AskResult

CREDS = Credentials(transport="http")
REPLY = "Use push_to_hub [1]. Or upload_folder [2]."
QUESTION = "How do I push a model?"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class ByClient:
    def __init__(self, groups: dict[str, tuple[str, ...]]) -> None:
        self.groups = groups

    def resolve(self, credentials: Credentials) -> Caller:
        client = credentials.client_id or "local"
        return Caller(client, self.groups.get(client, ("public",)))


def _cfg(**agent: Any) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        kb="hf_docs",
        description="docs",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(**agent),
    )


def _layer(backend: MemoryCacheBackend | None = None, **settings: Any) -> ResultCache:
    backend = backend or MemoryCacheBackend()
    return ResultCache(ExactCache(backend, CacheConfig()), agent_settings(**settings))


def _service(
    cache: ResultCache | None,
    *,
    backend: ScriptedSearchBackend | None = None,
    model: Any = None,
    resolver: Any = None,
    agent: dict[str, Any] | None = None,
    **settings: Any,
) -> tuple[SearchService, ScriptedSearchBackend, Any]:
    backend = backend or ScriptedSearchBackend("hf_docs")
    model = model or StreamingFakeChatModel(default_response=REPLY, chunk_chars=6)
    service = SearchService(
        agent_settings(**settings),
        configs={"hf_docs": _cfg(**(agent or {}))},
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=model,
        helper_model=FakeChatModel(responses=[]),
        cache=cache,
        resolver=resolver,
    )
    service.load()
    return service, backend, model


def _ask(service: SearchService, **kw: Any) -> AskResponse:
    kw.setdefault("question", QUESTION)
    return asyncio.run(service.ask_docs(kw.pop("creds", CREDS), kb="hf_docs", **kw))


def _model_calls(model: Any) -> int:
    return len(model.calls)


def test_a_repeated_question_is_answered_from_the_cache_without_the_model() -> None:
    cache = _layer()
    service, backend, model = _service(cache)
    first = _ask(service, request_id="r1")
    calls = _model_calls(model)
    second = _ask(service, request_id="r2")
    assert (first.cache.outcome, second.cache.outcome, second.cache.hit) == ("miss", "hit", True)
    assert _model_calls(model) == calls and len(backend.calls) == 1
    assert second.request_id == "r2" and second.answer == first.answer == (
        "Use push_to_hub. Or upload_folder."
    )
    assert [s.text for s in second.sentences] == [s.text for s in first.sentences]
    assert [c.chunk_id for c in second.sources] == [c.chunk_id for c in first.sources]
    assert second.session_id is None and second.tokens_used == 0 and second.thread_id == ""
    assert second.status == "answered" and second.cache_allowed is True


def test_the_same_question_in_other_case_and_spacing_hits() -> None:
    cache = _layer()
    service, backend, _ = _service(cache)
    _ask(service)
    assert _ask(service, question="  how do I push   a model ").cache.outcome == "hit"
    assert len(backend.calls) == 1


def test_a_different_question_misses() -> None:
    cache = _layer()
    service, backend, _ = _service(cache)
    _ask(service)
    assert _ask(service, question="How do I pull a model?").cache.outcome == "miss"
    assert len(backend.calls) == 2


def test_another_acl_group_set_never_shares_an_answer() -> None:
    cache = _layer()
    resolver = ByClient({"alice": ("public",), "bob": ("public", "hr")})
    service, backend, _ = _service(cache, resolver=resolver)
    alice = Credentials(transport="http", client_id="alice")
    bob = Credentials(transport="http", client_id="bob")
    assert _ask(service, creds=alice).cache.outcome == "miss"
    assert _ask(service, creds=bob).cache.outcome == "miss"
    assert _ask(service, creds=alice).cache.outcome == "hit"
    assert _ask(service, creds=bob).cache.outcome == "hit"
    assert [set(c["acl_groups"]) for c in backend.calls] == [{"public"}, {"public", "hr"}]


def test_a_new_data_generation_misses() -> None:
    backend_store = MemoryCacheBackend()
    cache = _layer(backend_store)
    service, _, _model = _service(cache)
    _ask(service)
    assert _ask(service).cache.outcome == "hit"
    backend_store.bump_generation("hf_docs")
    assert _ask(service).cache.outcome == "miss"


def test_another_answer_config_or_prompt_misses() -> None:
    store = MemoryCacheBackend()
    one, _, _ = _service(_layer(store))
    _ask(one)
    two, _, _ = _service(_layer(store, agent_answer_max_tokens=900), agent_answer_max_tokens=900)
    assert _ask(two).cache.outcome == "miss"
    three, _, _ = _service(_layer(store), agent={"max_rounds": 2})
    assert _ask(three).cache.outcome == "miss"
    same, _, _ = _service(_layer(store))
    assert _ask(same).cache.outcome == "hit"


def test_a_call_with_a_session_never_reads_the_cache() -> None:
    cache = _layer()
    service, backend, _ = _service(cache)
    _ask(service)  # fills the cache
    with_session = _ask(service, session_id="s1")
    assert with_session.cache.outcome == "bypass" and len(backend.calls) == 2
    assert with_session.session_id == "s1"
    # ... but its complete answer (no history yet) may fill it for others
    other = _ask(service, question="Where are the docs?", session_id="s2")
    assert other.cache.outcome == "bypass"
    assert _ask(service, question="Where are the docs?").cache.outcome == "hit"


def test_an_answer_that_used_history_is_never_written() -> None:
    store = MemoryCacheBackend()
    service, _, _ = _service(_layer(store))
    _ask(service, question="How do I push a model?", session_id="s1")
    before = len(store.live_keys())
    follow = _ask(service, question="and a dataset?", session_id="s1")
    assert follow.used_history is True and follow.cache_allowed is False
    assert len(store.live_keys()) == before
    assert _ask(service, question="and a dataset?").cache.outcome == "miss"


def test_a_resume_token_call_is_not_cached() -> None:
    cache = _layer()
    service, backend, _ = _service(cache, agent={"clarify": True, "rewrite": True})
    # A call with a resume token that is unknown fails before the cache: no entry, no crash.
    from rag.serve.errors import ServiceError

    with pytest.raises(ServiceError):
        _ask(service, session_id="s1", resume_token="x" * 24, answer="more")
    assert backend.calls == [] and cache.exact.backend.live_keys() == []  # type: ignore[attr-defined]


def _result(**over: Any) -> AskResult:
    base: dict[str, Any] = {
        "request_id": "r",
        "kb": "hf_docs",
        "status": AnswerStatus.ANSWERED,
        "found": True,
        "answer": "x",
        "cache_allowed": True,
    }
    base.update(over)
    return AskResult(**base)


@pytest.mark.parametrize(
    "over",
    [
        {"degraded": True},
        {"rerank_timeout": True},
        {"answer_parse_failed": True},
        {"coverage_unconfirmed": True, "stop_reason": "deadline"},
        {"status": AnswerStatus.NEED_CLARIFICATION, "resume_token": "t" * 20},
        {"used_history": True, "cache_allowed": False},
        {"clarified": True},
        {"cache_allowed": False},
        {"generation_error": "llm_timeout"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_the_layer_never_writes_these_answers(over: dict[str, Any]) -> None:
    store = MemoryCacheBackend()
    layer = _layer(store)
    cfg = _cfg()

    async def compute() -> AskResult:
        return _result(**over)

    run = asyncio.run(
        layer.ask(
            cfg,
            config_version="cfg",
            question="q",
            acl_groups=["public"],
            request_id="r",
            compute=compute,
        )
    )
    assert run.outcome == "miss" and run.stored is False and run.not_stored_reason
    assert store.live_keys() == []


def test_a_store_that_is_down_never_fails_ask_docs() -> None:
    store = MemoryCacheBackend()
    store.fail_with = CacheUnavailable("ConnectionError")
    service, backend, _ = _service(_layer(store))
    answer = _ask(service)
    assert answer.cache.outcome == "error" and answer.answer and len(backend.calls) == 1


def test_concurrent_identical_questions_run_the_graph_once() -> None:
    cache = _layer()
    backend = ScriptedSearchBackend("hf_docs", on_call=lambda q: __import__("time").sleep(0.2))
    service, _, _model = _service(cache, backend=backend)

    async def main() -> list[AskResponse]:
        return list(
            await asyncio.gather(
                *(
                    service.ask_docs(CREDS, kb="hf_docs", question=QUESTION, request_id=f"c{i}")
                    for i in range(10)
                )
            )
        )

    responses = asyncio.run(main())
    assert len(backend.calls) == 1
    assert sorted(r.request_id for r in responses) == sorted(f"c{i}" for i in range(10))
    assert all(r.answer for r in responses)
    assert sorted(r.cache.outcome for r in responses).count("miss") == 1


# ---- the stream ------------------------------------------------------------------------------


def _sse(text: str) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            items.append((name, json.loads("\n".join(data))))
    return items


def _stream(service: SearchService, question: str = QUESTION) -> list[tuple[str, Any]]:
    with TestClient(build_http_app(service)) as client:
        service.wait_ready(5)
        r = client.post("/v1/ask_docs/stream", json={"kb": "hf_docs", "question": question})
        assert r.status_code == 200, r.text
        return _sse(r.text)


def test_a_stream_hit_is_progress_answer_done_without_tokens() -> None:
    cache = _layer()
    service, backend, _ = _service(cache)
    miss = _stream(service)
    assert [n for n, _ in miss].count("token") > 0
    answer_miss = next(d for n, d in miss if n == "answer")
    assert answer_miss["cache"]["outcome"] == "miss"

    hit = _stream(service)
    names = [n for n, _ in hit]
    assert names == ["progress", "answer", "done"]
    first = hit[0][1]
    assert first["node"] == "cache" and first["phase"] == "hit" and first["layer"] == "l1"
    answer = AskResponse.model_validate(hit[1][1])
    assert answer.cache is not None and answer.cache.hit is True and answer.cache.layer == "l1"
    assert answer.answer == AskResponse.model_validate(answer_miss).answer
    done = hit[2][1]
    assert done["cache"] == {"layer": "l1", "hit": True, "outcome": "hit"}
    assert done["status"] == "answered" and done["tokens_used"] == 0
    assert len(backend.calls) == 1


def test_a_stream_miss_has_its_usual_events_and_writes_only_a_clean_run() -> None:
    store = MemoryCacheBackend()
    service, _, _ = _service(_layer(store))
    events_ = _stream(service)
    names = [n for n, _ in events_]
    assert names[-1] == "done" and "answer" in names and "token" in names
    assert len(store.live_keys()) == 1
    # a degraded run (one route down) is streamed as usual but not written
    degraded = ScriptedSearchBackend(
        "hf_docs", default=pages_result("hf_docs", [1, 2, 3], degraded=True)
    )
    store2 = MemoryCacheBackend()
    service2, _, _ = _service(_layer(store2), backend=degraded)
    out = _stream(service2)
    assert next(d for n, d in out if n == "done")["degraded"] is True
    assert store2.live_keys() == []


def test_a_service_without_the_layer_streams_as_before() -> None:
    service, _, _ = _service(None)
    out = _stream(service)
    done = next(d for n, d in out if n == "done")
    assert "cache" not in done
    assert next(d for n, d in out if n == "answer")["cache"] is None


def test_hits_and_misses_are_counted_per_tool() -> None:
    from rag.observe import attributes as attr
    from rag.testing import metric_points

    cache = _layer()
    with memory_observability() as obs:
        service, _, _ = _service(cache)
        _ask(service)
        _ask(service)
        points = metric_points(obs, attr.METRIC_CACHE)
    got = {(p[0]["rag.tool"], p[0]["rag.cache.outcome"]): p[1].value for p in points}
    assert got == {("ask_docs", "miss"): 1, ("ask_docs", "hit"): 1}


def test_ask_answers_can_be_left_out_of_the_cache_by_setting() -> None:
    cache = _layer(cache_ask_answers=False)
    service, backend, _ = _service(cache, cache_ask_answers=False)
    first, second = _ask(service), _ask(service)
    assert first.cache is None and second.cache is None and len(backend.calls) == 2
    assert cache.exact.backend.live_keys() == []  # type: ignore[attr-defined]


def test_the_structured_output_method_is_part_of_the_answer_version() -> None:
    from rag.cache.keys import answer_config_version

    cfg = _cfg()
    a = answer_config_version(cfg, agent_settings())
    b = answer_config_version(cfg, agent_settings(chat_structured_output_method="json_mode"))
    assert a != b
