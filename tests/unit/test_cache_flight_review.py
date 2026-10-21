"""Spec P5-01 review round 1 (items 1, 3, 4): a follower that gives up or is cancelled does not
touch the leader or the other followers; only results the cache would store are shared; the wait
for another process is bounded by `cache_lock_wait_ms` and ends when the lock is released."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from rag.cache.exact import CacheConfig, CacheRun, Codec, ExactCache, _Flights
from rag.cache.keys import CacheKeyFields, make_fields
from rag.cache.layer import ResultCache
from rag.cache.rules import Entry, search_cacheable, search_from_payload, search_payload
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.testing import observe_settings
from rag.testing.agent import pages_result
from rag.types import AnswerStatus, AskResult, RetrievalResult


def _codec() -> Codec[RetrievalResult]:
    return Codec(to_payload=search_payload, verdict=search_cacheable, not_found=lambda r: False)


def _decode(entry: Entry) -> RetrievalResult:
    return search_from_payload(entry.payload, request_id="cached", data_generation=0, timings_ms={})


def _build(question: str = "How do I push?") -> Callable[[int], CacheKeyFields]:
    return lambda generation: make_fields(
        "search",
        kb="hf_docs",
        question=question,
        acl_groups=["public"],
        top_k=3,
        data_generation=generation,
        config_version="cfg",
    )


def _adapt(request_id: str) -> Callable[[RetrievalResult], RetrievalResult]:
    return lambda r: r.model_copy(update={"request_id": request_id})


class Compute:
    def __init__(self, seconds: float, **flags: Any) -> None:
        self.seconds = seconds
        self.calls = 0
        self.flags = flags
        self._lock = threading.Lock()

    def _result(self) -> RetrievalResult:
        with self._lock:
            self.calls += 1
            n = self.calls
        base = pages_result("hf_docs", [1, 2, 3])
        return base.model_copy(update={"request_id": f"computed-{n}", **self.flags})

    def __call__(self) -> RetrievalResult:
        result = self._result()
        time.sleep(self.seconds)
        return result

    async def acall(self) -> RetrievalResult:
        result = self._result()
        await asyncio.sleep(self.seconds)
        return result


def _aget(cache: ExactCache, compute: Compute, i: int, **kw: Any) -> Any:
    return cache.aget_or_compute(
        _build(), compute.acall, codec=_codec(), decode=_decode, adapt=_adapt(f"req-{i}"), **kw
    )


# ---- item 1: a follower that leaves does not hurt anyone -------------------------------------


def test_a_follower_that_times_out_does_not_kill_the_leader_or_the_other_followers() -> None:
    cache, compute = ExactCache(MemoryCacheBackend(), CacheConfig()), Compute(0.6)

    async def main() -> list[Any]:
        leader = asyncio.create_task(_aget(cache, compute, 0))
        while compute.calls < 1:
            await asyncio.sleep(0.005)
        impatient = asyncio.create_task(_aget(cache, compute, 1, wait_s=0.1))
        patient = asyncio.create_task(_aget(cache, compute, 2))
        return await asyncio.gather(leader, impatient, patient, return_exceptions=True)

    leader_run, impatient_run, patient_run = asyncio.run(main())
    assert isinstance(leader_run, CacheRun) and leader_run.outcome == "miss"
    assert isinstance(impatient_run, CacheRun) and impatient_run.result.chunks
    assert isinstance(patient_run, CacheRun) and patient_run.outcome == "shared"
    assert patient_run.result.request_id == "req-2"
    assert compute.calls == 2  # the leader, and the follower that gave up waiting


def test_a_cancelled_follower_does_not_kill_the_leader_or_the_other_followers() -> None:
    cache, compute = ExactCache(MemoryCacheBackend(), CacheConfig()), Compute(0.5)

    async def main() -> list[Any]:
        leader = asyncio.create_task(_aget(cache, compute, 0))
        while compute.calls < 1:
            await asyncio.sleep(0.005)
        gone = asyncio.create_task(_aget(cache, compute, 1))
        stays = asyncio.create_task(_aget(cache, compute, 2))
        await asyncio.sleep(0.2)  # both are waiting on the flight
        gone.cancel()
        return await asyncio.gather(leader, gone, stays, return_exceptions=True)

    leader_run, gone, stays_run = asyncio.run(main())
    assert isinstance(gone, asyncio.CancelledError)
    assert isinstance(leader_run, CacheRun) and leader_run.outcome == "miss"
    assert isinstance(stays_run, CacheRun) and stays_run.outcome == "shared"
    assert compute.calls == 1


def test_a_thread_follower_that_times_out_leaves_the_flight_alone() -> None:
    cache, compute = ExactCache(MemoryCacheBackend(), CacheConfig()), Compute(0.5)
    runs: dict[str, Any] = {}

    def go(name: str, **kw: Any) -> None:
        runs[name] = cache.get_or_compute(
            _build(), compute, codec=_codec(), decode=_decode, adapt=_adapt(name), **kw
        )

    leader = threading.Thread(target=go, args=("leader",))
    leader.start()
    while compute.calls < 1:
        time.sleep(0.005)
    patient = threading.Thread(target=go, args=("patient",))
    patient.start()
    go("impatient", wait_s=0.05)
    leader.join()
    patient.join()
    assert runs["leader"].outcome == "miss" and runs["patient"].outcome == "shared"
    assert runs["impatient"].result.chunks


def test_finishing_a_flight_that_was_cancelled_does_not_raise() -> None:
    flights = _Flights()
    future, leader = flights.begin("k")
    assert leader
    future.cancel()
    flights.finish("k", future, CacheRun(pages_result("hf_docs", [1]), "miss"))  # no error
    flights.finish("k", future)  # nor twice
    assert flights.begin("k")[1] is True  # and the key is free


def test_a_stream_follower_that_is_cancelled_leaves_the_leader_stream_alone() -> None:
    """The service level: a request that waits on an identical running request is cancelled (the
    client left); the running request finishes and the other waiting request gets its answer."""
    from rag.cache.layer import ResultCache
    from rag.security.access import Credentials
    from rag.serve.service import SearchService
    from rag.testing import FakeChatModel, StreamingFakeChatModel
    from rag.testing.agent import ScriptedSearchBackend, agent_settings
    from rag.testing.serve import FakePipelineBuilder

    settings = agent_settings()
    cache = ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings)
    backend = ScriptedSearchBackend("hf_docs", on_call=lambda q: time.sleep(0.4))
    service = SearchService(
        settings,
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": backend}),
        answer_model=StreamingFakeChatModel(default_response="Use push_to_hub [1]."),
        helper_model=FakeChatModel(responses=[]),
        cache=cache,
    )
    service.load()
    creds = Credentials(transport="http")

    async def ask(i: int, **kw: Any) -> Any:
        return await service.ask_docs(
            creds, kb="hf_docs", question="How do I push a model?", request_id=f"r{i}", **kw
        )

    async def main() -> list[Any]:
        leader = asyncio.create_task(ask(0))
        await asyncio.sleep(0.15)
        short = asyncio.create_task(ask(1, deadline_ms=300))  # its time runs out while it waits
        gone = asyncio.create_task(ask(2))
        normal = asyncio.create_task(ask(3))
        await asyncio.sleep(0.15)
        gone.cancel()
        return await asyncio.gather(leader, short, gone, normal, return_exceptions=True)

    leader, short, gone, normal = asyncio.run(main())
    assert not isinstance(leader, BaseException) and leader.answer
    assert isinstance(gone, asyncio.CancelledError)
    assert not isinstance(normal, BaseException) and normal.cache.outcome == "shared"
    assert not isinstance(short, BaseException) or "agent_deadline" in str(short)


# ---- item 3: only what the cache would store is shared ---------------------------------------


def test_an_incomplete_leader_result_is_not_handed_to_a_follower() -> None:
    cache = ExactCache(MemoryCacheBackend(), CacheConfig())
    compute = Compute(0.3, degraded=True)
    barrier = threading.Barrier(4)

    def go(i: int) -> CacheRun[RetrievalResult]:
        barrier.wait()
        return cache.get_or_compute(
            _build(), compute, codec=_codec(), decode=_decode, adapt=_adapt(f"req-{i}")
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(go, range(4)))
    assert all(r.outcome != "shared" for r in runs)  # everyone has a result of its own
    assert compute.calls == 4
    assert all(r.result.degraded for r in runs)


def test_a_clarification_and_its_resume_token_are_not_shared() -> None:
    settings = observe_settings()
    layer = ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings)
    cfg = KnowledgeBaseConfig(kb="hf_docs", language="en", access=AccessConfig(callers=["*"]))
    calls: list[int] = []

    async def compute() -> AskResult:
        calls.append(1)
        n = len(calls)
        await asyncio.sleep(0.3)
        return AskResult(
            request_id=f"r{n}",
            kb="hf_docs",
            status=AnswerStatus.NEED_CLARIFICATION,
            answer="Which one?",
            resume_token=f"token-of-caller-{n}",
            cache_allowed=False,
        )

    async def ask(i: int) -> Any:
        return await layer.ask(
            cfg,
            config_version="c",
            question="it?",
            acl_groups=["public"],
            request_id=f"req-{i}",
            compute=compute,
        )

    async def main() -> list[Any]:
        first = asyncio.create_task(ask(0))
        await asyncio.sleep(0.05)
        return [await first, *await asyncio.gather(ask(1), ask(2))]

    runs = asyncio.run(main())
    tokens = [r.result.resume_token for r in runs]
    assert len(set(tokens)) == len(tokens)  # each caller has the token of its own run
    assert all(r.outcome != "shared" for r in runs)


def test_a_shared_ask_result_never_carries_a_resume_token() -> None:
    settings = observe_settings()
    layer = ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings)
    cfg = KnowledgeBaseConfig(kb="hf_docs", language="en", access=AccessConfig(callers=["*"]))

    async def compute() -> AskResult:
        await asyncio.sleep(0.2)
        return AskResult(request_id="leader", kb="hf_docs", answer="x", resume_token="secret")

    async def ask(i: int) -> Any:
        return await layer.ask(
            cfg,
            config_version="c",
            question="q",
            acl_groups=["public"],
            request_id=f"req-{i}",
            compute=compute,
        )

    async def main() -> list[Any]:
        first = asyncio.create_task(ask(0))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(ask(1))
        return [await first, await second]

    runs = asyncio.run(main())
    follower = runs[1]
    assert follower.outcome == "shared" and follower.result.resume_token is None


# ---- item 4: the wait for another process ----------------------------------------------------


def test_a_second_process_stops_waiting_when_the_lock_is_released() -> None:
    """The first process computes 0.3 s and its result cannot be stored; the second must not
    wait the 5 s of its request deadline."""
    store = MemoryCacheBackend()
    one = ExactCache(store, CacheConfig(lock_poll_ms=5, lock_wait_ms=3000))
    two = ExactCache(store, CacheConfig(lock_poll_ms=5, lock_wait_ms=3000))
    compute = Compute(0.3, degraded=True)
    runs: dict[str, Any] = {}

    def first() -> None:
        runs["one"] = one.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)

    t = threading.Thread(target=first)
    t.start()
    while compute.calls < 1:
        time.sleep(0.005)
    began = time.perf_counter()
    runs["two"] = two.get_or_compute(_build(), compute, codec=_codec(), decode=_decode, wait_s=5.0)
    waited = time.perf_counter() - began
    t.join()
    assert runs["two"].outcome == "miss" and compute.calls == 2
    assert waited < 1.2  # about the 0.3 s the first process took, plus its own computation 0.3 s


def test_the_wait_for_another_process_is_capped_by_the_lock_wait_setting() -> None:
    store = MemoryCacheBackend()
    one = ExactCache(store, CacheConfig(lock_wait_ms=150, lock_poll_ms=5))
    two = ExactCache(store, CacheConfig(lock_wait_ms=150, lock_poll_ms=5))
    compute = Compute(1.0)
    t = threading.Thread(
        target=lambda: one.get_or_compute(_build(), compute, codec=_codec(), decode=_decode)
    )
    t.start()
    while compute.calls < 1:
        time.sleep(0.005)
    began = time.perf_counter()
    run = two.get_or_compute(
        _build(), Compute(0.0), codec=_codec(), decode=_decode, wait_s=30.0
    )  # a request with 30 s left
    waited = time.perf_counter() - began
    t.join()
    assert run.outcome == "miss" and waited < 0.6  # not 30 s, not the first process's second


def test_the_wait_for_another_process_in_the_async_flow_is_bounded_too() -> None:
    store = MemoryCacheBackend()
    one = ExactCache(store, CacheConfig(lock_wait_ms=3000, lock_poll_ms=5))
    two = ExactCache(store, CacheConfig(lock_wait_ms=3000, lock_poll_ms=5))
    compute = Compute(0.3, degraded=True)

    async def main() -> tuple[Any, float]:
        first = asyncio.create_task(_aget(one, compute, 0))
        while compute.calls < 1:
            await asyncio.sleep(0.005)
        began = time.perf_counter()
        run = await _aget(two, compute, 1, wait_s=5.0)
        await first
        return run, time.perf_counter() - began

    run, waited = asyncio.run(main())
    assert run.outcome == "miss" and waited < 1.2
