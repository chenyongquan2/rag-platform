"""Spec P3-04 AC1-AC5: the events of one ask_docs run (progress, token), as the runtime hands them
to `emit`, with scripted models and scripted retrieval only."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from rag.agent import AgentHooks, AgentRuntime, RouteDecision
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel
from rag.testing.agent import (
    ScriptedSearchBackend,
    agent_settings,
    hyde_reply,
    judge_reply,
    kb_config,
    multi_query_reply,
    scored_result,
    structured_reply,
)
from rag.types import AskResult

KB = "hf_docs"
BELOW = scored_result(KB, [0.1])  # nothing reaches the threshold of 0.3
PASSAGE = (
    "ZEBRAMARK In gamma delta mode the gamma option configures delta by default, and the "
    "documentation says gamma delta is set in the config file."
)

Event = tuple[str, dict[str, Any]]


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _run(
    question: str = "How do I push a model?",
    *,
    backend: ScriptedSearchBackend | None = None,
    answer: Any = None,
    helper: Any = None,
    judge: Any = None,
    agent: dict[str, Any] | None = None,
    hooks: AgentHooks | None = None,
    runtime: AgentRuntime | None = None,
    emit: bool = True,
    **kwargs: Any,
) -> tuple[AskResult, list[Event]]:
    events: list[Event] = []
    runtime = runtime or AgentRuntime.from_settings(agent_settings(), hooks=hooks or AgentHooks())

    async def go() -> AskResult:
        return await runtime.ask(
            question,
            cfg=kb_config(KB, **(agent or {})),
            retriever=backend or ScriptedSearchBackend(KB),
            model=answer or StreamingFakeChatModel(default_response="Push it [1]. Or upload [2]."),
            helper_model=helper if helper is not None else FakeChatModel(responses=[]),
            judge_model=judge if judge is not None else FakeChatModel(responses=[]),
            emit=(lambda name, data: events.append((name, data))) if emit else None,
            **kwargs,
        )

    return asyncio.run(go()), events


def _progress(events: list[Event]) -> list[dict[str, Any]]:
    return [d for n, d in events if n == "progress"]


def _two_hop(question: str, *, kb: str, history: Any) -> RouteDecision:
    return RouteDecision(retrieve=True, kb=kb, two_hop=True)


# ---- AC1 ------------------------------------------------------------------------------------


def test_progress_events_for_every_node_with_round_and_chunks() -> None:
    result, events = _run(agent={"judge": True}, judge=FakeChatModel(responses=[judge_reply(True)]))
    assert result.status == "answered"
    steps = [(p["node"], p["phase"]) for p in _progress(events)]
    assert steps == [
        ("route", "start"),
        ("route", "end"),
        ("rewrite", "start"),
        ("rewrite", "end"),
        ("retrieve", "start"),
        ("retrieve", "end"),
        ("judge", "start"),
        ("judge", "end"),
        ("answer", "start"),
        ("answer", "end"),
        ("verify", "start"),
        ("verify", "end"),
    ]
    by = {(p["node"], p["phase"]): p for p in _progress(events)}
    assert by[("retrieve", "start")]["round"] == 1
    end = by[("retrieve", "end")]
    assert end["round"] == 1 and end["chunks"] == 3 and end["chunks_total"] == 3
    assert end["ms"] >= 0
    assert by[("judge", "end")]["verdict"] == "enough"
    assert by[("route", "end")]["retrieve"] is True
    assert by[("answer", "end")]["found"] is True


def test_a_second_round_reports_its_round_and_the_chunks_found_so_far() -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "alpha setting": scored_result(KB, [0.9], pages=[1]),
            "beta setting": scored_result(KB, [0.1], pages=[3]),
            "beta option": scored_result(KB, [0.8, 0.5], pages=[5, 1]),
        },
    )
    helper = FakeChatModel(
        responses=[
            structured_reply(
                "RewritePlan",
                standalone_question="alpha setting and beta setting",
                search_queries=["alpha setting", "beta setting"],
            ),
            multi_query_reply((1, ["beta option"])),
        ]
    )
    judge = FakeChatModel(responses=[judge_reply(True, False), judge_reply(True)])
    _, events = _run(
        "alpha setting and beta setting",
        backend=backend,
        helper=helper,
        judge=judge,
        agent={"judge": True, "rewrite": True},
        hooks=AgentHooks(decide_route=_two_hop),
    )
    ends = [p for p in _progress(events) if p["node"] == "retrieve" and p["phase"] == "end"]
    assert [(p["round"], p["chunks"], p["chunks_total"]) for p in ends] == [(1, 1, 1), (2, 2, 2)]
    verdicts = [p["verdict"] for p in _progress(events) if p["node"] == "judge" and "verdict" in p]
    assert verdicts == ["partly_enough", "enough"]


# ---- AC2 ------------------------------------------------------------------------------------


def test_token_events_come_only_from_the_answer_node() -> None:
    reply = "Push it with push_to_hub [1]. Or upload [2]."
    helper = FakeChatModel(
        responses=[
            structured_reply(
                "RewritePlan",
                standalone_question="REWRITE-TEXT how to push",
                search_queries=["REWRITE-TEXT how to push"],
            )
        ]
    )
    judge = FakeChatModel(responses=[judge_reply(True)])
    result, events = _run(
        answer=StreamingFakeChatModel(default_response=reply, chunk_chars=4),
        helper=helper,
        judge=judge,
        agent={"judge": True, "rewrite": True},
    )
    tokens = [d for n, d in events if n == "token"]
    assert len(tokens) > 3  # a stream, not one piece
    assert "".join(t["text"] for t in tokens) == reply
    assert {t["attempt"] for t in tokens} == {1}
    # Everything before the first token is progress; no other event carries any model text.
    first = next(i for i, (n, _) in enumerate(events) if n == "token")
    assert all(n == "progress" for n, _ in events[:first])
    assert "REWRITE-TEXT" not in json.dumps(events)
    assert result.status == "answered"


def test_hyde_and_judge_text_never_reaches_a_token_event() -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "gamma delta": BELOW,
            "gamma epsilon": BELOW,
            PASSAGE: scored_result(KB, [0.8], pages=[10]),
        },
    )
    helper = FakeChatModel(
        responses=[multi_query_reply((1, ["gamma epsilon"])), hyde_reply((1, PASSAGE))]
    )
    judge = FakeChatModel(responses=[judge_reply(False), judge_reply(False), judge_reply(True)])
    result, events = _run(
        "gamma delta",
        backend=backend,
        helper=helper,
        judge=judge,
        agent={"judge": True, "rewrite": True},
        answer=StreamingFakeChatModel(default_response="Gamma is set in the config [1]."),
    )
    assert result.rounds == 3  # the HyDE round ran, so its text was in play
    dump = json.dumps(events)
    assert "ZEBRAMARK" not in dump and "gamma epsilon" not in dump
    assert "".join(d["text"] for n, d in events if n == "token") == (
        "Gamma is set in the config [1]."
    )


# ---- AC3 ------------------------------------------------------------------------------------


def test_a_retried_answer_attempt_is_marked() -> None:
    answer = StreamingFakeChatModel(
        responses=["No citation anywhere.", "Now it cites [1]."], chunk_chars=6
    )
    result, events = _run(answer=answer)
    assert result.status == "answered" and result.citation_check.parse_attempts == 2
    names = [(n, d.get("attempt"), d.get("phase")) for n, d in events if n in ("token", "progress")]
    retry_at = next(i for i, (n, _, phase) in enumerate(names) if phase == "retry")
    before = [a for n, a, _ in names[:retry_at] if n == "token"]
    after = [a for n, a, _ in names[retry_at + 1 :] if n == "token"]
    assert before and set(before) == {1} and after and set(after) == {2}
    retry = next(d for n, d in events if n == "progress" and d.get("phase") == "retry")
    assert retry["node"] == "answer" and retry["attempt"] == 2
    text2 = "".join(d["text"] for n, d in events if n == "token" and d["attempt"] == 2)
    assert text2 == "Now it cites [1]."


# ---- AC4 ------------------------------------------------------------------------------------


def _stable(result: AskResult) -> dict[str, Any]:
    data = result.model_dump(mode="json")
    for key in ("timings_ms", "thread_id", "request_id"):
        data.pop(key)
    return data


def test_the_streamed_run_gives_the_same_result_as_ask() -> None:
    kwargs: dict[str, Any] = {"request_id": "r-same", "session_id": "s1"}
    plain, no_events = _run(emit=False, **kwargs)
    streamed, events = _run(**kwargs)
    assert no_events == [] and events  # the plain call is a call without a listener
    assert _stable(plain) == _stable(streamed)
    assert plain.status == "answered" and streamed.answer == plain.answer


def test_a_listener_that_raises_does_not_break_the_run() -> None:
    # A closed client must not be able to fail the graph from inside `emit`.
    runtime = AgentRuntime.from_settings(agent_settings())

    def broken(name: str, data: dict[str, Any]) -> None:
        raise RuntimeError("client gone")

    async def go() -> AskResult:
        return await runtime.ask(
            "q",
            cfg=kb_config(KB),
            retriever=ScriptedSearchBackend(KB),
            model=StreamingFakeChatModel(default_response="Fine [1]."),
            emit=broken,
        )

    assert asyncio.run(go()).status == "answered"


# ---- AC5 ------------------------------------------------------------------------------------


def test_a_request_waiting_for_the_session_lock_reports_queued() -> None:
    gate = threading.Event()
    entered = threading.Event()

    def hold(question: str) -> None:
        if question == "first":
            entered.set()
            gate.wait(timeout=10)

    backend = ScriptedSearchBackend(KB, on_call=hold)
    runtime = AgentRuntime.from_settings(agent_settings())
    events: dict[str, list[Event]] = {"first": [], "second": [], "other": [], "alone": []}

    async def ask(question: str, session: str | None, key: str) -> AskResult:
        return await runtime.ask(
            question,
            cfg=kb_config(KB),
            retriever=backend,
            session_id=session,
            model=StreamingFakeChatModel(default_response="Fine [1]."),
            emit=lambda name, data: events[key].append((name, data)),
        )

    async def go() -> None:
        first = asyncio.create_task(ask("first", "s1", "first"))
        await asyncio.to_thread(entered.wait, 10)
        second = asyncio.create_task(ask("second", "s1", "second"))  # same session: waits
        other = asyncio.create_task(ask("other", "s2", "other"))  # another session: does not
        await other
        await asyncio.sleep(0.1)
        assert not second.done()
        gate.set()
        await asyncio.gather(first, second)
        await ask("alone", "s3", "alone")

    asyncio.run(go())
    queued = {k: [p for p in _progress(v) if p["node"] == "queued"] for k, v in events.items()}
    assert queued["second"] == [{"node": "queued", "phase": "waiting"}]
    assert events["second"][0][1]["node"] == "queued"  # before anything else
    assert queued["first"] == queued["other"] == queued["alone"] == []
    # Once it has the lock, the waiting request runs the same nodes as any other.
    assert ("route", "start") in [(p["node"], p["phase"]) for p in _progress(events["second"])]
