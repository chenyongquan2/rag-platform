"""Events of one ask_docs run, for the streaming endpoint (design s16-9; spec P3-04).

Nodes write them with LangGraph's stream writer (`runtime.stream_writer`, the `custom` stream
mode); `AgentRuntime` reads them from the graph's stream and calls the `emit(name, data)` function
the caller passed to `ask`. Without one the events are dropped: the run is the same either way.

    progress  {node, phase, round?, ms?, ...}   a node started or ended (`streamed`), the request
                                                waits behind its session (`queued`), the answer
                                                starts a new attempt (`retry`)
    token     {text, attempt}                   a piece of the answer text, written by
                                                `answer_node` only

Why the answer's text is not taken from LangGraph's `messages` stream mode, which the design
(s16-9) names: the answer model is wrapped by the token cap (`rag.agent.budget.guard_model`) with
`with_config(callbacks=[...])`, and that replaces the callbacks LangGraph attaches to the run, so
no message chunk is ever emitted for it (checked against langgraph 1.2.15: guarded model, no
chunks; unguarded, chunks tagged `langgraph_node="answer"`). It also could not say that the
answer layer started a second attempt. `answer_node` therefore writes the pieces `stream_answer`
yields. Text of any other node (route, rewrite, judge, HyDE) cannot reach a token event because
no other node writes one, and no progress event carries a query.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langgraph.runtime import Runtime

from rag.agent.context import AgentContext
from rag.agent.state import AgentState, RetrievalRound

logger = logging.getLogger(__name__)

StreamEmit = Callable[[str, dict[str, Any]], None]
"""`emit(name, data)`: `name` is `progress` or `token`; called from the run's event loop, so it
must not block."""

NodeFn = Callable[[AgentState, Runtime[AgentContext]], Awaitable[dict[str, Any]]]
Detail = Callable[[AgentState, Mapping[str, Any]], dict[str, Any]]


def write(runtime: Runtime[AgentContext], name: str, data: dict[str, Any]) -> None:
    """Hand one event to the run's stream."""
    runtime.stream_writer({"event": name, "data": data})


def progress(runtime: Runtime[AgentContext], node: str, phase: str, **data: Any) -> None:
    write(runtime, "progress", {"node": node, "phase": phase, **data})


def safe_emit(emit: StreamEmit | None) -> StreamEmit | None:
    """`emit` that never raises: a listener that fails (a client that left) must not fail the
    run; the first failure is logged and the events after it are dropped."""
    if emit is None:
        return None
    broken = False

    def call(name: str, data: dict[str, Any]) -> None:
        nonlocal broken
        if broken:
            return
        try:
            emit(name, data)
        except Exception:
            broken = True
            logger.warning("the event listener failed; later events are dropped", exc_info=True)

    return call


def _unique(rounds: list[RetrievalRound]) -> set[str]:
    return {c.chunk_id for r in rounds for c in r.chunks}


def streamed(
    node: str, *, round_of: Callable[[AgentState], int] | None = None, detail: Detail | None = None
) -> Callable[[NodeFn], NodeFn]:
    """Make a graph node write `progress` start and end events. `round_of` gives the round the
    node works for (omitted for nodes that belong to none); `detail(state, update)` adds fields
    to the end event from what the node returned. A node that stops the run (`interrupt`) or
    fails writes no end event."""

    def decorate(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def run(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
            extra: dict[str, Any] = {"round": round_of(state)} if round_of else {}
            progress(runtime, node, "start", **extra)
            began = time.perf_counter()
            update = await fn(state, runtime)
            ms = round((time.perf_counter() - began) * 1000.0, 3)
            fields = detail(state, update) if detail else {}
            progress(runtime, node, "end", **extra, ms=ms, **fields)
            return update

        return run

    return decorate


def retrieve_detail(state: AgentState, update: Mapping[str, Any]) -> dict[str, Any]:
    """Chunks the round found (above the threshold, once each) and all rounds' so far."""
    new: list[RetrievalRound] = list(update.get("rounds") or [])
    before: list[RetrievalRound] = list(state.get("rounds") or [])
    return {"chunks": len(_unique(new)), "chunks_total": len(_unique(before + new))}
