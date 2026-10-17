"""Spec P0-11: request and layer spans, request-id propagation, errors, auto-instrumentation."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from rag.observe import (
    attributes as attr,
)
from rag.observe import (
    current_request_id,
    current_trace_id,
    layer_span,
    request_span,
    set_error_category,
    shutdown_observability,
)
from rag.observe.context import current_span_id
from rag.testing import FakeChatModel, FakeEmbedding, memory_observability

OBSERVE_SRC = Path(__file__).resolve().parents[2] / "src" / "rag" / "observe"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _by_name(spans: tuple[ReadableSpan, ...] | list[ReadableSpan]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans}


def test_layer_spans_nest_under_request_span() -> None:
    with memory_observability() as obs:
        with (
            request_span(
                "search_docs",
                kb="hf_docs",
                caller="tester",
                config_version="cfg-1",
                data_generation=7,
            ) as scope,
            layer_span("retrieve", attributes={attr.ROUTE: "bm25"}),
            layer_span("rerank", attributes={attr.RERANK_N: 30}),
        ):
            pass
        spans = _by_name(obs.span_exporter.get_finished_spans())

    root, retrieve, rerank = spans["search_docs"], spans["retrieve"], spans["rerank"]
    assert root.parent is None
    assert retrieve.parent is not None and retrieve.parent.span_id == root.context.span_id
    assert rerank.parent is not None and rerank.parent.span_id == retrieve.context.span_id
    assert {s.context.trace_id for s in spans.values()} == {int(scope.trace_id, 16)}
    assert len(scope.request_id) == 32

    assert root.attributes[attr.TOOL] == "search_docs"
    assert root.attributes[attr.KB] == "hf_docs"
    assert root.attributes[attr.CALLER] == "tester"
    assert root.attributes[attr.CONFIG_VERSION] == "cfg-1"
    assert root.attributes[attr.DATA_GENERATION] == 7
    assert root.attributes[attr.REQUEST_ID] == scope.request_id
    for child in (retrieve, rerank):
        assert child.attributes[attr.REQUEST_ID] == scope.request_id
        assert child.attributes[attr.KB] == "hf_docs"
        assert child.attributes[attr.STEP] == child.name
    assert retrieve.attributes[attr.ROUTE] == "bm25"
    assert rerank.attributes[attr.RERANK_N] == 30


def test_request_id_propagation() -> None:
    with memory_observability():
        assert current_request_id() is None
        with request_span("search_docs", kb="k", request_id="req-1") as scope:
            assert scope.request_id == "req-1"
            assert current_request_id() == "req-1"

            async def in_task() -> tuple[str | None, str | None]:
                await asyncio.sleep(0)
                return current_request_id(), current_trace_id()

            assert asyncio.run(in_task()) == ("req-1", scope.trace_id)

            ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(ctx.run, current_request_id).result() == "req-1"

            # A plain thread starts with an empty context: callers must copy it.
            seen: list[str | None] = []
            thread = threading.Thread(target=lambda: seen.append(current_request_id()))
            thread.start()
            thread.join()
            assert seen == [None]

            with request_span("ask_docs", kb="k") as inner:
                assert current_request_id() == inner.request_id != "req-1"
            assert current_request_id() == "req-1"
        assert current_request_id() is None
        assert current_trace_id() is None

        with request_span("search_docs", kb="k") as a, request_span("search_docs", kb="k") as b:
            assert a.request_id != b.request_id


class BoomError(RuntimeError):
    pass


def test_span_error_records_category() -> None:
    def classify(exc: BaseException) -> str | None:
        return "rerank_timeout" if isinstance(exc, BoomError) else None

    with memory_observability() as obs:
        with (
            pytest.raises(BoomError, match="too slow"),
            request_span("search_docs", kb="k"),
            layer_span("rerank", classify=classify),
        ):
            raise BoomError("too slow")
        with request_span("search_docs", kb="k"), layer_span("retrieve") as span:
            set_error_category("degraded")
            assert span.is_recording()
        with pytest.raises(ValueError), request_span("search_docs", kb="k"):
            raise ValueError("bug")
        spans = obs.span_exporter.get_finished_spans()

    rerank = next(s for s in spans if s.name == "rerank")
    assert rerank.status.status_code == StatusCode.ERROR
    assert rerank.attributes[attr.ERROR_TYPE] == "BoomError"
    assert rerank.attributes[attr.ERROR_CATEGORY] == "rerank_timeout"
    assert [e.name for e in rerank.events] == ["exception"]
    first_root = next(s for s in spans if s.name == "search_docs")
    # The category found by the inner span travels with the exception to the request span.
    assert first_root.status.status_code == StatusCode.ERROR
    assert first_root.attributes[attr.ERROR_CATEGORY] == "rerank_timeout"

    retrieve = next(s for s in spans if s.name == "retrieve")
    assert retrieve.attributes[attr.ERROR_CATEGORY] == "degraded"
    assert retrieve.status.status_code == StatusCode.UNSET  # degraded is a partial success

    bug_root = [s for s in spans if s.name == "search_docs"][-1]
    assert bug_root.status.status_code == StatusCode.ERROR
    assert bug_root.attributes[attr.ERROR_TYPE] == "ValueError"
    assert bug_root.attributes[attr.ERROR_CATEGORY] == attr.UNCLASSIFIED


def test_layer_span_decorator_sync_and_async() -> None:
    @layer_span("embed_query")
    def embed(x: int) -> tuple[int, str | None]:
        return x, current_trace_id()

    span_after_await: list[str | None] = []

    @layer_span("rerank")
    async def rerank(x: int) -> int:
        await asyncio.sleep(0)
        span_after_await.append(current_span_id())
        return x * 2

    with memory_observability() as obs:
        with request_span("search_docs", kb="k") as scope:
            assert embed(1) == (1, scope.trace_id)
            assert embed(2)[0] == 2
            assert asyncio.run(rerank(3)) == 6
        spans = obs.span_exporter.get_finished_spans()

    root = next(s for s in spans if s.name == "search_docs")
    children = [s for s in spans if s is not root]
    assert sorted(s.name for s in children) == ["embed_query", "embed_query", "rerank"]
    for child in children:
        assert child.parent is not None and child.parent.span_id == root.context.span_id
    rerank_span = next(s for s in children if s.name == "rerank")
    # The span is opened when the coroutine runs and is still current after the await.
    assert span_after_await == [format(rerank_span.context.span_id, "016x")]


class _GraphState(TypedDict):
    text: str


def _run_frameworks() -> None:
    """One LlamaIndex embedding call, one LangChain chat call and a one-node LangGraph graph,
    each inside its own layer span, all inside one request span. Fakes only, no network."""
    graph_model = FakeChatModel(responses=["graph answer"])
    builder = StateGraph(_GraphState)
    builder.add_node("answer", lambda s: {"text": str(graph_model.invoke(s["text"]).content)})
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    with request_span("ask_docs", kb="k"):
        with layer_span("embed_query"):
            FakeEmbedding(dim=8).get_query_embedding("secret question")
        with layer_span("generate"):
            FakeChatModel(responses=["chat answer"]).invoke("secret prompt")
        with layer_span("orchestrate"):
            graph.invoke({"text": "secret graph input"})


def test_auto_instrumentation_nests_framework_spans() -> None:
    with memory_observability(otel_auto_instrument=True) as obs:
        _run_frameworks()
        spans = obs.span_exporter.get_finished_spans()

    ours = _by_name([s for s in spans if attr.STEP in (s.attributes or {}) or s.parent is None])
    root = ours["ask_docs"]
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}

    def parent_name(span: ReadableSpan) -> str | None:
        if span.parent is None:
            return None
        return next(s.name for s in spans if s.context.span_id == span.parent.span_id)

    embedding = next(s for s in spans if s.name == "FakeEmbedding.get_query_embedding")
    assert parent_name(embedding) == "embed_query"
    chat = [s for s in spans if s.name == "FakeChatModel"]
    assert "generate" in {parent_name(s) for s in chat}
    langgraph = next(s for s in spans if s.name == "LangGraph")
    assert parent_name(langgraph) == "orchestrate"
    # Vectors never go into traces (1024 floats per text).
    vectors = [v for s in spans for k, v in (s.attributes or {}).items() if "vector" in k]
    assert vectors and set(vectors) == {"__REDACTED__"}
    # Content is on by default (local P0): the prompt is visible.
    assert any("secret prompt" in str(v) for s in chat for v in (s.attributes or {}).values())


def test_auto_instrumentation_switch_and_content() -> None:
    with memory_observability(otel_auto_instrument=False) as obs:
        _run_frameworks()
        names = {s.name for s in obs.span_exporter.get_finished_spans()}
    assert names == {"ask_docs", "embed_query", "generate", "orchestrate"}

    with memory_observability(otel_auto_instrument=True, otel_trace_content=False) as obs:
        _run_frameworks()
        spans = obs.span_exporter.get_finished_spans()
    assert len(spans) > 4
    values = [str(v) for s in spans for v in (s.attributes or {}).values()]
    assert not [v for v in values if "secret" in v]


_OWN_NAME = re.compile(r"^(rag|gen_ai|error|service|vcs)\.[a-z0-9_.]+$")


def test_attribute_names_live_in_one_module() -> None:
    offenders: list[str] = []
    for path in sorted(OBSERVE_SRC.glob("*.py")):
        if path.name == "attributes.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and _OWN_NAME.match(node.value)
            ):
                offenders.append(f"{path.name}:{node.lineno} {node.value!r}")
    assert offenders == [], "attribute and metric names belong in rag/observe/attributes.py"

    names = [
        value
        for key, value in vars(attr).items()
        if key.isupper() and isinstance(value, str) and _OWN_NAME.match(value)
    ]
    assert len(names) == len(set(names)), "each name is defined once"
