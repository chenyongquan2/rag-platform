"""Spec P0-08 AC10-AC11: ContextExpander and get_context (design s10-3, s7-2, s7-6)."""

from __future__ import annotations

from typing import Any

import pytest
from elastic_transport import ConnectionTimeout
from llama_index.core.schema import NodeWithScore, TextNode

from rag.retrieve.context import ContextExpander, ContextReader
from rag.retrieve.deadline import Deadline
from rag.retrieve.query import HybridQuery
from rag.retrieve.search import node_to_chunk
from rag.testing.services import FakeElasticsearch
from rag.types import ContextPiece

KB = "hf_docs"
ALIAS = "kb_hf_docs"
INDEX = "kb_hf_docs_v1"
A = "hf_docs:a.md"
B = "hf_docs:b.md"


def _words(name: str) -> str:
    return f"{name} w w w"  # 4 tokens with the whitespace counter below


def count_tokens(text: str) -> int:
    return len(text.split())


def _source(doc: str, i: int, parent: str, groups: list[str] | None = None) -> dict[str, Any]:
    name = f"{doc.split(':')[1][0]}{i}"
    return {
        "chunk_id": f"{doc}#{i}",
        "doc_id": doc,
        "kb": KB,
        "content": _words(name),
        "title_path": f"{doc} > s",
        "source_rel": doc.split(":")[1],
        "chunk_index": i,
        "parent_id": parent,
        "acl_groups": groups or ["public"],
        "language": "en",
        "doc_type": "markdown",
    }


def _es() -> FakeElasticsearch:
    es = FakeElasticsearch()
    es.mappings[INDEX] = {"properties": {}}
    es.aliases[ALIAS] = INDEX
    docs: dict[str, dict[str, Any]] = {}
    for i in range(6):  # page A: two sections of three chunks
        docs[f"{A}#{i}"] = _source(A, i, f"{A}#s{i // 3}")
    for i in range(3):
        docs[f"{B}#{i}"] = _source(B, i, B)
    docs[f"{B}#3"] = _source(B, 3, B, groups=["team"])  # not visible to "public"
    other = _source("hf_docsx:a.md", 1, "x")  # another knowledge base, same page name
    other["kb"] = "hf_docsx"
    docs["hf_docsx:a.md#1"] = other
    es.docs[INDEX] = docs
    return es


def _reader(es: Any) -> ContextReader:
    return ContextReader(es, kb=KB, index=ALIAS, count_tokens=count_tokens)


def _node(doc: str, i: int, es: FakeElasticsearch) -> NodeWithScore:
    source = dict(es.docs[INDEX][f"{doc}#{i}"])
    text = source.pop("content")
    source.pop("acl_groups")
    return NodeWithScore(node=TextNode(id_=f"{doc}#{i}", text=text, metadata=source), score=0.5)


def _query(deadline: Deadline | None = None) -> HybridQuery:
    return HybridQuery("q", acl_groups=("public",), deadline=deadline or Deadline.after_ms(5000))


def _context(node: NodeWithScore) -> list[str]:
    return [p.chunk_id for p in node_to_chunk(node, KB).context]


def test_expander_adds_neighbours_deduped_with_selected() -> None:
    es = _es()
    expander = ContextExpander(reader=_reader(es), mode="neighbors", window=1, max_tokens=4000)
    # A#2 and A#3 are both selected (neither is context of the other); A#1 and A#3 share A#2.
    nodes = [_node(A, 2, es), _node(A, 3, es), _node(B, 1, es)]
    query = _query()
    out = expander.postprocess_nodes(nodes, query_bundle=query)

    assert [n.node.node_id for n in out] == [f"{A}#2", f"{A}#3", f"{B}#1"]
    assert _context(out[0]) == [f"{A}#1"]
    assert _context(out[1]) == [f"{A}#4"]
    assert _context(out[2]) == [f"{B}#0", f"{B}#2"]
    piece = node_to_chunk(out[2], KB).context[0]
    assert piece == ContextPiece(chunk_id=f"{B}#0", chunk_index=0, text="b0 w w w")
    assert query.trace.expanded == 4
    assert "context" not in node_to_chunk(out[0], KB).metadata

    nodes = [_node(A, 1, es), _node(A, 3, es)]
    out = expander.postprocess_nodes(nodes, query_bundle=_query())
    assert _context(out[0]) == [f"{A}#0", f"{A}#2"]
    assert _context(out[1]) == [f"{A}#4"]  # A#2 already given once

    # ACL and knowledge base come from the caller's query: B#3 (team only) is never added.
    out = expander.postprocess_nodes([_node(B, 2, es)], query_bundle=_query())
    assert _context(out[0]) == [f"{B}#1"]
    with pytest.raises(TypeError, match="HybridQuery"):
        from llama_index.core.schema import QueryBundle

        expander.postprocess_nodes([_node(B, 2, es)], query_bundle=QueryBundle("q"))


def test_expander_parent_mode_adds_section() -> None:
    es = _es()
    expander = ContextExpander(reader=_reader(es), mode="parent", window=1, max_tokens=4000)
    out = expander.postprocess_nodes([_node(A, 1, es), _node(A, 4, es)], query_bundle=_query())
    assert _context(out[0]) == [f"{A}#0", f"{A}#2"]
    assert _context(out[1]) == [f"{A}#3", f"{A}#5"]

    none = ContextExpander(reader=_reader(es), mode="none", window=1, max_tokens=4000)
    out = none.postprocess_nodes([_node(A, 1, es)], query_bundle=_query())
    assert _context(out[0]) == []


def test_expander_respects_token_budget() -> None:
    es = _es()
    # Selected A#2 has 4 tokens; budget 12 leaves 8: the two pieces at distance 1 fit, the
    # pieces at distance 2 do not.
    expander = ContextExpander(reader=_reader(es), mode="neighbors", window=2, max_tokens=12)
    query = _query()
    out = expander.postprocess_nodes([_node(A, 2, es)], query_bundle=query)
    assert _context(out[0]) == [f"{A}#1", f"{A}#3"]
    assert query.trace.expanded == 2

    # Selected chunks over the budget are kept (they are the result); nothing is added.
    expander = ContextExpander(reader=_reader(es), mode="neighbors", window=1, max_tokens=6)
    out = expander.postprocess_nodes([_node(A, 2, es), _node(B, 1, es)], query_bundle=_query())
    assert [n.node.node_id for n in out] == [f"{A}#2", f"{B}#1"]
    assert _context(out[0]) == [] and _context(out[1]) == []


class _FailingEs:
    def __init__(self) -> None:
        self.calls = 0

    def options(self, **_: Any) -> _FailingEs:
        return self

    def search(self, **_: Any) -> Any:
        self.calls += 1
        raise ConnectionTimeout("timed out")


def test_expander_skips_on_es_error_or_deadline() -> None:
    es = _es()
    nodes = [_node(A, 2, es)]
    failing = _FailingEs()
    expander = ContextExpander(reader=_reader(failing), mode="neighbors", window=1, max_tokens=99)
    query = _query()
    out = expander.postprocess_nodes(nodes, query_bundle=query)
    assert [n.node.node_id for n in out] == [f"{A}#2"]
    assert _context(out[0]) == []
    assert query.trace.expand_error == "ConnectionTimeout"

    clock_now = [100.0]
    expired = Deadline(at=50.0, clock=lambda: clock_now[0])
    query = _query(deadline=expired)
    failing.calls = 0
    out = expander.postprocess_nodes(nodes, query_bundle=query)
    assert failing.calls == 0
    assert query.trace.expand_error == "deadline"
    assert _context(out[0]) == []


def test_get_context_modes() -> None:
    reader = _reader(_es())

    def ids(mode: str, chunk: str, **kw: Any) -> list[str]:
        result = reader.get_context(chunk, mode, acl_groups=["public"], **kw)  # type: ignore[arg-type]
        assert result.found and not result.updated
        return [c.chunk_id for c in result.chunks]

    assert ids("chunk", f"{A}#2") == [f"{A}#2"]
    assert ids("neighbors", f"{A}#2") == [f"{A}#1", f"{A}#2", f"{A}#3"]
    assert ids("neighbors", f"{A}#0", window=2) == [f"{A}#0", f"{A}#1", f"{A}#2"]
    assert ids("parent", f"{A}#4") == [f"{A}#3", f"{A}#4", f"{A}#5"]
    assert ids("page", f"{B}#0") == [f"{B}#0", f"{B}#1", f"{B}#2"]  # B#3 is team only

    # Budget: the asked-for chunk first, then the nearest ones; page order in the result.
    result = reader.get_context(f"{A}#2", "page", acl_groups=["public"], max_tokens=12)
    assert [c.chunk_id for c in result.chunks] == [f"{A}#1", f"{A}#2", f"{A}#3"]
    assert result.truncated is True
    chunk = result.chunks[1]
    assert chunk.text == "a2 w w w"
    assert chunk.doc_id == A and chunk.title_path == [A, "s"]
    assert "acl_groups" not in chunk.metadata

    with pytest.raises(ValueError, match="mode"):
        reader.get_context(f"{A}#2", "everything", acl_groups=["public"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reader.get_context(f"{A}#2", "chunk", acl_groups="public")  # type: ignore[arg-type]


def test_get_context_falls_back_to_page_when_chunk_is_gone() -> None:
    reader = _reader(_es())
    # The page was re-chunked: A#9 is gone, the page's current content comes back, flagged.
    result = reader.get_context(f"{A}#9", "neighbors", acl_groups=["public"])
    assert result.found is True
    assert result.updated is True
    assert result.doc_id == A
    assert [c.chunk_id for c in result.chunks] == [f"{A}#{i}" for i in range(6)]

    # Within the budget, from the start of the page.
    result = reader.get_context(f"{A}#9", "chunk", acl_groups=["public"], max_tokens=8)
    assert [c.chunk_id for c in result.chunks] == [f"{A}#0", f"{A}#1"]
    assert result.truncated is True

    # A chunk the caller may not see counts as gone; the page shows only what it may see.
    result = reader.get_context(f"{B}#3", "chunk", acl_groups=["public"])
    assert result.updated is True
    assert [c.chunk_id for c in result.chunks] == [f"{B}#0", f"{B}#1", f"{B}#2"]

    # Neither chunk nor page: not found.
    result = reader.get_context("hf_docs:zzz.md#0", "chunk", acl_groups=["public"])
    assert result.found is False and result.chunks == [] and result.updated is False
    with pytest.raises(ValueError, match="doc_id#"):
        reader.get_context("no-hash", "chunk", acl_groups=["public"])
