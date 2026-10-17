"""Spec P0-07 AC4: drop chunks whose body text is the same (design s9-3)."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.retrieve.dedupe import TextDedupe, dedupe_key
from rag.retrieve.query import HybridQuery


def _node(chunk_id: str, text: str, score: float, **meta: object) -> NodeWithScore:
    metadata = {"doc_id": chunk_id.split("#")[0], **meta}
    return NodeWithScore(node=TextNode(id_=chunk_id, text=text, metadata=metadata), score=score)


def test_dedupe_keeps_first_equal_body() -> None:
    nodes = [
        _node("a#0", "Install with `pip install x`.\n\nThen run it.", 0.032),
        _node("b#0", "Something else.", 0.031),
        # Same body as a#0 up to case, Unicode width and whitespace: a duplicate.
        _node("c#3", "  install WITH `pip install x`.\nThen   run it. ", 0.030),
        _node("d#1", "Install with `pip install y`.\n\nThen run it.", 0.029),
        _node("e#0", "Something else.", 0.016),
    ]
    query = HybridQuery("q", acl_groups=("public",))
    kept = TextDedupe().postprocess_nodes(nodes, query_bundle=query)

    assert [n.node.node_id for n in kept] == ["a#0", "b#0", "d#1"]
    assert [n.score for n in kept] == [0.032, 0.031, 0.029]  # scores and order untouched
    assert query.trace.duplicates == {"c#3": "a#0", "e#0": "b#0"}
    assert query.trace.candidates_after_dedupe == 3

    # Without a HybridQuery (plain LlamaIndex use) it still dedupes, there is just no trace.
    plain = TextDedupe().postprocess_nodes(nodes, query_bundle=QueryBundle("q"))
    assert [n.node.node_id for n in plain] == ["a#0", "b#0", "d#1"]
    assert TextDedupe().postprocess_nodes([]) == []

    fullwidth_a = chr(0xFF21)  # NFKC turns it into "A"
    assert dedupe_key(f"{fullwidth_a}  b\r\nC") == dedupe_key("a b c")  # NFKC, case, spaces
    assert dedupe_key("a b") != dedupe_key("ab")


def test_dedupe_ignores_page_path_lines() -> None:
    # The same paragraph on two pages; each text carries its own page-path line, which must
    # not make the bodies differ. Other lines that only mention a path stay part of the body.
    path_a = "huggingface/hub-docs/blob/main/docs/hub/a.md"
    path_b = "huggingface/transformers/blob/main/docs/source/en/b.md"
    nodes = [
        _node("a#0", f"{path_a}\nShared paragraph.", 0.03, source_rel=path_a),
        _node(
            "b#2",
            f"Shared paragraph.\n https://github.com/{path_b} ",
            0.02,
            source_rel=path_b,
            source_url=f"https://github.com/{path_b}",
        ),
        _node("c#0", f"See {path_a}\nShared paragraph.", 0.01, source_rel="other.md"),
    ]
    kept = TextDedupe().postprocess_nodes(nodes)
    assert [n.node.node_id for n in kept] == ["a#0", "c#0"]

    assert dedupe_key(f"{path_a}\nShared paragraph.", drop_lines=[path_a]) == dedupe_key(
        "Shared paragraph."
    )
    # Only whole lines equal to a path are dropped, never the path inside a sentence.
    assert dedupe_key(f"See {path_a}", drop_lines=[path_a]) != dedupe_key("See")
    # Keys to compare are configurable.
    only_rel = TextDedupe(path_keys=("source_rel",))
    assert [n.node.node_id for n in only_rel.postprocess_nodes(nodes)] == ["a#0", "b#2", "c#0"]
