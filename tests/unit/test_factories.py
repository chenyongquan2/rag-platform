from __future__ import annotations

import threading
from pathlib import Path
from typing import TypedDict

import pytest
from llama_index.core import Document

from rag.factories import (
    LockTimeoutError,
    make_checkpointer,
    make_docstore,
    make_lock_provider,
    url_path,
)


def _file_url(path: Path) -> str:
    return "file://" + path.as_posix()


def test_url_path_keeps_relative_and_absolute_paths() -> None:
    assert url_path("file://data/docstore/hf.json") == Path("data/docstore/hf.json")
    assert url_path("sqlite://data/cp.db") == Path("data/cp.db")
    assert url_path("file://C:/x/y.json") == Path("C:/x/y.json")
    with pytest.raises(ValueError, match="scheme"):
        url_path("data/no-scheme.json")
    with pytest.raises(ValueError, match="path"):
        url_path("file://")


def test_file_docstore_persists_and_reloads(tmp_path: Path) -> None:
    url = _file_url(tmp_path / "docstore")
    store = make_docstore(url, "hf_docs")
    store.add_documents([Document(text="hello", id_="hf_docs:a.md:0")])
    store.persist()
    assert (tmp_path / "docstore" / "hf_docs.json").is_file()

    reloaded = make_docstore(url, "hf_docs")
    assert reloaded.document_exists("hf_docs:a.md:0")
    assert reloaded.get_document("hf_docs:a.md:0").text == "hello"
    assert not make_docstore(url, "other_kb").document_exists("hf_docs:a.md:0")
    with pytest.raises(ValueError, match="namespace"):
        make_docstore(url, "../escape")


class _State(TypedDict):
    count: int


def test_sqlite_checkpointer(tmp_path: Path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    def build(checkpointer):  # type: ignore[no-untyped-def]
        graph = StateGraph(_State)
        graph.add_node("inc", lambda s: {"count": s["count"] + 1})
        graph.add_edge(START, "inc")
        graph.add_edge("inc", END)
        return graph.compile(checkpointer=checkpointer)

    url = "sqlite://" + (tmp_path / "cp" / "langgraph.db").as_posix()
    config = {"configurable": {"thread_id": "session-1"}}

    with make_checkpointer(url) as saver:
        assert build(saver).invoke({"count": 1}, config) == {"count": 2}

    with make_checkpointer(url) as saver:
        assert build(saver).get_state(config).values == {"count": 2}

    with make_checkpointer("memory://") as saver:
        assert isinstance(saver, InMemorySaver)


@pytest.mark.parametrize("kind", ["file", "memory"])
def test_lock_providers(kind: str, tmp_path: Path) -> None:
    url = _file_url(tmp_path / "locks") if kind == "file" else "memory://"
    provider = make_lock_provider(url)
    outcome: list[str] = []

    def contender() -> None:
        try:
            with provider.lock("ingest:hf_docs", timeout=0.05):
                outcome.append("acquired")
        except LockTimeoutError:
            outcome.append("timeout")

    with provider.lock("ingest:hf_docs"):
        t = threading.Thread(target=contender)
        t.start()
        t.join()
        # A different name is independent.
        with provider.lock("session:abc", timeout=0.05):
            pass
    assert outcome == ["timeout"]

    t = threading.Thread(target=contender)
    t.start()
    t.join()
    assert outcome == ["timeout", "acquired"]


def test_unsupported_schemes_raise() -> None:
    with pytest.raises(NotImplementedError, match="postgresql"):
        make_docstore("postgresql://user@localhost/rag", "hf_docs")
    with pytest.raises(NotImplementedError, match="postgresql"):
        make_checkpointer("postgresql://user@localhost/rag")
    with pytest.raises(NotImplementedError, match="redis"):
        make_lock_provider("redis://localhost:6379/0")
