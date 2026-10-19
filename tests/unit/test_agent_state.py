"""Spec P3-01 AC2: checkpoint thread ids."""

from __future__ import annotations

import pytest

from rag.agent import thread_id


def test_thread_id_format() -> None:
    assert thread_id("sess-1", "req-9") == "sess-1:req-9"
    assert thread_id("a.b_c-2", "0123456789abcdef0123456789abcdef") == (
        "a.b_c-2:0123456789abcdef0123456789abcdef"
    )
    # Without a session the thread is still one per request.
    assert thread_id(None, "req-9") == "-:req-9"
    # A colon in the session id would make "a:b" + "c" and "a" + "b:c" the same thread.
    for bad in ("a:b", "", "a b", "x" * 65, "é"):
        with pytest.raises(ValueError):
            thread_id(bad, "req-9")
