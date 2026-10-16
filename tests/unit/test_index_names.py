from __future__ import annotations

import pytest

from rag.index.names import KbIndexNames


def test_names_and_version_parsing() -> None:
    names = KbIndexNames("hf_docs")
    assert names.alias == "kb_hf_docs"
    assert names.index(1) == "kb_hf_docs_v1"
    assert names.index(12) == "kb_hf_docs_v12"
    assert names.pattern == "kb_hf_docs_v*"

    assert names.version_of("kb_hf_docs_v3") == 3
    assert names.version_of("kb_hf_docs") is None  # the alias is not a version
    assert names.version_of("kb_hf_docs_v0") is None
    assert names.version_of("kb_hf_docs_v01") is None
    assert names.version_of("kb_hf_docs_v2_old") is None
    assert names.version_of("kb_other_v1") is None
    # The pattern of kb "hf" also matches indices of kb "hf_v2"; parsing must not.
    assert KbIndexNames("hf").version_of("kb_hf_v2_v1") is None
    assert KbIndexNames("hf_v2").version_of("kb_hf_v2_v1") == 1

    prefixed = KbIndexNames("hf_docs", prefix="test1a2b_")
    assert prefixed.alias == "test1a2b_kb_hf_docs"
    assert prefixed.index(2) == "test1a2b_kb_hf_docs_v2"
    assert prefixed.version_of("test1a2b_kb_hf_docs_v2") == 2
    assert prefixed.version_of("kb_hf_docs_v2") is None

    with pytest.raises(ValueError, match="version"):
        names.index(0)
    for bad_kb in ("", "HF", "_x", "a b", "a*", "a,b"):
        with pytest.raises(ValueError, match="knowledge base"):
            KbIndexNames(bad_kb)
    for bad_prefix in ("Test_", "-x_", "a*", "_x"):
        with pytest.raises(ValueError, match="prefix"):
            KbIndexNames("hf_docs", prefix=bad_prefix)
