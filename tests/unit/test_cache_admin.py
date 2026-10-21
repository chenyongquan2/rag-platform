"""Spec P5-01 AC2/AC6: dropping the cache of one knowledge base by prefix (admin helper and CLI)."""

from __future__ import annotations

import pytest

from rag.cache.cli import main
from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.keys import cache_key, make_fields
from rag.cache.layer import ResultCache
from rag.cache.store import CacheUnavailable, MemoryCacheBackend
from rag.testing import observe_settings


def _fill(backend: MemoryCacheBackend, prefix: str = "") -> dict[str, str]:
    keys: dict[str, str] = {}
    for kb, kind in (("hf_docs", "search"), ("hf_docs", "ask"), ("mt5", "search")):
        fields = make_fields(kind, kb=kb, question=f"q {kb} {kind}", index_prefix=prefix)  # type: ignore[arg-type]
        key = cache_key(fields)
        backend.set(key, "v", ttl_s=60)
        backend.set(key + ":lock", "tok", ttl_s=60)  # a flight lock of that key
        keys[f"{kb}:{kind}"] = key
    return keys


def test_drop_kb_removes_only_that_kbs_entries_and_locks() -> None:
    backend = MemoryCacheBackend()
    keys = _fill(backend)
    layer = ResultCache(ExactCache(backend, CacheConfig()), observe_settings())
    assert layer.drop_kb("hf_docs") == 4
    assert backend.get(keys["hf_docs:search"]) is None and backend.get(keys["hf_docs:ask"]) is None
    assert backend.get(keys["mt5:search"]) == "v"
    assert backend.get(keys["mt5:search"] + ":lock") == "tok"


def test_a_kb_name_that_is_a_prefix_of_another_does_not_widen_the_drop() -> None:
    backend = MemoryCacheBackend()
    for kb in ("hf", "hf_docs"):
        backend.set(cache_key(make_fields("search", kb=kb, question="q")), "v", ttl_s=60)
    layer = ResultCache(ExactCache(backend, CacheConfig()), observe_settings())
    assert layer.drop_kb("hf") == 1  # `hf:` is the prefix, not `hf`
    assert len(backend.live_keys()) == 1


def test_drop_respects_the_index_prefix_of_the_service() -> None:
    backend = MemoryCacheBackend()
    plain = _fill(backend)
    fake = _fill(backend, prefix="fake_")
    layer = ResultCache(
        ExactCache(backend, CacheConfig()), observe_settings(search_index_prefix="fake_")
    )
    assert layer.drop_kb("hf_docs") == 4
    assert backend.get(plain["hf_docs:search"]) == "v"
    assert backend.get(fake["hf_docs:search"]) is None


def test_the_cli_drops_a_kb_and_says_how_many(capsys: pytest.CaptureFixture[str]) -> None:
    backend = MemoryCacheBackend()
    _fill(backend)
    code = main(["drop", "--kb", "hf_docs"], settings=observe_settings(), backend=backend)
    assert code == 0
    assert "dropped 4 keys of knowledge base hf_docs" in capsys.readouterr().out
    assert len(backend.live_keys()) == 2


def test_the_cli_reports_a_store_that_is_down(capsys: pytest.CaptureFixture[str]) -> None:
    backend = MemoryCacheBackend()
    backend.fail_with = CacheUnavailable("ConnectionError")
    code = main(["drop", "--kb", "hf_docs"], settings=observe_settings(), backend=backend)
    assert code == 1
    assert "cache store unavailable" in capsys.readouterr().err


def test_the_cli_refuses_a_bad_kb_name(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["drop", "--kb", "Bad Name*"], settings=observe_settings(), backend=MemoryCacheBackend()
    )
    assert code == 2
    assert "invalid knowledge base name" in capsys.readouterr().err
