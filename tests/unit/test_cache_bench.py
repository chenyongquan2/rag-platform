"""Spec P5-01: the replay workload and its summary (the measurement tool; the measurement itself
is in docs/reports/P5-report.md)."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rag.cache.bench import (
    build_workload,
    read_cache_records,
    replay_search,
    summarize,
    top1_agreement,
)
from rag.cache.exact import CacheConfig, ExactCache
from rag.cache.keys import canonical_question
from rag.cache.layer import ResultCache
from rag.cache.store import MemoryCacheBackend
from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.security.access import Credentials
from rag.serve.service import SearchService
from rag.testing import memory_observability, observe_settings
from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

POOL = [
    (f"q:{i}", f"What is the purpose of thing number {chr(97 + i % 26)}{i}?") for i in range(60)
]


def test_the_workload_is_deterministic_and_zipf_shaped() -> None:
    a = build_workload(POOL, 600, seed=3)
    b = build_workload(POOL, 600, seed=3)
    assert a == b
    assert build_workload(POOL, 600, seed=4) != a
    counts = Counter(r.qid for r in a).most_common()
    assert counts[0][1] > 4 * counts[len(counts) // 2][1]  # a few questions asked often
    assert len(counts) < len(POOL)  # and a tail never asked


def test_variants_are_only_repeats_and_are_merged_by_the_key() -> None:
    requests = build_workload(POOL, 800, variant_share=0.5, seed=1)
    seen: set[str] = set()
    variants = 0
    originals = dict(POOL)
    for r in requests:
        if r.variant != "exact":
            variants += 1
            assert r.qid in seen  # never the first request of a question
            assert r.text != originals[r.qid]
            assert canonical_question(r.text) == canonical_question(originals[r.qid])
        seen.add(r.qid)
    assert variants > 50
    assert not [r for r in build_workload(POOL, 100, variant_share=0.0) if r.variant != "exact"]


def _service(delay_s: float) -> tuple[SearchService, FakeSearchPipeline]:
    pipeline = FakeSearchPipeline("hf_docs")
    original = pipeline.retrieve

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay_s)
        return original(*args, **kwargs)

    pipeline.retrieve = slow  # type: ignore[method-assign]
    settings = observe_settings()
    service = SearchService(
        settings,
        configs={
            "hf_docs": KnowledgeBaseConfig(
                kb="hf_docs", language="en", access=AccessConfig(callers=["*"])
            )
        },
        builder=FakePipelineBuilder({"hf_docs": pipeline}),
        cache=ResultCache(ExactCache(MemoryCacheBackend(), CacheConfig()), settings),
    )
    service.load()
    return service, pipeline


def test_replay_and_summary(tmp_path: Path) -> None:
    records_path = tmp_path / "business.jsonl"
    requests = build_workload(POOL[:20], 80, seed=5, variant_share=0.3)
    with memory_observability(business_record_path=str(records_path)):
        service, pipeline = _service(0.02)
        records = replay_search(service, Credentials(transport="http"), requests, kb="hf_docs")
    distinct = len({r.qid for r in requests})
    assert len(pipeline.calls) == distinct  # every distinct question computed once
    report = summarize(records, read_cache_records(records_path))
    assert report["requests"] == 80 and report["distinct_questions"] == distinct
    assert report["outcomes"] == {"miss": distinct, "hit": 80 - distinct}
    assert report["hit_rate"] == pytest.approx(report["best_possible_hit_rate"])
    miss, hit = report["latency_ms"]["miss"], report["latency_ms"]["hit"]
    assert miss["median"] > 15 and hit["median"] < miss["median"]
    assert report["time_saved_ms"]["total"] > 0
    assert report["wall_without_cache_ms"] > report["wall_ms"]
    assert report["lookup_ms_on_miss"]["n"] == distinct
    assert report["variants"]["hits"] == report["variants"]["requests"] > 0  # the normaliser works
    assert top1_agreement(records) == {"hits": 80 - distinct, "same_top1": 80 - distinct}
    assert report["errors"] == 0


def test_summary_of_nothing_is_not_an_error() -> None:
    report = summarize([])
    assert report["requests"] == 0 and report["hit_rate"] is None


def test_read_cache_records_ignores_other_lines(tmp_path: Path) -> None:
    path = tmp_path / "b.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "a", "extra": {"cache": {"lookup_ms": 1.5}}}),
                json.dumps({"request_id": "b", "extra": {}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert read_cache_records(path) == {"a": {"lookup_ms": 1.5}}
