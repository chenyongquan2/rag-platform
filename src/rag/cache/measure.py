"""The measurements of spec P5-01 against the local services (`python -m rag.cache bench`).

    overhead   cost of the key and of a Redis round trip (what a miss pays for the lookup)
    semantic   lookup time of the two stores of the semantic cache, memory and Elasticsearch
    replay     a Zipf-shaped workload through the real `search_docs` (real Elasticsearch, real
               reranker, the cache in Redis under the prefix `p501meas:`, shadow mode on)
    ask        a few real `ask_docs` calls twice (a miss with the model, then a hit)
    down       the same requests with the cache pointed at a port nobody listens on

Nothing here touches the real cache keys: the Redis prefix is `p501meas:` and deleted at the
end, the semantic store index is `p501meas_cache_semantic_*`, the query embeddings come from a
local SQLite file (`data/cache/cache_pairs.sqlite3`), so the shared L0 cache in Redis is not
written. The data generation of the knowledge base is only read.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from rag.cache.bench import (
    build_workload,
    read_cache_records,
    replay_search,
    summarize,
    top1_agreement,
)
from rag.cache.keys import cache_key, make_fields
from rag.cache.store import RedisCacheBackend
from rag.config import ServiceSettings, load_kb_config, repo_path
from rag.cost.ledger import CostLedger
from rag.factories import make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX
from rag.security.access import Credentials

PREFIX = "p501meas:"
ES_PREFIX = "p501meas_"
KB = "hf_docs"
CREDS = Credentials(transport="http")


def measure_settings(out: Path, **over: Any) -> ServiceSettings:
    """Settings for a measurement run: nothing exported, the records in `out`, the cache keys
    under `PREFIX`, query vectors from the local pair cache."""
    values: dict[str, Any] = {
        "otel_mode": "memory",
        "otel_auto_instrument": False,
        "log_dir": "",
        "log_stderr_level": "WARNING",
        "business_record_path": str(out / "business.jsonl"),
        "cache_mode": "l1_shadow",
        "cache_key_prefix": PREFIX,
        "cache_shadow_dir": str(out / "shadow"),
        "cache_semantic_store": "memory",
        "cache_semantic_threshold": 0.20,
        "embed_query_cache_url": "sqlite://data/cache/cache_pairs.sqlite3",
        "service_allowed_hosts": [],
    }
    values.update(over)
    return ServiceSettings(**values)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _timed(fn: Any, n: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(n):
        began = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - began) * 1000.0)
    return {
        "n": n,
        "median_ms": statistics.median(samples),
        "p95_ms": _pct(samples, 0.95),
        "max_ms": max(samples),
    }


def drop_measure_keys(settings: ServiceSettings) -> int:
    """Remove the keys of this run (only those under `PREFIX`) with SCAN."""
    client = make_redis_client(settings.redis_url, socket_timeout=2.0)
    backend = RedisCacheBackend(client, key_prefix=PREFIX)
    try:
        return backend.delete_prefix("")
    finally:
        client.close()


def measure_overhead(settings: ServiceSettings) -> dict[str, Any]:
    """What a miss pays for the cache besides the computation (milliseconds)."""
    client = make_redis_client(settings.redis_url, socket_timeout=settings.cache_timeout_ms / 1000)
    backend = RedisCacheBackend(client, key_prefix=PREFIX, generation_key_prefix=DEFAULT_KEY_PREFIX)
    question = "What is the default value of the allow_flagging parameter in Gradio's Interface?"

    def key() -> None:
        cache_key(
            make_fields(
                "search",
                kb=KB,
                question=question,
                acl_groups=["public"],
                top_k=8,
                data_generation=1,
                config_version="x" * 16,
            )
        )

    value = "x" * 6000  # about the size of a result of 8 chunks
    try:
        backend.set("bench:probe", value, ttl_s=60)
        out = {
            "key_build": _timed(key, 2000),
            "redis_generation": _timed(lambda: backend.generation(KB), 500),
            "redis_get_miss": _timed(lambda: backend.get("bench:absent"), 500),
            "redis_get_hit_6kb": _timed(lambda: backend.get("bench:probe"), 500),
            "redis_set_6kb": _timed(lambda: backend.set("bench:probe", value, ttl_s=60), 500),
            "timeout_ms": settings.cache_timeout_ms,
        }
    finally:
        backend.delete_prefix("bench:")
        client.close()
    return out


def measure_semantic_stores(settings: ServiceSettings, *, stored: int = 2000) -> dict[str, Any]:
    """Lookup time of the memory store and the Elasticsearch store, 1024-dimension vectors."""
    import numpy as np

    from rag.cache.semantic import (
        EsSemanticStore,
        MemorySemanticStore,
        Partition,
        StoredQuery,
    )
    from rag.factories import make_es_client

    rng = np.random.default_rng(7)
    dim = settings.embed_dim
    vectors = rng.normal(size=(stored, dim)).astype(np.float32)
    queries = rng.normal(size=(100, dim)).astype(np.float32)
    partition = Partition("bench", "f", "a", "cfg", 1, 8, "en")
    now = time.time()
    result: dict[str, Any] = {"stored": stored, "dim": dim}
    memory = MemorySemanticStore()
    for i in range(stored):
        memory.add(
            partition,
            vectors[i],
            StoredQuery(f"e{i}", f"q{i}", ("c",), ("d",), now),
            ttl_s=3600,
        )
    it = iter(range(10**9))
    result["memory"] = _timed(
        lambda: memory.search(partition, queries[next(it) % 100], k=3, now=now), 100
    )
    es = make_es_client(settings.es_url, request_timeout=30.0, max_retries=0)
    store = EsSemanticStore(es, dim=dim, index_prefix=ES_PREFIX, refresh=True, request_timeout_s=30)
    try:
        for i in range(stored):
            store.add(
                partition,
                vectors[i],
                StoredQuery(f"e{i}", f"q{i}", ("c",), ("d",), now),
                ttl_s=3600,
            )
        it2 = iter(range(10**9))
        result["elasticsearch"] = _timed(
            lambda: store.search(partition, queries[next(it2) % 100], k=3, now=now), 100
        )
        result["elasticsearch_one_add"] = _timed(
            lambda: store.add(
                partition, vectors[0], StoredQuery("x", "q", ("c",), ("d",), now), ttl_s=3600
            ),
            20,
        )
    finally:
        store.drop("bench")
        es.close()
    return result


def _real_service(settings: ServiceSettings, with_cache: bool = True) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from rag.serve.service import SearchService

    cfg = load_kb_config(repo_path("kb/public/hf_docs.yaml"))
    service = SearchService.from_settings(settings, configs={KB: cfg})
    if not with_cache:
        service._cache = None
    service.load()
    return service


def question_pool() -> list[tuple[str, str]]:
    """The public 65 questions, the synthetic dev ones and the 15 colloquial rewrites."""
    from rag.cache.cli import PAIR_SOURCES

    root = repo_path("evals/datasets/hf_docs")
    pool: list[tuple[str, str]] = []
    for name in [*PAIR_SOURCES["dev"], "colloquial.jsonl"]:
        with (root / name).open(encoding="utf-8", newline="") as fh:
            for line in fh:
                if line.strip():
                    data = json.loads(line)
                    pool.append((data["qid"], data["question"]))
    return pool


def measure_replay(settings: ServiceSettings, out: Path, *, n: int = 400) -> dict[str, Any]:
    from rag.observe import init_observability, shutdown_observability

    init_observability(settings, "rag-cache-bench", configure_logging=False)
    ledger = CostLedger.from_settings(settings)
    spent_before = ledger.total_spent()
    service = _real_service(settings)
    try:
        requests = build_workload(question_pool(), n, zipf_s=1.0, variant_share=0.15, seed=7)
        records = replay_search(service, CREDS, requests, kb=KB, deadline_ms=30_000)
        if service.cache is not None:
            service.cache.flush_shadow(60.0)  # the shadow lookups run beside the requests
    finally:
        service.close()
        shutdown_observability()
    summary = summarize(records, read_cache_records(settings.business_record_path))
    summary["top1_agreement"] = top1_agreement(records)
    summary["spent_cny"] = ledger.total_spent() - spent_before
    summary["workload"] = {"n": n, "zipf_s": 1.0, "variant_share": 0.15, "seed": 7}
    (out / "replay_records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return summary


def measure_ask(settings: ServiceSettings, *, questions: int = 5) -> dict[str, Any]:
    import asyncio

    from rag.observe import init_observability, shutdown_observability

    init_observability(settings, "rag-cache-bench", configure_logging=False)
    ledger = CostLedger.from_settings(settings)
    spent_before = ledger.total_spent()
    service = _real_service(settings)
    rows: list[dict[str, Any]] = []
    try:
        pool = [q for q in question_pool() if q[0].startswith("huggingface_doc_qa_eval")]
        for qid, question in pool[:questions]:
            for attempt in ("first", "second"):
                began = time.perf_counter()
                response = asyncio.run(
                    service.ask_docs(CREDS, kb=KB, question=question, deadline_ms=30_000)
                )
                rows.append(
                    {
                        "qid": qid,
                        "attempt": attempt,
                        "ms": (time.perf_counter() - began) * 1000.0,
                        "outcome": response.cache.outcome if response.cache else None,
                        "status": response.status,
                        "found": response.found,
                        "tokens": response.tokens_used,
                        "flags": response.error_categories,
                    }
                )
    finally:
        service.close()
        shutdown_observability()
    return {"rows": rows, "spent_cny": ledger.total_spent() - spent_before}


def measure_down(settings: ServiceSettings, *, requests: int = 20) -> dict[str, Any]:
    """Cache pointed at a closed port: same requests, and the same requests with no cache."""
    from rag.observe import init_observability, shutdown_observability

    init_observability(settings, "rag-cache-bench", configure_logging=False)
    pool = question_pool()[:requests]
    out: dict[str, Any] = {}
    try:
        for name, with_cache, port in (("no_cache", False, 26379), ("closed_port", True, 1)):
            run_settings = settings.model_copy(
                update={"cache_url": f"redis://127.0.0.1:{port}/0", "cache_mode": "l1"}
            )
            service = _real_service(run_settings, with_cache=with_cache)
            try:
                records = replay_search(
                    service,
                    CREDS,
                    build_workload(pool, requests, zipf_s=0.0, variant_share=0.0, seed=1),
                    kb=KB,
                    deadline_ms=30_000,
                )
            finally:
                service.close()
            ms = [r["ms"] for r in records]
            out[name] = {
                "requests": len(records),
                "median_ms": statistics.median(ms),
                "p95_ms": _pct(ms, 0.95),
                "outcomes": summarize(records)["outcomes"],
                "errors": sum(1 for r in records if r["error"]),
            }
    finally:
        shutdown_observability()
    return out


def run(step: str, out: Path, *, n: int = 400, questions: int = 5) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    settings = measure_settings(out)
    # `cache_pairs.sqlite3` is relative to the repo root: make the settings path absolute.
    settings = settings.model_copy(
        update={
            "embed_query_cache_url": "sqlite://"
            + str(repo_path("data/cache/cache_pairs.sqlite3")).replace("\\", "/")
        }
    )
    try:
        if step == "overhead":
            result = measure_overhead(settings)
        elif step == "semantic":
            result = measure_semantic_stores(settings)
        elif step == "replay":
            result = measure_replay(settings, out, n=n)
        elif step == "ask":
            result = measure_ask(settings, questions=questions)
        elif step == "down":
            result = measure_down(settings)
        else:
            raise ValueError(f"unknown step {step!r}")
    finally:
        if step in ("replay", "ask", "down"):
            dropped = drop_measure_keys(settings)
            print(f"dropped {dropped} keys under {PREFIX}")
    (out / f"bench_{step}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
