"""Measure the exact cache on a replayed workload (design s12-4; spec P5-01).

    requests = build_workload(pool, n=400, zipf_s=1.0, variant_share=0.15, seed=7)
    records = replay_search(service, creds, requests)
    report = summarize(records)

The workload: `n` requests drawn from a pool of questions; the chance of the question of rank r
(the pool in a seeded random order) is proportional to `1 / r ** zipf_s`, the shape real question
logs have (a few questions asked often, a long tail asked once). `variant_share` of the requests
write the question another way the key normaliser is meant to merge (case, spaces, closing
mark), so the run also shows whether the normaliser works on real traffic. The requests run one
after another: this measures latency, not throughput.

Time saved by the cache = for every hit, the time its question took when it was computed (its
own first request) minus the hit's time. A request is a hit, a miss (computed, written), or
neither (`bypass`, `error`; none are expected here).
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.cache.keys import canonical_question
from rag.cache.pairs import POSITIVE_VARIANTS
from rag.security.access import Credentials
from rag.serve.service import SearchService


@dataclass(frozen=True)
class Request:
    index: int
    qid: str
    text: str
    variant: str
    """`exact` (the question as it is in the set) or the name of the surface variant."""


def build_workload(
    pool: Sequence[tuple[str, str]],
    n: int,
    *,
    zipf_s: float = 1.0,
    variant_share: float = 0.15,
    seed: int = 7,
) -> list[Request]:
    """`n` requests over `pool` = [(qid, question)]; deterministic for a given seed."""
    rng = random.Random(seed)
    ranked = list(pool)
    rng.shuffle(ranked)
    weights = [1.0 / (rank + 1) ** zipf_s for rank in range(len(ranked))]
    picks = rng.choices(range(len(ranked)), weights=weights, k=n)
    variants = list(POSITIVE_VARIANTS)
    out: list[Request] = []
    seen: set[int] = set()
    for i, pick in enumerate(picks):
        qid, question = ranked[pick]
        text, kind = question, "exact"
        # Only a repeat is written another way (the first request of a question is as it is).
        if pick in seen and rng.random() < variant_share:
            name = variants[rng.randrange(len(variants))]
            variant = POSITIVE_VARIANTS[name](question)
            if variant is not None and canonical_question(variant) == canonical_question(question):
                text, kind = variant, name
        seen.add(pick)
        out.append(Request(i, qid, text, kind))
    return out


def replay_search(
    service: SearchService,
    credentials: Credentials,
    requests: Sequence[Request],
    *,
    kb: str,
    deadline_ms: int | None = None,
    pause_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Run the requests in order; one dict per request: the client-side time, the cache outcome
    and what the response said."""
    records: list[dict[str, Any]] = []
    for request in requests:
        request_id = f"bench-{request.index:05d}"
        began = time.perf_counter()
        error = None
        outcome, top1, flags = "failed", None, []
        try:
            response = service.search_docs(
                credentials,
                kb=kb,
                query=request.text,
                request_id=request_id,
                deadline_ms=deadline_ms,
            )
            outcome = response.cache.outcome if response.cache is not None else "no_cache"
            top1 = response.chunks[0].chunk_id if response.chunks else None
            flags = list(response.error_categories)
        except Exception as exc:  # a failed request is a data point, not the end of the run
            error = type(exc).__name__
        records.append(
            {
                "request_id": request_id,
                "index": request.index,
                "qid": request.qid,
                "variant": request.variant,
                "ms": (time.perf_counter() - began) * 1000.0,
                "outcome": outcome,
                "top1": top1,
                "flags": flags,
                "error": error,
            }
        )
        if pause_s:
            time.sleep(pause_s)
    return records


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def summarize(
    records: Sequence[Mapping[str, Any]],
    cache_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hit rate, latency of hits and misses, time saved, and the cost of the lookup on a miss.
    `cache_records`: the `cache` entry of the business record of each request id (`lookup_ms`,
    `store_ms`)."""
    cache_records = cache_records or {}
    outcomes = Counter(str(r["outcome"]) for r in records)
    by_outcome: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_outcome[str(r["outcome"])].append(float(r["ms"]))
    first_time: dict[str, float] = {}
    for r in records:
        if r["outcome"] == "miss" and r["qid"] not in first_time:
            first_time[str(r["qid"])] = float(r["ms"])
    saved = [
        first_time[str(r["qid"])] - float(r["ms"])
        for r in records
        if r["outcome"] in ("hit", "shared") and str(r["qid"]) in first_time
    ]
    total = len(records)
    hits = outcomes.get("hit", 0) + outcomes.get("shared", 0)
    distinct = len({r["qid"] for r in records})
    lookups = [
        float(cache_records[r["request_id"]]["lookup_ms"])
        for r in records
        if r["outcome"] == "miss" and r["request_id"] in cache_records
    ]
    stores = [
        float(cache_records[r["request_id"]].get("store_ms") or 0.0)
        for r in records
        if r["outcome"] == "miss" and r["request_id"] in cache_records
    ]
    wall = sum(float(r["ms"]) for r in records)
    variants = [r for r in records if r["variant"] != "exact"]
    return {
        "requests": total,
        "distinct_questions": distinct,
        "outcomes": dict(outcomes),
        "hit_rate": hits / total if total else None,
        "best_possible_hit_rate": (total - distinct) / total if total else None,
        "latency_ms": {k: _stats(v) for k, v in sorted(by_outcome.items())},
        "time_saved_ms": {
            "total": sum(saved),
            "per_hit": _stats(saved),
            "hits_counted": len(saved),
        },
        "wall_ms": wall,
        "wall_without_cache_ms": wall + sum(saved),
        "lookup_ms_on_miss": _stats(lookups),
        "store_ms_on_miss": _stats(stores),
        "lookup_share_of_miss_time": (
            statistics.median(lookups) / statistics.median(by_outcome["miss"])
            if lookups and by_outcome.get("miss")
            else None
        ),
        "variants": {
            "requests": len(variants),
            "hits": sum(r["outcome"] in ("hit", "shared") for r in variants),
        },
        "errors": sum(1 for r in records if r["error"]),
    }


def top1_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """For every hit: is the top chunk it returned the one the question's first (computed)
    request returned? The cache must hand back what the pipeline gave."""
    first: dict[str, str | None] = {}
    same = total = 0
    for r in records:
        if r["outcome"] == "miss" and r["qid"] not in first:
            first[str(r["qid"])] = r["top1"]
    for r in records:
        if r["outcome"] in ("hit", "shared") and str(r["qid"]) in first:
            total += 1
            same += r["top1"] == first[str(r["qid"])]
    return {"hits": total, "same_top1": same}


def read_cache_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """`extra.cache` of every business record in `path`, by request id."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            cache = (data.get("extra") or {}).get("cache")
            if cache is not None:
                out[str(data["request_id"])] = cache
    return out
