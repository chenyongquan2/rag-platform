"""Spec P2-03 (d): kNN timing, index size and neighbour overlap of one index, no gateway call.

    uv run python scripts/p2_03/knn_bench.py bench --alias p203bbq_kb_hf_docs --name bbq_hnsw \
        [--questions evals/datasets/hf_docs/synthetic_dev.jsonl] [--repeats 5] [--out FILE]
    uv run python scripts/p2_03/knn_bench.py overlap EXACT.json OTHER.json [--ks 5,10,20]

`bench` asks the index the kNN request the search chain sends (`EsHybridRetriever.knn_body`:
k 20, num_candidates 100, the KB and access filter inside the knn) for every question of the set
with the question vector read from the L0 query cache (a missing vector stops the script: it
never calls the gateway). The first pass over the questions is a warm-up and is not timed; the
timed passes are repeated `--repeats` times. It records wall time and Elasticsearch's `took` per
request, the top-20 chunk ids of every question, the index size and the size of the `embedding`
field (`_disk_usage`), and the segment count. `overlap` compares the top-k chunk ids of two such
files question by question: the share of the exact (flat) list's top k that the other index returns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from rag.config import ServiceSettings, load_kb_configs
from rag.eval.questions import load_question_set
from rag.factories import make_es_client
from rag.retrieve.filters import build_es_filter

DEV = "evals/datasets/hf_docs/synthetic_dev.jsonl"
VECTOR_FIELD = "embedding"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _query_vectors(settings: ServiceSettings, questions: list[str]) -> dict[str, list[float]]:
    from rag.embed import make_embedding

    model = make_embedding("query", settings)
    vectors: dict[str, list[float]] = {}
    for text in questions:
        cached = model.cached_query_embedding(text)  # type: ignore[attr-defined]
        if cached is None:
            raise SystemExit(
                f"no cached query vector for {text!r}; this script never calls the gateway"
            )
        vectors[text] = list(cached)
    return vectors


def bench(args: argparse.Namespace) -> int:
    settings = ServiceSettings()
    cfg = load_kb_configs(settings.kb_dir)["hf_docs"]
    retrieval = cfg.retrieval
    questions = load_question_set(Path(args.questions), kb="hf_docs").questions
    texts = [q.question for q in questions]
    vectors = _query_vectors(settings, texts)
    es_filter = build_es_filter(kb="hf_docs", acl_groups=settings.service_acl_groups, filters=None)
    es = make_es_client(settings.es_url, request_timeout=60.0)

    def body(vector: list[float]) -> dict[str, Any]:
        return {
            "knn": {
                "field": VECTOR_FIELD,
                "query_vector": vector,
                "k": retrieval.top_k_per_route,
                "num_candidates": retrieval.num_candidates,
                "filter": es_filter,
            },
            "size": retrieval.top_k_per_route,
            "_source": False,
            "track_total_hits": False,
        }

    index = args.alias
    docs = es.count(index=index)["count"]
    lists: dict[str, list[str]] = {}
    for text in texts:  # warm-up pass: also the lists the overlap is computed from
        resp = es.search(index=index, body=body(vectors[text]))
        lists[text] = [h["_id"] for h in resp["hits"]["hits"]]
    load_before = psutil.cpu_percent(interval=1.0)
    wall_ms: list[float] = []
    took_ms: list[float] = []
    for _ in range(args.repeats):
        for text in texts:
            t0 = time.perf_counter()
            resp = es.search(index=index, body=body(vectors[text]))
            wall_ms.append((time.perf_counter() - t0) * 1000.0)
            took_ms.append(float(resp["took"]))
    load_after = psutil.cpu_percent(interval=1.0)

    stats = es.indices.stats(index=index, metric=["store", "docs", "segments"])
    total = next(iter(stats["indices"].values()))["primaries"]
    usage = es.indices.disk_usage(index=index, run_expensive_tasks=True)
    index_usage = next(v for k, v in usage.items() if k != "_shards")
    vec = index_usage["fields"][VECTOR_FIELD]
    out = {
        "name": args.name,
        "alias": index,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "docs": docs,
        "questions": len(texts),
        "repeats": args.repeats,
        "k": retrieval.top_k_per_route,
        "num_candidates": retrieval.num_candidates,
        "store_bytes": total["store"]["size_in_bytes"],
        "segments": total["segments"]["count"],
        "embedding_field_bytes": vec["total_in_bytes"],
        "embedding_field_parts": {k: v for k, v in vec.items() if k.endswith("_in_bytes")},
        "wall_ms": {
            "p50": statistics.median(wall_ms),
            "p95": _percentile(wall_ms, 0.95),
            "mean": statistics.fmean(wall_ms),
            "max": max(wall_ms),
        },
        "took_ms": {
            "p50": statistics.median(took_ms),
            "p95": _percentile(took_ms, 0.95),
            "mean": statistics.fmean(took_ms),
            "max": max(took_ms),
        },
        "cpu_percent_before_after": [load_before, load_after],
        "lists": lists,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"{args.name}: docs {docs}, store {out['store_bytes'] / 1e6:.1f} MB, embedding field "
        f"{out['embedding_field_bytes'] / 1e6:.1f} MB, segments {out['segments']}, wall P50 "
        f"{out['wall_ms']['p50']:.1f} ms P95 {out['wall_ms']['p95']:.1f} ms, took P50 "
        f"{out['took_ms']['p50']:.1f} ms; cpu {load_before:.0f}% -> {load_after:.0f}%"
    )
    print(f"wrote {path}")
    return 0


def overlap(args: argparse.Namespace) -> int:
    exact = json.loads(Path(args.exact).read_text(encoding="utf-8"))
    other = json.loads(Path(args.other).read_text(encoding="utf-8"))
    ks = [int(k) for k in args.ks.split(",")]
    print(
        f"{other['name']} against {exact['name']} (exact): share of the exact top k also returned"
    )
    for k in ks:
        shares = []
        for text, ids in exact["lists"].items():
            truth = set(ids[:k])
            if truth:
                shares.append(len(truth & set(other["lists"][text][:k])) / len(truth))
        print(
            f"  top {k}: mean {statistics.fmean(shares):.4f}, min {min(shares):.2f}, "
            f"questions with a different set {sum(1 for s in shares if s < 1)} of {len(shares)}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bench")
    p.add_argument("--alias", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--questions", default=DEV)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("overlap")
    p.add_argument("exact")
    p.add_argument("other")
    p.add_argument("--ks", default="5,10,20")
    args = parser.parse_args()
    return bench(args) if args.command == "bench" else overlap(args)


if __name__ == "__main__":
    sys.exit(main())
