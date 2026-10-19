"""Spec P2-03 (e): put the embeddings of a chunk size into the L0 cache, several requests at a time.

    .venv/Scripts/python.exe -u scripts/p2_03/prefill_chunk_cache.py --size 200 --workers 4 \
        --max-delta 2.85 [--out data/reports/p2-03/prefill_c200.json]

Why: one ingest sends its embedding requests one after the other (about 175 texts a minute while
this was measured), so a full second embedding of the corpus would take hours. This script chunks
the corpus at `--size`, takes the chunk texts that are not in the cache and embeds them with
`--workers` requests in flight (4 is the repo's own `embed_max_concurrency`), through the same
embedding model, cache and cost ledger as an ingest. Every request is written to the cache when it
returns, so an interrupted run loses nothing. A rebuild afterwards finds every text in the cache
(cost 0) and measures the rest of the work on its own.

It stops by itself when the ledger total has grown by more than `--max-delta` CNY since it started
(the hard cap of spec P2-03), and it refuses to start above the 28 CNY stop line.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.schema import MetadataMode

import rag.data  # noqa: F401  (registers the connectors named in knowledge-base YAML)
from rag.config import ServiceSettings, load_kb_configs
from rag.cost import total_spent
from rag.embed import make_embedding
from rag.embed.cache import SqliteEmbeddingCache, cache_model_key, text_sha256
from rag.factories import url_path
from rag.ingest.build import KbProcessor
from rag.registry import connectors


def _with_chunk_size(cfg, max_tokens: int):  # type: ignore[no-untyped-def]
    pipelines = {}
    for ext, pipe in cfg.pipelines.items():
        steps = []
        for step in pipe.steps:
            if step.name == "token_split":
                step = step.model_copy(update={"params": {**step.params, "max_tokens": max_tokens}})
            steps.append(step)
        pipelines[ext] = pipe.model_copy(update={"steps": steps})
    return cfg.model_copy(update={"pipelines": pipelines})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kb", default="hf_docs")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-delta", type=float, required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = ServiceSettings()
    start_spent = total_spent(settings)
    if start_spent + args.max_delta > 28.0:
        raise SystemExit(
            f"spent {start_spent:.3f} + {args.max_delta} would pass the 28 CNY stop line"
        )
    base_cfg = load_kb_configs(settings.kb_dir)[args.kb]
    cfg = _with_chunk_size(base_cfg, args.size)
    extra = dict(base_cfg.source.model_extra or {})
    extra["limit"] = None
    connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)
    cache = SqliteEmbeddingCache(url_path(settings.embed_ingest_cache_url))
    model_key = cache_model_key(
        settings.embed_model, settings.embed_dim, settings.embed_text_prefix
    )

    proc = KbProcessor(cfg, settings=settings)
    documents = proc.clean(proc.read(list(connector.documents())).documents)
    nodes = proc.chunk(documents)
    texts: dict[str, str] = {}
    for node in nodes:
        text = node.get_content(metadata_mode=MetadataMode.EMBED)
        texts.setdefault(text_sha256(text), text)
    found = cache.get_many(model_key, list(texts))
    new = sorted(h for h in texts if h not in found)
    batch_size = settings.embed_batch_size
    batches = [new[i : i + batch_size] for i in range(0, len(new), batch_size)]
    print(
        f"size {args.size}: {len(texts)} unique texts, {len(found)} cached, {len(new)} to embed "
        f"in {len(batches)} requests, {args.workers} at a time, spent so far {start_spent:.4f}",
        flush=True,
    )

    model = make_embedding("ingest", settings, cache=cache)
    stop = threading.Event()
    done = 0
    lock = threading.Lock()
    t0 = time.perf_counter()
    errors: list[str] = []

    def work(batch: list[str]) -> None:
        nonlocal done
        if stop.is_set():
            return
        if total_spent(settings) - start_spent > args.max_delta:
            stop.set()
            errors.append("cost cap reached")
            return
        try:
            model._get_text_embeddings([texts[h] for h in batch])  # writes the cache itself
        except Exception as exc:
            stop.set()
            errors.append(f"{type(exc).__name__}: {exc}")
            return
        with lock:
            done += 1
            if done % 100 == 0:
                spent = total_spent(settings) - start_spent
                rate = done * batch_size / (time.perf_counter() - t0) * 60
                print(
                    f"{done}/{len(batches)} requests, {rate:.0f} texts a minute, "
                    f"spent +{spent:.4f}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, batches))
    seconds = time.perf_counter() - t0
    spent = total_spent(settings) - start_spent
    out = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "size": args.size,
        "workers": args.workers,
        "unique_texts": len(texts),
        "to_embed": len(new),
        "requests_done": done,
        "requests_total": len(batches),
        "seconds": round(seconds, 1),
        "texts_per_minute": round(done * batch_size / seconds * 60) if seconds else 0,
        "spent_cny": round(spent, 6),
        "errors": errors,
        "complete": done == len(batches) and not errors,
    }
    path = Path(args.out or f"data/reports/p2-03/prefill_c{args.size}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1), flush=True)
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
