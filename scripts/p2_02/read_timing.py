"""Spec P2-02: time to read and clean the whole corpus, with no embedding and no Elasticsearch.

    HF_DATA_DIR=<dir with the downloaded dataset> uv run python scripts/p2_02/read_timing.py \
        [--kb hf_docs] [--repeats 3] [--with-chunk] [--out data/reports/p2-02/read_timing.json]

It does what the ingest job does before it compares with the docstore (`IngestJob.prepare`):
`connector.documents()` (list the pages), `KbProcessor.read` (reader) and `KbProcessor.clean` (the
cleaning rules); with `--with-chunk` also `KbProcessor.chunk` (the chunk steps, which need the
reranker's tokenizer from the local HF cache). Each repeat builds a new processor, because the
rules keep statistics. Design s19 P2: if reading and cleaning take more than about 10 minutes,
file-level skip (do not read a file whose size and mtime did not change) is worth adding.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

import rag.data  # noqa: F401  (registers the connectors named in knowledge-base YAML)
from rag.config import ServiceSettings, load_kb_configs
from rag.ingest.build import KbProcessor
from rag.registry import connectors


def one_pass(kb: str, *, with_chunk: bool) -> dict[str, float | int]:
    settings = ServiceSettings()
    cfg = load_kb_configs(settings.kb_dir)[kb]
    extra = dict(cfg.source.model_extra or {})
    extra["limit"] = None  # every page
    connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)
    processor = KbProcessor(cfg, settings=settings)
    if with_chunk:
        processor.tokenizer  # noqa: B018  (load the tokenizer outside the timed chunk step)

    t0 = time.perf_counter()
    sources = list(connector.documents())
    t1 = time.perf_counter()
    read = processor.read(sources)
    t2 = time.perf_counter()
    documents = processor.clean(read.documents)
    t3 = time.perf_counter()
    result: dict[str, float | int] = {
        "pages": len(sources),
        "read_errors": len(read.errors),
        "skipped_no_pipeline": sum(1 for why in read.skipped.values() if why == "no_pipeline"),
        "documents_read": len(read.documents),
        "documents_kept": len(documents),
        "list_s": round(t1 - t0, 2),
        "read_s": round(t2 - t1, 2),
        "clean_s": round(t3 - t2, 2),
        "read_and_clean_s": round(t3 - t1, 2),
        "total_s": round(t3 - t0, 2),
    }
    if with_chunk:
        t4 = time.perf_counter()
        nodes = processor.chunk(documents)
        result["chunk_s"] = round(time.perf_counter() - t4, 2)
        result["chunks"] = len(nodes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kb", default="hf_docs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-chunk", action="store_true")
    parser.add_argument("--out", default="data/reports/p2-02/read_timing.json")
    args = parser.parse_args()

    passes = []
    for i in range(args.repeats):
        load = psutil.cpu_percent(interval=1.0)
        result = one_pass(args.kb, with_chunk=args.with_chunk)
        result["cpu_percent_before"] = load
        passes.append(result)
        print(f"pass {i + 1}: {result}", flush=True)
    totals = [p["total_s"] for p in passes]
    summary = {
        "kb": args.kb,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {
            "platform": platform.platform(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "passes": passes,
        "total_s_min": min(totals),
        "total_s_median": statistics.median(totals),
        "total_s_max": max(totals),
        "over_ten_minutes": max(totals) > 600,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    median = summary["total_s_median"]
    print(f"wrote {out}; total min/median/max {min(totals)}/{median}/{max(totals)} s")


if __name__ == "__main__":
    main()
