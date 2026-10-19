"""Spec P2-03: how many chunk texts a different chunk size would have to embed, and what it costs.

    uv run python scripts/p2_03/estimate_chunk_cost.py --sizes 400,300,200 \
        [--kb hf_docs] [--out data/reports/p2-03/chunk_cost_estimate.json]

No Elasticsearch, no gateway call. The corpus is read and cleaned once; for every size the whole
corpus is chunked with `token_split.max_tokens` set to that size (the live setting is 400), the
text each chunk would send to the embedding model (the chunk text and its metadata whitelist,
LlamaIndex `MetadataMode.EMBED`) is looked up in the local embedding cache, and only the texts
that are not there are counted as new embedding work. Size 400 is the control: it must find every
text in the cache (0 new texts), otherwise the way this script builds the text is wrong.

The price per token is read from the P0-12 ledger lines of the same model (cny / tokens); the
number of tokens the gateway bills for a text is estimated from its characters with the ratio
measured on the 400-token chunks of P0-12 (billed tokens of the ingest run / characters).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.schema import MetadataMode

import rag.data  # noqa: F401  (registers the connectors named in knowledge-base YAML)
from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_configs
from rag.embed.cache import SqliteEmbeddingCache, cache_model_key, text_sha256
from rag.factories import url_path
from rag.ingest.build import KbProcessor
from rag.registry import connectors


def with_chunk_size(cfg: KnowledgeBaseConfig, max_tokens: int) -> KnowledgeBaseConfig:
    """`cfg` with every `token_split` step set to `max_tokens` (nothing else changes)."""
    pipelines = {}
    for ext, pipe in cfg.pipelines.items():
        steps = []
        for step in pipe.steps:
            if step.name == "token_split":
                step = step.model_copy(update={"params": {**step.params, "max_tokens": max_tokens}})
            steps.append(step)
        pipelines[ext] = pipe.model_copy(update={"steps": steps})
    return cfg.model_copy(update={"pipelines": pipelines})


def ledger_price(ledger_path: Path, model: str) -> tuple[float, int, float]:
    """(cny per token, tokens, cny) summed over the ingest-profile embedding lines of `model`."""
    tokens = 0
    cny = 0.0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") != "embedding" or rec.get("model") != model:
            continue
        if rec.get("profile") != "ingest":
            continue
        tokens += int(rec.get("total_tokens", 0) or 0)
        cny += float(rec.get("cny", 0.0) or 0.0)
    return (cny / tokens if tokens else 0.0), tokens, cny


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kb", default="hf_docs")
    parser.add_argument("--sizes", default="400,300,200")
    parser.add_argument("--out", default="data/reports/p2-03/chunk_cost_estimate.json")
    parser.add_argument("--margin", type=float, default=1.15, help="safety factor on the tokens")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    settings = ServiceSettings()
    base_cfg = load_kb_configs(settings.kb_dir)[args.kb]
    extra = dict(base_cfg.source.model_extra or {})
    extra["limit"] = None
    connector = connectors.create(
        base_cfg.source.connector, kb=base_cfg.kb, settings=settings, **extra
    )
    cache = SqliteEmbeddingCache(url_path(settings.embed_ingest_cache_url))
    model_key = cache_model_key(
        settings.embed_model, settings.embed_dim, settings.embed_text_prefix
    )

    t0 = time.perf_counter()
    processor = KbProcessor(base_cfg, settings=settings)
    sources = list(connector.documents())
    documents = processor.clean(processor.read(sources).documents)
    print(
        f"read and cleaned {len(documents)} pages in {time.perf_counter() - t0:.0f} s", flush=True
    )

    results = {}
    for size in sizes:
        cfg = with_chunk_size(base_cfg, size)
        proc = KbProcessor(cfg, settings=settings)
        t1 = time.perf_counter()
        # a new processor re-reads nothing: it chunks the cleaned documents (same pipeline version
        # hash would change with the size, so re-tag the documents for this processor)
        version = proc.version_for(".md")
        docs = []
        for d in documents:
            d2 = d.model_copy(deep=False)
            d2.metadata = {**d.metadata, "pipeline_version": version}
            docs.append(d2)
        nodes = proc.chunk(docs)
        chunk_s = time.perf_counter() - t1
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        hashes = [text_sha256(t) for t in texts]
        unique = list(dict.fromkeys(hashes))
        found = cache.get_many(model_key, unique)
        text_by_hash = dict(zip(hashes, texts, strict=True))
        new = [h for h in unique if h not in found]
        results[size] = {
            "chunks": len(nodes),
            "unique_texts": len(unique),
            "cache_hits": len(unique) - len(new),
            "new_texts": len(new),
            "chars_all": sum(len(text_by_hash[h]) for h in unique),
            "chars_new": sum(len(text_by_hash[h]) for h in new),
            "chunk_s": round(chunk_s, 1),
        }
        print(size, results[size], flush=True)

    price, ledger_tokens, ledger_cny = ledger_price(
        Path(settings.cost_ledger_path), settings.embed_model
    )
    base = results.get(400)
    tokens_per_char = None
    if base is not None:
        # The ledger holds the P0-12 ingest (every chunk of the 400 setting) and a few thousand
        # query calls of a few tokens each: the ingest dominates, queries add well under 1%.
        tokens_per_char = ledger_tokens / base["chars_all"]
    for row in results.values():
        if tokens_per_char is not None:
            tokens = row["chars_new"] * tokens_per_char * args.margin
            row["estimated_tokens"] = round(tokens)
            row["estimated_cny"] = round(tokens * price, 3)
    out = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.embed_model,
        "price_cny_per_token": price,
        "ledger_tokens": ledger_tokens,
        "ledger_cny": round(ledger_cny, 6),
        "tokens_per_char": tokens_per_char,
        "margin": args.margin,
        "sizes": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
