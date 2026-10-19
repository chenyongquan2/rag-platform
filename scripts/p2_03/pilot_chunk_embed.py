"""Spec P2-03 (e): embed a small sample of the new texts of a chunk size, to measure the real price.

    uv run python scripts/p2_03/pilot_chunk_embed.py --size 200 --sample 1000 \
        [--max-cny 3.0] [--out data/reports/p2-03/pilot_c200.json]

The estimate of `estimate_chunk_cost.py` turns characters into tokens with the ratio measured on the
400-token chunks of P0-12. Chunks of another size may have another ratio (more title path per text
when chunks are short). This script chunks the corpus at `--size`, takes every n-th new text (new =
not in the L0 cache) in hash order, embeds the sample through the normal ingest embedding (cache and
cost ledger as in a real ingest, so the sample stays in the cache and is not paid for twice) and
reads the tokens and CNY the ledger recorded for exactly these calls. It then projects the cost of
all new texts from the sample's CNY per character and prints it next to the cap.

Before it spends anything it checks that the spent total plus the cost of a sample stays below the
budget stop line.
"""

from __future__ import annotations

import argparse
import json
import time
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


def _ledger_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kb", default="hf_docs")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--max-cny", type=float, default=3.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = ServiceSettings()
    ledger_path = Path(settings.cost_ledger_path)
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
    texts = {}
    for node in nodes:
        text = node.get_content(metadata_mode=MetadataMode.EMBED)
        texts.setdefault(text_sha256(text), text)
    found = cache.get_many(model_key, list(texts))
    new = sorted(h for h in texts if h not in found)
    chars_new = sum(len(texts[h]) for h in new)
    print(f"size {args.size}: {len(texts)} unique texts, {len(new)} new, {chars_new} characters")

    step = max(1, len(new) // args.sample)
    sample = new[::step][: args.sample]
    chars_sample = sum(len(texts[h]) for h in sample)
    spent_before = total_spent(settings)
    # a sample of 1000 texts costs about 0.07 CNY; refuse to run near the stop line
    if spent_before + 0.2 > 28.0:
        raise SystemExit(f"spent {spent_before:.3f} CNY: too close to the 28 CNY stop line")
    lines_before = len(_ledger_lines(ledger_path))
    model = make_embedding("ingest", settings, cache=cache)
    t0 = time.perf_counter()
    model.get_text_embedding_batch([texts[h] for h in sample])
    seconds = time.perf_counter() - t0
    added = [r for r in _ledger_lines(ledger_path)[lines_before:] if r.get("kind") == "embedding"]
    tokens = sum(int(r.get("total_tokens", 0) or 0) for r in added)
    cny = sum(float(r.get("cny", 0.0) or 0.0) for r in added)
    per_char = cny / chars_sample
    projected = per_char * chars_new  # the sample itself is already paid, so subtract it
    projected_rest = per_char * (chars_new - chars_sample)
    out = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "size": args.size,
        "unique_texts": len(texts),
        "new_texts": len(new),
        "chars_new": chars_new,
        "sample_texts": len(sample),
        "sample_chars": chars_sample,
        "sample_tokens": tokens,
        "sample_cny": round(cny, 6),
        "sample_seconds": round(seconds, 1),
        "tokens_per_char": tokens / chars_sample,
        "projected_total_cny": round(projected, 3),
        "projected_rest_cny": round(projected_rest, 3),
        "cap_cny": args.max_cny,
        "within_cap_with_5_percent_margin": projected * 1.05 <= args.max_cny,
    }
    path = Path(args.out or f"data/reports/p2-03/pilot_c{args.size}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
