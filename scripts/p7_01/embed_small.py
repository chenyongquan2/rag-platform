"""Spec P7-01 (3): a SMALL-INDEX approximation of switching the embedding model. No index rebuild.

The real index (26,372 chunks) is not touched and nothing is written to Elasticsearch. The
comparison runs in memory with exact cosine search (numpy) over a small corpus:
  - every chunk of the gold pages of the 99 dev questions, and
  - a fixed random sample of 2,000 other chunks (seed 0) as distractors.
Fewer distractors make the task much easier than the real index, so only the DIFFERENCE between
the two models on the same small corpus means anything, and only as a rough direction.

Models: the current gateway model (qwen/text-embedding-v4): chunk vectors from the local ingest
cache, question vectors from the local file / Redis L0 cache (a miss goes to the gateway and the
ledger); and one local alternative (sentence-transformers, CPU, free), default
BAAI/bge-base-en-v1.5 with its query instruction. Texts are the same embed texts the ingest
pipeline sends (chunk text plus the `metadata.embed` whitelist, LlamaIndex MetadataMode.EMBED).

  uv run python scripts/p7_01/embed_small.py [--model BAAI/bge-base-en-v1.5] [--distractors 2000]
Output: data/reports/p7-01/embed_small.json and per-question rows (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from llama_index.core.schema import MetadataMode

import rag.data  # noqa: F401  (registers the connectors named in knowledge-base YAML)
from rag.cache.pair_embed import embed_pair_texts
from rag.config import ServiceSettings, load_kb_configs
from rag.embed.cache import SqliteEmbeddingCache, cache_model_key, text_sha256
from rag.eval.questions import load_question_set
from rag.eval.stats import mcnemar_exact, paired_permutation_test
from rag.factories import url_path
from rag.ingest.build import KbProcessor
from rag.registry import connectors

OUT = Path("data/reports/p7-01")
DEV = "evals/datasets/hf_docs/synthetic_dev.jsonl"
QUERY_INSTRUCTION = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def page_metrics(ranked_pages: list[str], gold: set[str], k: int = 8) -> dict[str, float]:
    pages = list(dict.fromkeys(ranked_pages))[:k]
    rank = next((i + 1 for i, p in enumerate(pages) if p in gold), None)
    return {
        "hit@1": float(rank == 1),
        "hit@8": float(rank is not None),
        "mrr@8": 1.0 / rank if rank else 0.0,
    }


def search(q: np.ndarray, docs: np.ndarray, doc_pages: list[str], top: int = 40) -> list[str]:
    scores = docs @ q
    idx = np.argsort(-scores)[:top]
    return [doc_pages[i] for i in idx]


def normed(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--distractors", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    settings = ServiceSettings()
    cfg = load_kb_configs(settings.kb_dir)["hf_docs"]
    extra = dict(cfg.source.model_extra or {})
    extra["limit"] = None
    connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)
    t0 = time.perf_counter()
    proc = KbProcessor(cfg, settings=settings)
    nodes = proc.chunk(proc.clean(proc.read(list(connector.documents())).documents))
    print(f"chunked {len(nodes)} chunks in {time.perf_counter() - t0:.0f} s", flush=True)

    questions = load_question_set(DEV, kb="hf_docs").questions
    gold_pages = {pg for q in questions for pg in q.relevant_pages}

    def page_of(node: object) -> str:
        meta = node.metadata  # type: ignore[attr-defined]
        return str(meta.get("doc_id") or meta.get("page_id") or node.ref_doc_id)  # type: ignore[attr-defined]

    pages = [page_of(n) for n in nodes]
    gold_idx = [i for i, pg in enumerate(pages) if pg in gold_pages]
    rest = [i for i, pg in enumerate(pages) if pg not in gold_pages]
    rng = random.Random(args.seed)
    distract = sorted(rng.sample(rest, min(args.distractors, len(rest))))
    chosen = gold_idx + distract
    texts = [nodes[i].get_content(metadata_mode=MetadataMode.EMBED) for i in chosen]
    doc_pages = [pages[i] for i in chosen]
    missing_gold = gold_pages - set(doc_pages)
    print(
        f"small corpus: {len(gold_idx)} gold-page chunks ({len(gold_pages)} pages, "
        f"{len(missing_gold)} not found) + {len(distract)} distractors = {len(chosen)}",
        flush=True,
    )

    # current model: chunk vectors from the ingest cache (no call), questions from the caches
    model_key = cache_model_key(
        settings.embed_model, settings.embed_dim, settings.embed_text_prefix
    )
    cache = SqliteEmbeddingCache(url_path(settings.embed_ingest_cache_url))
    hashes = [text_sha256(t) for t in texts]
    found = cache.get_many(model_key, list(dict.fromkeys(hashes)))
    cache.close()
    misses = sum(h not in found for h in hashes)
    if misses:
        raise SystemExit(f"{misses} chunk texts are not in the ingest cache: text building differs")
    cur_docs = normed(np.array([found[h] for h in hashes], dtype=np.float32))
    qvecs, counts = embed_pair_texts(
        [q.question for q in questions], settings, cache_path=OUT / "p701_query_vectors.sqlite3"
    )
    print(f"current-model question vectors: {counts}", flush=True)
    cur_q = normed(np.array([qvecs[q.question] for q in questions], dtype=np.float32))

    # alternative local model
    from sentence_transformers import SentenceTransformer

    t1 = time.perf_counter()
    st = SentenceTransformer(args.model, device="cpu", local_files_only=True)
    load_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    alt_docs = st.encode(texts, batch_size=args.batch, normalize_embeddings=True)
    doc_s = time.perf_counter() - t2
    prefix = QUERY_INSTRUCTION.get(args.model, "")
    t3 = time.perf_counter()
    alt_q = st.encode([prefix + q.question for q in questions], normalize_embeddings=True)
    q_ms = (time.perf_counter() - t3) * 1000 / len(questions)
    print(
        f"{args.model}: load {load_s:.0f} s, {len(texts)} chunks in {doc_s:.0f} s "
        f"({len(texts) / doc_s:.1f} chunks/s), {q_ms:.0f} ms per question",
        flush=True,
    )

    rows = []
    for i, q in enumerate(questions):
        gold = set(q.relevant_pages)
        a = page_metrics(search(cur_q[i], cur_docs, doc_pages), gold)
        b = page_metrics(search(alt_q[i], np.asarray(alt_docs), doc_pages), gold)
        rows.append({"qid": q.qid, "current": a, "alternative": b})

    summary: dict[str, object] = {
        "label": "SMALL-INDEX APPROXIMATION (gold-page chunks + random distractors), vector only",
        "current_model": settings.embed_model,
        "alternative_model": args.model,
        "questions": len(questions),
        "corpus_chunks": len(chosen),
        "gold_chunks": len(gold_idx),
        "distractors": len(distract),
        "seed": args.seed,
        "query_vector_sources": counts,
        "alt_load_s": round(load_s, 1),
        "alt_chunks_per_s": round(len(texts) / doc_s, 2),
        "alt_query_ms": round(q_ms, 1),
        "alt_full_corpus_hours_est": round(len(nodes) / (len(texts) / doc_s) / 3600, 2),
        "metrics": {},
    }
    for m in ("hit@1", "hit@8", "mrr@8"):
        a = [r["current"][m] for r in rows]
        b = [r["alternative"][m] for r in rows]
        entry: dict[str, object] = {"current": float(np.mean(a)), "alternative": float(np.mean(b))}
        if m.startswith("hit"):
            better = sum(y > x for x, y in zip(a, b, strict=True))
            worse = sum(x > y for x, y in zip(a, b, strict=True))
            entry |= {"better": better, "worse": worse, "p_mcnemar": mcnemar_exact(better, worse)}
        else:
            res = paired_permutation_test(a, b, iterations=10_000, seed=0)
            entry |= {"mean_diff": res.mean_diff, "p_permutation": res.p_value}
        summary["metrics"][m] = entry  # type: ignore[index]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "embed_small.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (OUT / "embed_small_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
