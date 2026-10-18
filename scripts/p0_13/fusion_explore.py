# ruff: noqa  -- one-off P0-13 measurement script, kept as it was run (not part of the package)
"""Spec P0-13: fusion variants on the 65 HF questions, outside the `run` CLI.

uv run python scripts/p0_13/fusion_explore.py data/reports/p0-13/explore

Query vectors only from the L0 cache (a cache miss stops the script: no paid call). The vector
list (top 50) is fetched once; each variant sends its own BM25 query to ES, fuses with RRF
(optional weights and BM25 depth cap), dedupes by body like `TextDedupe`, and reranks the first
10 with the local cross-encoder (scores cached per question and chunk in the output dir).
Writes `summary.md` (one row per variant) and `variants.json` (per question: BM25, fused and
reranked top 10 doc ids). Variants marked "scratch" in docs/decisions.md came from here.
"""

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

from rag.config import ServiceSettings
from rag.config.kb import load_kb_config
from rag.eval.questions import load_question_set
from rag.retrieve.dedupe import dedupe_key
from rag.retrieve.filters import build_es_filter
from rag.retrieve.rerank import rerank_text
from rag.retrieve.search import hybrid_parts

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
settings = ServiceSettings()
cfg = load_kb_config("kb/public/hf_docs.yaml", kb_root="kb")
qs = load_question_set(
    "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl", kb="hf_docs"
).questions
retr, _, es, _ = hybrid_parts(cfg, settings)
flt = build_es_filter(kb="hf_docs", acl_groups=settings.service_acl_groups)
STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "where",
        "when",
        "why",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "i",
        "you",
        "your",
        "we",
        "our",
    ]
)

vec_cache = OUT / "vec_lists.json"
if vec_cache.exists():
    VEC = json.loads(vec_cache.read_text())
else:
    VEC = {}
    for q in qs:
        v = retr.embed_model.cached_query_embedding(q.question)
        assert v is not None, f"cache miss {q.qid}: refusing a paid call"
        body = retr.knn_body(list(v), flt)
        body["size"] = 50
        body["knn"]["k"] = 50
        body["knn"]["num_candidates"] = 100
        r = es.search(index=retr.index, body=body)
        VEC[q.qid] = [(h["_id"], h["_source"]["doc_id"]) for h in r["hits"]["hits"]]
    vec_cache.write_text(json.dumps(VEC))

SRC = {}


def bm25(q, *, boost=2.0, typ="best_fields", stop=False, size=50):
    text = q
    if stop:
        toks = re.findall(r"\S+", q)
        kept = [t for t in toks if re.sub(r"\W", "", t).lower() not in STOP]
        text = " ".join(kept) or q
    mm = {"query": text, "type": typ}
    mm["fields"] = ["content"] if boost == 0 else ["content", f"title_path^{boost:g}"]
    body = {
        "query": {"bool": {"must": [{"multi_match": mm}], "filter": flt}},
        "size": size,
        "_source": {"includes": ["doc_id", "content", "title_path", "source_rel", "source_url"]},
    }
    r = es.search(index=retr.index, body=body)
    out = []
    for h in r["hits"]["hits"]:
        SRC[h["_id"]] = h["_source"]
        out.append((h["_id"], h["_source"]["doc_id"]))
    return out


def fetch(ids):
    miss = [i for i in ids if i not in SRC]
    if miss:
        r = es.mget(
            index=retr.index,
            ids=miss,
            source_includes=["doc_id", "content", "title_path", "source_rel", "source_url"],
        )
        for d in r["docs"]:
            SRC[d["_id"]] = d["_source"]


def fuse(routes, weights, k=60, depth=None):
    sc, best = {}, {}
    doc = {}
    for name, lst in routes.items():
        lim = lst[: depth.get(name, 20)] if depth else lst[:20]
        for rank, (cid, did) in enumerate(lim, 1):
            sc[cid] = sc.get(cid, Fraction(0)) + Fraction(weights[name]).limit_denominator(1000) / (
                k + rank
            )
            best[cid] = min(best.get(cid, 99), rank)
            doc[cid] = did
    order = sorted(sc, key=lambda c: (-sc[c], best[c], c))
    fetch(order)
    seen, out = set(), []
    for c in order:
        s = SRC[c]
        key = dedupe_key(
            s.get("content", ""), [x for x in (s.get("source_rel"), s.get("source_url")) if x]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((c, doc[c]))
    return out


rcache_p = OUT / "rerank_cache.json"
RC = json.loads(rcache_p.read_text()) if rcache_p.exists() else {}
scorer = None


def rerank(q, cands):
    global scorer
    need = [c for c, _ in cands if f"{q.qid}|{c}" not in RC]
    if need:
        if scorer is None:
            from rag.retrieve.rerank import load_cross_encoder

            scorer = load_cross_encoder(settings)
        fetch(need)
        pairs = [
            (q.question, rerank_text(SRC[c].get("title_path", ""), SRC[c].get("content", "")))
            for c in need
        ]
        for c, s in zip(need, scorer.predict(pairs)):
            RC[f"{q.qid}|{c}"] = float(s)
        rcache_p.write_text(json.dumps(RC))
    order = sorted(range(len(cands)), key=lambda i: (-RC[f"{q.qid}|{cands[i][0]}"], i))
    return [cands[i] for i in order]


def metrics(lst, rel, k=8):
    pages = [d for _, d in lst]
    r = next((i for i, p in enumerate(pages[:10], 1) if p in rel), None)
    return {
        "hit1": float(r == 1),
        "hit5": float(r is not None and r <= 5),
        "hit10": float(r is not None),
        "mrr": 1 / r if r else 0.0,
    }


VARIANTS = {  # every variant tried in P0-13
    "V0 baseline (title^2 best_fields, w 1:1)": dict(),
    "W2 vector weight 2": dict(w={"bm25": 1, "vector": 2}),
    "W3 vector weight 3": dict(w={"bm25": 1, "vector": 3}),
    "W1.5 vector weight 1.5": dict(w={"bm25": 1, "vector": 1.5}),
    "D5 BM25 depth 5": dict(depth={"bm25": 5, "vector": 20}),
    "D10 BM25 depth 10": dict(depth={"bm25": 10, "vector": 20}),
    "T1 title^1": dict(boost=1.0),
    "T0.5 title^0.5": dict(boost=0.5),
    "T0 content only": dict(boost=0),
    "MF most_fields title^1": dict(boost=1.0, typ="most_fields"),
    "SW stopwords removed, title^2": dict(stop=True),
    "SW+T1 stopwords removed, title^1": dict(stop=True, boost=1.0),
    "SW+T1+W2": dict(stop=True, boost=1.0, w={"bm25": 1, "vector": 2}),
    "VEC vector only": dict(w={"bm25": 0, "vector": 1}),
    "T1+W1.5": dict(boost=1.0, w={"bm25": 1, "vector": 1.5}),
    "T1+W2": dict(boost=1.0, w={"bm25": 1, "vector": 2}),
    "T1+D10": dict(boost=1.0, depth={"bm25": 10, "vector": 20}),
}
rows, perq = [], {}
for name, v in VARIANTS.items():
    agg = {"pre": {}, "post": {}, "bm25": {}}
    perq[name] = {}
    for q in qs:
        rel = set(q.relevant_pages)
        b = bm25(
            q.question,
            boost=v.get("boost", 2.0),
            typ=v.get("typ", "best_fields"),
            stop=v.get("stop", False),
        )
        w = v.get("w", {"bm25": 1, "vector": 1})
        routes = {"vector": [tuple(x) for x in VEC[q.qid]]}
        if w["bm25"]:
            routes["bm25"] = b
        fused = fuse(routes, w, depth=v.get("depth"))
        post = rerank(q, fused[:10]) + fused[10:]
        for key, lst in (("pre", fused), ("post", post), ("bm25", b)):
            m = metrics(lst, rel)
            for mk, mv in m.items():
                agg[key].setdefault(mk, []).append(mv)
        perq[name][q.qid] = {
            "bm25": [d for _, d in b[:10]],
            "fused": [d for _, d in fused[:10]],
            "reranked": [d for _, d in post[:10]],
            "relevant": sorted(rel),
            "pre": metrics(fused, rel),
            "post": metrics(post, rel),
        }
    mean = lambda xs: sum(xs) / len(xs)
    rows.append(
        (
            name,
            mean(agg["bm25"]["hit1"]),
            mean(agg["bm25"]["mrr"]),
            mean(agg["pre"]["hit1"]),
            mean(agg["pre"]["mrr"]),
            mean(agg["pre"]["hit10"]),
            mean(agg["post"]["hit1"]),
            mean(agg["post"]["mrr"]),
            mean(agg["post"]["hit5"]),
        )
    )
lines = [
    "| variant | BM25 Hit@1 | BM25 MRR@10 | pre Hit@1 | pre MRR@10 | pre Hit@10 | post Hit@1 | post MRR@10 | post Hit@5 |",
    "|---" * 9 + "|",
]
lines += ["| " + r[0] + " | " + " | ".join(f"{x:.3f}" for x in r[1:]) + " |" for r in rows]
text = "\n".join(lines) + "\n"
(OUT / "summary.md").write_text(text, encoding="utf-8")
(OUT / "variants.json").write_text(json.dumps(perq, indent=1), encoding="utf-8")
print(text, end="")
