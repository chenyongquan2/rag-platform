# ruff: noqa  -- one-off P0-13 measurement script, kept as it was run (not part of the package)
"""Spec P0-13: rerank score distributions for threshold calibration (design s10-2).
Usage: threshold_scores.py <out.json> <kb_dir>. Threshold 0 so every reranked candidate is kept."""

import json
import sys

from rag.config import ServiceSettings
from rag.config.kb import load_kb_config
from rag.eval.questions import load_question_set
from rag.factories import make_redis_client
from rag.retrieve.pipeline import make_search_pipeline

out, kb_dir = sys.argv[1], sys.argv[2]
settings = ServiceSettings()
settings = settings.model_copy(
    update={"search_deadline_ms": 60000, "search_max_deadline_ms": 60000}
)
cfg = load_kb_config(f"{kb_dir}/public/hf_docs.yaml", kb_root=kb_dir)
cfg = cfg.model_copy(
    update={"retrieval": cfg.retrieval.model_copy(update={"rerank_n": 10, "threshold": 0.0})}
)
pipe = make_search_pipeline(
    cfg,
    settings,
    acl_groups=settings.service_acl_groups,
    redis=make_redis_client(settings.redis_url),
)
rows = []
for path, kind in (
    ("evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl", "answerable"),
    ("evals/datasets/hf_docs/no_answer.jsonl", "no_answer"),
):
    for q in load_question_set(path).questions:
        r = pipe.retrieve(q.question, kb="hf_docs", top_k=8)
        assert not r.rerank_timeout, q.qid
        hits = [{"doc_id": h.doc_id, "score": h.score} for h in r.rerank]
        rel = set(q.relevant_pages)
        rows.append(
            {
                "qid": q.qid,
                "kind": kind,
                "max": max(h["score"] for h in hits),
                "correct": [h["score"] for h in hits if h["doc_id"] in rel],
                "others": [h["score"] for h in hits if h["doc_id"] not in rel],
                "top_correct": bool(hits) and max(hits, key=lambda h: h["score"])["doc_id"] in rel,
            }
        )
        print(q.qid, kind, round(rows[-1]["max"], 4), flush=True)
json.dump(rows, open(out, "w"), indent=1)
