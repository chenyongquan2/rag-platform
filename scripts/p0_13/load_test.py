# ruff: noqa  -- one-off P0-13 measurement script, kept as it was run (not part of the package)
"""Spec P0-13: whole-chain search latency under 1, 2, 4 concurrent requests (rerank_n 10).
Usage: load_test.py <out.json> <kb_dir> ; torch threads and slots from env (one config per process)."""

import json
import statistics
import sys
import threading
import time

import psutil

from rag.config import ServiceSettings
from rag.config.kb import load_kb_config
from rag.eval.questions import load_question_set
from rag.factories import make_redis_client
from rag.retrieve.pipeline import make_search_pipeline

out, kb_dir = sys.argv[1], sys.argv[2]
settings = ServiceSettings()
cfg = load_kb_config(f"{kb_dir}/public/hf_docs.yaml", kb_root=kb_dir)
cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"rerank_n": 10})})
qs = [
    q.question
    for q in load_question_set("evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl").questions
]
pipe = make_search_pipeline(
    cfg,
    settings,
    acl_groups=settings.service_acl_groups,
    redis=make_redis_client(settings.redis_url),
)
pipe.retrieve(qs[0], kb="hf_docs", top_k=8)  # warm
me = psutil.Process()
res = {
    "threads": settings.reranker_torch_threads,
    "slots": settings.rerank_max_concurrency,
    "runs": {},
}
PER_CLIENT = {1: 24, 2: 16, 4: 10}
for conc in (1, 2, 4):
    psutil.cpu_percent(None)
    time.sleep(3)
    base = psutil.cpu_percent(None)  # our process idle
    lat, timeouts, lock, idx = [], 0, threading.Lock(), [0]

    def client(cid):
        global timeouts
        for i in range(PER_CLIENT[conc]):
            q = qs[(cid * 17 + i) % len(qs)]
            t0 = time.perf_counter()
            r = pipe.retrieve(q, kb="hf_docs", top_k=8)
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt)
                timeouts += int(bool(r.rerank_timeout))

    psutil.cpu_percent(None)
    ths = [threading.Thread(target=client, args=(c,)) for c in range(conc)]
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    during = psutil.cpu_percent(None)
    lat.sort()
    p = lambda f: lat[min(len(lat) - 1, int(round(f * (len(lat) - 1))))]
    res["runs"][conc] = {
        "n": len(lat),
        "p50_s": statistics.median(lat),
        "p95_s": p(0.95),
        "max_s": lat[-1],
        "rerank_timeouts": timeouts,
        "cpu_before_pct": base,
        "cpu_during_pct": during,
        "throughput_qps": len(lat) / wall,
    }
    print(conc, res["runs"][conc], flush=True)
json.dump(res, open(out, "w"), indent=1)
