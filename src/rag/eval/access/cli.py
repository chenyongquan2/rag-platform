"""Run the permission isolation check against the real indexes (spec P8-01).

    uv run python -m rag.eval.access [--questions 3] [--out data/reports/p801-access]

What runs: the service as `python -m rag.serve http` would build it (the real Elasticsearch
indexes, read only; the reranker; the result cache), in OAuth mode with an in-process test
issuer and two callers:

- `local`  (scope `rag.local`): every knowledge base, ACL groups public + internal;
- `guest`  (scope `rag.guest`): only the KBs open to every caller (hf_docs), group public.

No money: the chat model is the fake one (`CHAT_BACKEND=fake`) and the query embedding gateway
URL is replaced by an address nothing listens on, so a question whose vector is not in the L0
cache is searched by BM25 alone (counted as degraded) instead of being sent to the gateway.
The result cache uses its own key prefix `p801:l1:<run>:`, deleted at the end. The report goes
to `<out>.json` and `<out>.md`; the exit code is 1 when anything leaked.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from rag.config import AuthCallerConfig, ServiceSettings, repo_path
from rag.eval.access.leaks import AccessCase, CallerSpec, run_leak_check
from rag.eval.questions import load_question_set

DEFAULT_OUT = "data/reports/p801-access"
NO_GATEWAY = "http://127.0.0.1:9/v1"  # the discard port: connections are refused at once
QUESTION_SETS = {
    "hf_docs": "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl",
    "pdf_demo": "evals/datasets/pdf_demo/questions.jsonl",
}
# The MT5 manual has no question set; questions about what it documents.
MT5_QUESTIONS = (
    "IMTManagerAPI::UserCreate 方法的参数是什么?",
    "How do I subscribe to tick data with the Manager API?",
    "IMTConGroup 结构里保证金调用级别怎么设置?",
    "Which gateway API callback reports a deal execution?",
    "Report API 如何注册一个新的报表?",
)
LOCAL = AuthCallerConfig(scope="rag.local", caller="local", acl_groups=["public", "internal"])
GUEST = AuthCallerConfig(scope="rag.guest", caller="guest", acl_groups=["public"])


def cases(per_kb: int) -> list[AccessCase]:
    out: list[AccessCase] = []
    for kb, path in QUESTION_SETS.items():
        questions = load_question_set(repo_path(path), kb=kb).questions[:per_kb]
        out += [AccessCase(kb, q.question) for q in questions]
    out += [AccessCase("mt5", q) for q in MT5_QUESTIONS[:per_kb]]
    return out


def leak_settings(
    jwks_path: Path, run: str, base: ServiceSettings | None = None
) -> ServiceSettings:
    from rag.testing.auth import AUDIENCE, ISSUER

    base = base or ServiceSettings()
    return base.model_copy(
        update={
            "auth_mode": "oauth",
            "auth_issuer": ISSUER,
            "auth_audience": AUDIENCE,
            "auth_jwks_path": str(jwks_path),
            "auth_jwks_url": "",
            "auth_callers": [LOCAL, GUEST],
            "chat_backend": "fake",
            "embed_base_url": NO_GATEWAY,
            "cache_mode": "l1",
            "cache_key_prefix": f"p801:l1:{run}:",
            "rate_limit_mode": "off",
            "budget_daily_cny": 0.0,
            "budget_daily_caller_cny": 0.0,
            "otel_mode": "off",
            "business_record_path": f"data/observe/p801/{run}/business.jsonl",
            "service_allowed_hosts": ["testserver"],
        }
    )


def pick_cache_case(settings: ServiceSettings, *, skip: set[str]) -> AccessCase | None:
    """An hf_docs question nobody in the run asks whose query vector is in the L0 cache: a
    question searched by BM25 alone is degraded, and degraded results are never cached, so the
    cache check would prove nothing. None when no such question is found."""
    from rag.embed import make_embedding

    embed = make_embedding("query", settings)
    lookup = getattr(embed, "cached_query_embedding", None)
    if lookup is None:
        return None
    questions = load_question_set(repo_path(QUESTION_SETS["hf_docs"]), kb="hf_docs").questions
    for q in questions:
        if q.question not in skip and lookup(q.question) is not None:
            return AccessCase("hf_docs", q.question)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.eval.access", description=__doc__)
    parser.add_argument("--questions", type=int, default=3, help="questions per knowledge base")
    parser.add_argument("--out", default=DEFAULT_OUT, help="report path without extension")
    parser.add_argument("--no-ask", action="store_true", help="search and context only")
    args = parser.parse_args(argv)

    from starlette.testclient import TestClient

    from rag.factories import make_redis_client
    from rag.security.tokens import make_token_verifier
    from rag.serve.http_app import build_http_app
    from rag.serve.service import SearchService
    from rag.testing.auth import FakeIssuer

    run = uuid.uuid4().hex[:8]
    issuer = FakeIssuer()
    with tempfile.TemporaryDirectory() as tmp:
        settings = leak_settings(issuer.write_jwks(Path(tmp) / "jwks.json"), run)
        service = SearchService.from_settings(settings)
        app = build_http_app(service, token_verifier=make_token_verifier(settings))

        def bearer(scope: str) -> dict[str, str]:
            return {"Authorization": f"Bearer {issuer.token(scopes=(scope,), expires_in_s=7200)}"}

        open_kbs = frozenset(kb for kb, cfg in service.configs.items() if "*" in cfg.access.callers)
        callers = [
            CallerSpec("local", bearer("rag.local"), frozenset(service.configs)),
            CallerSpec("guest", bearer("rag.guest"), open_kbs),
        ]
        todo = cases(args.questions)
        cache_case = pick_cache_case(settings, skip={c.question for c in todo})
        began = time.perf_counter()
        try:
            with TestClient(app) as client:
                if not service.wait_ready(900):
                    print(f"service not ready: {service.status()}", file=sys.stderr)
                    return 2
                loaded = time.perf_counter() - began
                report = run_leak_check(
                    client,
                    callers,
                    todo,
                    ask=not args.no_ask,
                    cache_case=cache_case,
                )
        finally:
            redis = make_redis_client(settings.redis_url, socket_timeout=2.0)
            try:
                keys = list(redis.scan_iter(match=f"{settings.cache_key_prefix}*", count=500))
                if keys:
                    redis.delete(*keys)
            finally:
                redis.close()
    took = time.perf_counter() - began
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"run": run, "load_s": round(loaded, 1), "total_s": round(took, 1), **report.to_json()}
    out.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    text = f"# Access leak check {run}\n\nload {loaded:.0f} s, total {took:.0f} s\n\n"
    out.with_suffix(".md").write_text(text + report.markdown() + "\n", encoding="utf-8", newline="")
    print(report.markdown())
    return 1 if report.leaked_chunks or report.exposures else 0
