"""A small closed-loop load test of the HTTP API (spec P8-01, design s16-6: "say how much it can
take only after a load test").

    uv run python -m rag.ops.loadtest --url http://127.0.0.1:28181 --kb demo_docs \\
        --tool search_docs --levels 1,4,8,16 --duration 30          # against --dev-fake
    uv run python -m rag.ops.loadtest --url http://127.0.0.1:28182 --kb hf_docs \\
        --questions evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl --cached-only \\
        --levels 1,2,4,8 --duration 60 --server-pid <pid>          # real search_docs

For each concurrency level, that many workers send requests back to back for `--duration`
seconds (after `--warmup` seconds whose requests are not counted), each worker taking the next
question of the list. Reported per level: requests, errors by status, degraded and rerank-timeout
answers, throughput, P50/P95/P99 latency, the machine's CPU use and, with `--server-pid`, the
server process's. `--cached-only` keeps the questions whose query vector is in the L0 cache, so a
real search costs nothing and is not degraded. ask_docs is accepted only for the dev-fake KB
(`demo_docs`): this tool never makes real model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

FAKE_KB = "demo_docs"
DEFAULT_QUESTIONS = (
    "How do I push a model to the Hub?",
    "How do I train a tokenizer?",
    "How do I load a dataset in streaming mode?",
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100) of `values`; NaN when empty."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass
class LevelResult:
    concurrency: int
    duration_s: float
    requests: int
    ok: int
    statuses: dict[str, int]
    degraded: int
    rerank_timeout: int
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    cpu_percent: float | None = None
    server_cpu_percent: float | None = None
    errors: list[str] = field(default_factory=list)


def body_for(tool: str, kb: str, question: str) -> dict[str, Any]:
    if tool == "search_docs":
        return {"kb": kb, "query": question}
    return {"kb": kb, "question": question}


async def _sample_cpu(stop: asyncio.Event, pid: int | None, out: dict[str, list[float]]) -> None:
    import psutil

    psutil.cpu_percent(interval=None)
    proc = psutil.Process(pid) if pid else None
    if proc is not None:
        proc.cpu_percent(interval=None)
    while not stop.is_set():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 1.0)
        out["machine"].append(psutil.cpu_percent(interval=None))
        if proc is not None:
            out["server"].append(proc.cpu_percent(interval=None) / psutil.cpu_count())


async def run_level(
    client: httpx.AsyncClient,
    *,
    tool: str,
    kb: str,
    questions: Sequence[str],
    concurrency: int,
    duration_s: float,
    warmup_s: float = 0.0,
    headers: dict[str, str] | None = None,
    server_pid: int | None = None,
    measure_cpu: bool = True,
) -> LevelResult:
    path = f"/v1/{tool}"
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    flags = Counter[str]()
    errors: list[str] = []
    counter = 0
    began = time.perf_counter()
    measure_from = began + warmup_s
    end = measure_from + duration_s

    async def worker() -> None:
        nonlocal counter
        while time.perf_counter() < end:
            question = questions[counter % len(questions)]
            counter += 1
            t0 = time.perf_counter()
            try:
                r = await client.post(path, json=body_for(tool, kb, question), headers=headers)
                status = str(r.status_code)
                data = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
            except httpx.HTTPError as exc:
                status, data = type(exc).__name__, {}
                if len(errors) < 5:
                    errors.append(f"{type(exc).__name__}: {exc}")
            t1 = time.perf_counter()
            if t0 < measure_from or t1 > end:
                continue  # warm-up, or finished after the window: not counted
            latencies.append((t1 - t0) * 1000.0)
            statuses[status] += 1
            if isinstance(data, dict):
                flags["degraded"] += bool(data.get("degraded"))
                flags["rerank_timeout"] += bool(data.get("rerank_timeout"))

    cpu: dict[str, list[float]] = {"machine": [], "server": []}
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_cpu(stop, server_pid, cpu)) if measure_cpu else None
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    stop.set()
    if sampler is not None:
        await sampler
    ok = statuses.get("200", 0)
    return LevelResult(
        concurrency=concurrency,
        duration_s=duration_s,
        requests=len(latencies),
        ok=ok,
        statuses=dict(statuses),
        degraded=flags["degraded"],
        rerank_timeout=flags["rerank_timeout"],
        throughput_rps=round(ok / duration_s, 3),
        p50_ms=round(percentile(latencies, 50), 1),
        p95_ms=round(percentile(latencies, 95), 1),
        p99_ms=round(percentile(latencies, 99), 1),
        max_ms=round(max(latencies), 1) if latencies else math.nan,
        cpu_percent=round(sum(cpu["machine"]) / len(cpu["machine"]), 1) if cpu["machine"] else None,
        server_cpu_percent=round(sum(cpu["server"]) / len(cpu["server"]), 1)
        if cpu["server"]
        else None,
        errors=errors,
    )


def markdown(results: Sequence[LevelResult], *, target_p95_ms: float | None = None) -> str:
    lines = [
        "| concurrency | requests | ok | other statuses | degraded | rerank timeout | req/s | "
        "P50 ms | P95 ms | P99 ms | machine CPU % | server CPU % |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        other = ", ".join(f"{k}: {v}" for k, v in sorted(r.statuses.items()) if k != "200") or "-"
        lines.append(
            f"| {r.concurrency} | {r.requests} | {r.ok} | {other} | {r.degraded} | "
            f"{r.rerank_timeout} | {r.throughput_rps:.2f} | {r.p50_ms:.0f} | {r.p95_ms:.0f} | "
            f"{r.p99_ms:.0f} | {r.cpu_percent if r.cpu_percent is not None else '-'} | "
            f"{r.server_cpu_percent if r.server_cpu_percent is not None else '-'} |"
        )
    if target_p95_ms is not None:
        within = [r.concurrency for r in results if r.ok and r.p95_ms <= target_p95_ms]
        best = max(within) if within else None
        lines.append("")
        lines.append(f"Highest concurrency with P95 <= {target_p95_ms:.0f} ms: {best}")
    return "\n".join(lines)


def cached_questions(questions: Sequence[str]) -> list[str]:
    """The questions whose query vector is in the L0 cache (Redis), read with the service's own
    embedding factory; nothing is sent to the gateway."""
    from rag.config import ServiceSettings
    from rag.embed import make_embedding

    embed = make_embedding("query", ServiceSettings())
    lookup = getattr(embed, "cached_query_embedding", None)
    if lookup is None:
        return []
    return [q for q in questions if lookup(q) is not None]


async def run_load(
    url: str,
    *,
    tool: str,
    kb: str,
    questions: Sequence[str],
    levels: Sequence[int],
    duration_s: float,
    warmup_s: float,
    server_pid: int | None,
    timeout_s: float,
) -> list[LevelResult]:
    limits = httpx.Limits(max_connections=max(levels) + 4, max_keepalive_connections=max(levels))
    async with httpx.AsyncClient(base_url=url, timeout=timeout_s, limits=limits) as client:
        results = []
        for level in levels:
            result = await run_level(
                client,
                tool=tool,
                kb=kb,
                questions=questions,
                concurrency=level,
                duration_s=duration_s,
                warmup_s=warmup_s,
                server_pid=server_pid,
            )
            print(markdown([result]).splitlines()[-1], flush=True)
            results.append(result)
        return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.ops.loadtest", description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tool", choices=("search_docs", "ask_docs"), default="search_docs")
    parser.add_argument("--kb", default=FAKE_KB)
    parser.add_argument("--questions", help="JSONL question set (default: three fixed questions)")
    parser.add_argument("--limit", type=int, default=60, help="questions taken from the set")
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--levels", default="1,4,8,16")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--target-p95-ms", type=float)
    parser.add_argument("--out", help="JSON report path")
    args = parser.parse_args(argv)
    if args.tool == "ask_docs" and args.kb != FAKE_KB:
        print("ask_docs is load-tested only against the dev-fake KB demo_docs", file=sys.stderr)
        return 2
    questions: list[str] = list(DEFAULT_QUESTIONS)
    if args.questions:
        from rag.config import repo_path
        from rag.eval.questions import load_question_set

        questions = [q.question for q in load_question_set(repo_path(args.questions)).questions]
    if args.cached_only:
        questions = cached_questions(questions)
    questions = questions[: args.limit]
    if not questions:
        print("no questions (none has a cached query vector?)", file=sys.stderr)
        return 2
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    results = asyncio.run(
        run_load(
            args.url,
            tool=args.tool,
            kb=args.kb,
            questions=questions,
            levels=levels,
            duration_s=args.duration,
            warmup_s=args.warmup,
            server_pid=args.server_pid,
            timeout_s=args.timeout,
        )
    )
    table = markdown(results, target_p95_ms=args.target_p95_ms)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "url": args.url,
            "tool": args.tool,
            "kb": args.kb,
            "questions": len(questions),
            "levels": [asdict(r) for r in results],
            "markdown": table,
        }
        out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
