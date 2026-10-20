"""Time the streaming endpoint from the client's side (spec P3-04, design s16-9: the first
progress within 1 to 2 seconds).

    uv run python scripts/stream_latency.py --url http://127.0.0.1:28099 --kb demo_docs \
        --question "How do I push a model with push_to_hub?" --runs 20

Each run is a new conversation (no history, so the same question is a fresh run). It reports, in
milliseconds since the request was sent: the first byte of the response, the first `progress`
event, the first `token` event, the `answer` event and the end. The numbers are what a client sees
over this connection; the server's own view is `timings_ms` of the `done` event.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid

import httpx


def one_run(url: str, kb: str, question: str, timeout_s: float) -> dict[str, float]:
    body = {"kb": kb, "question": question, "session_id": f"lat-{uuid.uuid4().hex[:12]}"}
    marks: dict[str, float] = {}
    began = time.perf_counter()
    buffer = ""
    with httpx.stream("POST", f"{url}/v1/ask_docs/stream", json=body, timeout=timeout_s) as r:
        if r.status_code != 200:
            raise SystemExit(f"HTTP {r.status_code}: {r.read().decode()[:300]}")
        for text in r.iter_text():
            marks.setdefault("first_byte", (time.perf_counter() - began) * 1000)
            buffer += text
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                name = next(
                    (ln[6:].strip() for ln in block.split("\n") if ln.startswith("event:")), ""
                )
                if name in ("progress", "token", "answer", "need_clarification", "error", "done"):
                    marks.setdefault(f"first_{name}", (time.perf_counter() - began) * 1000)
                if name == "done":
                    data = next(ln[5:] for ln in block.split("\n") if ln.startswith("data:"))
                    marks["server_request"] = json.loads(data)["timings_ms"].get("request", 0.0)
    marks["end"] = (time.perf_counter() - began) * 1000
    return marks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28080")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()
    runs = [one_run(args.url, args.kb, args.question, args.timeout_s) for _ in range(args.runs)]
    print(f"{args.runs} runs, milliseconds since the request was sent")
    print(f"{'event':<22}{'min':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for key in ("first_byte", "first_progress", "first_token", "first_answer", "end"):
        values = sorted(r[key] for r in runs if key in r)
        if not values:
            continue
        p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
        print(
            f"{key:<22}{values[0]:>9.0f}{statistics.median(values):>9.0f}{p95:>9.0f}{values[-1]:>9.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
