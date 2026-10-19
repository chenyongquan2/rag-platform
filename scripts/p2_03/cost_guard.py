"""Spec P2-03: stop a real-embedding job when it has spent its share of the budget.

    uv run python scripts/p2_03/cost_guard.py --max-delta 2.95 --match p203c200_ [--poll 20]

Records the spent total of the cost ledger when it starts, then polls it. When the total has grown
by more than `--max-delta` CNY it kills every process whose command line contains `--match` (and
their children) and exits with code 3. It exits with code 0 when no such process is left. The
embedding cache keeps every vector already paid for, so a stopped job loses no money, only time.
"""

from __future__ import annotations

import argparse
import sys
import time

import psutil

from rag.config import ServiceSettings
from rag.cost import total_spent


def matching(marker: str) -> list[psutil.Process]:
    me = psutil.Process().pid
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if marker in cmd and "cost_guard.py" not in cmd and p.info["pid"] != me:
            found.append(p)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-delta", type=float, required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--poll", type=float, default=20.0)
    args = parser.parse_args()
    settings = ServiceSettings()
    start = total_spent(settings)
    print(f"guard: spent {start:.6f} CNY at start, limit +{args.max_delta} CNY", flush=True)
    seen = False
    while True:
        procs = matching(args.match)
        seen = seen or bool(procs)
        delta = total_spent(settings) - start
        if delta > args.max_delta:
            for p in procs:
                for child in p.children(recursive=True):
                    child.kill()
                p.kill()
            print(f"guard: spent +{delta:.4f} CNY, killed {len(procs)} process(es)", flush=True)
            return 3
        if seen and not procs:
            print(f"guard: job ended, spent +{delta:.4f} CNY", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
