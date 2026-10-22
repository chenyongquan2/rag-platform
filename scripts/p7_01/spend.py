"""Spec P7-01: ledger lines since a UTC time, summed by kind, model and profile.

  uv run python scripts/p7_01/spend.py 2026-10-22T05:55:00
The ledger is shared by every checkout, so lines of other work in the same window count too.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from rag.config import ServiceSettings


def main(since: str) -> int:
    path = Path(ServiceSettings().cost_ledger_path)
    sums: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ts = str(rec.get("ts") or rec.get("time") or rec.get("created_at") or "")
        if ts < since:
            continue
        key = (str(rec.get("kind")), str(rec.get("model")), str(rec.get("profile")))
        sums[key][0] += 1
        sums[key][1] += float(rec.get("cny") or 0.0)
        total += float(rec.get("cny") or 0.0)
    for (kind, model, profile), (calls, cny) in sorted(sums.items()):
        print(f"{kind:9} {model:32} {profile:8} calls={calls:4.0f} cny={cny:.4f}")
    print(f"total since {since}: {total:.4f} CNY")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
