"""Command line: `python -m rag.cost summary [--ledger PATH]`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag.config import ServiceSettings
from rag.cost.ledger import CostLedger, LedgerSummary


def format_summary(summary: LedgerSummary) -> str:
    lines = [
        f"ledger: {summary.path}",
        f"calls: {summary.calls}",
    ]
    budget = f"budget {summary.budget_cny:.2f}, " if summary.budget_cny is not None else ""
    left = summary.budget_stop_cny - summary.spent_cny
    lines.append(
        f"spent: {summary.spent_cny:.6f} CNY ({budget}stop at {summary.budget_stop_cny:.2f}, "
        f"left before stop {left:.6f})"
    )
    for (kind, model), totals in summary.by_kind_model.items():
        lines.append(
            f"  {kind:<10} {model:<28} calls={totals.calls} tokens={totals.tokens} "
            f"cny={totals.cny:.6f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.cost", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_summary = sub.add_parser("summary", help="money spent so far, per kind and model")
    p_summary.add_argument("--ledger", help="ledger file (default: cost_ledger_path setting)")
    args = parser.parse_args(argv)

    ledger = CostLedger.from_settings(ServiceSettings())
    if args.ledger:
        ledger.path = Path(args.ledger)
    print(format_summary(ledger.summary()))
    return 0
