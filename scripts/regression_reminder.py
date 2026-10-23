"""Pre-commit reminder (spec EVAL-4): the commit touches code the regression set guards.

Run by the `regression-reminder` hook of `.pre-commit-config.yaml`, which only starts it when a
staged file matches the guarded paths. It prints one message and always exits 0: it is a reminder,
not a check, it never blocks a commit and it never calls a model.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

COMMAND = "uv run python -m rag.eval.agentic regress"


def reminder(files: Sequence[str] = ()) -> str:
    lines = [
        "please run the regression command before you merge this change:",
        f"  {COMMAND}",
        "(about 0.45 CNY and a few minutes; it prints the estimate first. This commit touches the "
        "agent, the answer step or the retrieval configuration.)",
    ]
    if files:
        lines.append("touched: " + ", ".join(files[:5]) + (" ..." if len(files) > 5 else ""))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    print(reminder(list(argv if argv is not None else sys.argv[1:])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
