"""Run the support judge over stored answers, one line per answer (spec P1-06).

A key (source, system, qid, run) that already has a line is not asked again, so the same command
resumes after a crash or continues a pilot (`limit`) into the full run; `retry_unknown` asks
again the answers whose newest line has an unknown verdict. The run stops before an answer once
the spend of the lines in the file plus `spent_before` has reached `max_cny`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.answer.support import SupportJudgeConfig, judge_support
from rag.eval.answer.record import Usage
from rag.eval.answer.usage import collect_ledger_rows
from rag.eval.support.records import (
    SupportItem,
    SupportLine,
    latest_lines,
    read_lines,
    repair_tail,
    write_line,
)


@dataclass
class RunStats:
    ran: int = 0
    """New lines written (skipped items included: they cost nothing)."""
    done_before: int = 0
    pending: int = 0
    spent_cny: float = 0.0
    """This file's lines plus `spent_before`, at the end."""
    stopped: str | None = None
    """`max_cny` when the spend cap stopped the run."""


async def judge_item(
    item: SupportItem, *, run: int, model: Any, config: SupportJudgeConfig
) -> SupportLine:
    if item.skip_reason is not None:
        return SupportLine(**item.key_fields(), run=run, item=item)
    with collect_ledger_rows() as rows:
        result = await judge_support(
            item.sentences, item.chunks, model=model, config=config, question=item.question
        )
    return SupportLine(
        **item.key_fields(), run=run, item=item, result=result, usage=Usage.from_rows(rows)
    )


async def run_support(
    items: Sequence[SupportItem],
    *,
    run: int,
    model: Any,
    config: SupportJudgeConfig,
    out_path: Path,
    max_cny: float = 0.6,
    spent_before: float = 0.0,
    limit: int | None = None,
    retry_unknown: bool = False,
    progress: Callable[[int, int, SupportLine], None] | None = None,
) -> RunStats:
    repair_tail(out_path)
    latest = {x.key: x for x in latest_lines(read_lines(out_path))}
    stats = RunStats()
    todo: list[SupportItem] = []
    for item in items:
        key = (item.source, item.system, item.qid, run)
        line = latest.get(key)
        if line is not None and not (retry_unknown and line.has_unknown):
            stats.done_before += 1
        else:
            todo.append(item)
    if limit is not None:
        todo, rest = todo[:limit], todo[limit:]
        stats.pending = len(rest)
    spent = spent_before + sum(x.usage.cny for x in read_lines(out_path))
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(config.concurrency)
    total = len(todo)

    async def one(item: SupportItem) -> None:
        nonlocal spent
        async with semaphore:
            async with lock:
                if item.skip_reason is None and spent >= max_cny:
                    stats.stopped = "max_cny"
                    stats.pending += 1
                    return
            line = await judge_item(item, run=run, model=model, config=config)
            async with lock:
                write_line(out_path, line)
                spent += line.usage.cny
                stats.ran += 1
                if progress is not None:
                    progress(stats.ran, total, line)

    await asyncio.gather(*(one(item) for item in todo))
    stats.spent_cny = round(spent, 6)
    return stats
