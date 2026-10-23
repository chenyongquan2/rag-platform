"""One pass of the repeated runs: where its records are, in which order the systems run, the
retrieval-only warm-up, and the cost estimate that is checked against the stop line (spec EVAL-3).

A pass is a records directory in the layout of every other run of this package
(`standard/records.jsonl`, `agentic/records.jsonl`, `judge.jsonl`). Pass 1 is not run again: the
old questions come from the stored P3-07 run (several directories may be named, a later one wins per
system and question, so the retry directory can follow the main one) and the new questions from the
EVAL-2 step-2 run. Passes 2 and later live in `BASE/passN`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from rag.eval.agentic.analysis import judge_scores
from rag.eval.agentic.records import (
    SYSTEMS,
    SystemRecord,
    judge_file,
    read_judge_lines,
    read_records,
    system_file,
)
from rag.eval.agentic.repeat.sample import QuestionList
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.record import AnswerItem
from rag.eval.retrieval.cli import UsageError

MARGIN = 0.2
"""Share added to every estimate (plan section 14: the P3-05 pilot was 18% too low)."""
JUDGE_CNY_PER_CALL = 0.0014
MANIFEST_NAME = "repeat.json"
FALLBACK_TYPE = "two_hop"
"""A type with no earlier records is priced like this one (plan section 14)."""


# ---- where a pass is -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One system's answer to one question in one pass, with the judge's score (pass 1 of the
    judge; `None` when the answer has not been judged)."""

    record: SystemRecord
    score: int | None = None
    judge_cny: float = 0.0


PassData = dict[str, dict[str, Cell]]
"""system -> question id -> cell."""


def empty_pass() -> PassData:
    return {system: {} for system in SYSTEMS}


def pass_dir(base: str | Path, pass_no: int) -> Path:
    return Path(base) / f"pass{pass_no}"


def load_pass(dirs: Sequence[str | Path], qids: Collection[str] | None) -> PassData:
    """The cells of the questions `qids` (all when `None`) found in `dirs`; for a question in
    several directories the last one wins, with its own judge score."""
    data = empty_pass()
    for directory in dirs:
        lines = read_judge_lines(judge_file(directory))
        for system in SYSTEMS:
            scores = judge_scores(lines, system)
            cost = {x.qid: x.usage.cny for x in lines if x.system == system and x.run == 1}
            for record in read_records(system_file(directory, system)):
                if qids is not None and record.qid not in qids:
                    continue
                data[system][record.qid] = Cell(
                    record=record,
                    score=scores.get(record.qid),
                    judge_cny=cost.get(record.qid, 0.0),
                )
    return data


def merge_pass(*parts: PassData) -> PassData:
    out = empty_pass()
    for part in parts:
        for system, cells in part.items():
            out[system].update(cells)
    return out


def load_stored_pass(
    old_dirs: Sequence[str | Path], new_dirs: Sequence[str | Path], ql: QuestionList
) -> PassData:
    """Pass 1: old questions from the P3-07 directories, hard and ambiguous ones from the step-2
    directories. A question the directories do not have is absent."""
    return merge_pass(
        load_pass(old_dirs, set(ql.old_qids())), load_pass(new_dirs, set(ql.new_qids()))
    )


def system_order(pass_no: int) -> tuple[str, str]:
    """Which system runs first: even passes standard RAG, odd passes the graph, so cache warmth
    is split between the two systems (plan section 9; pass 1 was standard first)."""
    return ("standard", "agentic") if pass_no % 2 == 0 else ("agentic", "standard")


# ---- manifest ------------------------------------------------------------------------------------


class RepeatManifest(BaseModel):
    """Where pass 1 and the question list are, saved in `BASE/repeat.json` by `repeat-run` so the
    later commands need not repeat the paths."""

    question_list: str = ""
    pass1_old: list[str] = Field(default_factory=list)
    pass1_new: list[str] = Field(default_factory=list)


def load_manifest(base: str | Path) -> RepeatManifest:
    file = Path(base) / MANIFEST_NAME
    if not file.is_file():
        return RepeatManifest()
    return RepeatManifest.model_validate_json(file.read_text(encoding="utf-8"))


def save_manifest(base: str | Path, manifest: RepeatManifest) -> None:
    Path(base).mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest.model_dump(), indent=2) + "\n"
    (Path(base) / MANIFEST_NAME).write_text(text, encoding="utf-8", newline="\n")


# ---- warm-up -------------------------------------------------------------------------------------


@dataclass
class WarmUp:
    searched: int = 0
    failed: int = 0
    rounds: int = 0


def warm_up(
    contexts: Callable[[AnswerItem], object],
    items: Sequence[EvalItem],
    *,
    max_failed: int = 0,
    rounds: int = 2,
    say: Callable[[str], None] = print,
) -> WarmUp:
    """Search every question once with the retrieval chain only (no chat model), so the first
    question of the pass does not pay for a cold reranker or cold caches (P3-07 finding 3). A
    search whose rerank timed out or that raised is tried again in the next round; more than
    `max_failed` still failing after the last round refuses the pass."""
    result = WarmUp()
    pending = list(items)
    for number in range(1, rounds + 1):
        if not pending:
            break
        result.rounds = number
        failed: list[EvalItem] = []
        for item in pending:
            result.searched += 1
            try:
                context = contexts(AnswerItem(qid=item.qid, question=item.question))
            except Exception as exc:  # a search that raises is as unready as one that times out
                say(f"warm-up: {item.qid}: search failed ({type(exc).__name__}: {exc})")
                failed.append(item)
                continue
            if getattr(context, "rerank_timeout", False):
                failed.append(item)
        say(f"warm-up round {number}: {len(pending)} searches, {len(failed)} failed")
        pending = failed
    result.failed = len(pending)
    if result.failed > max_failed:
        raise UsageError(
            f"{result.failed} searches still fail (rerank timeout or search error) after "
            f"{result.rounds} warm-up rounds ({', '.join(i.qid for i in pending[:5])}); "
            "wait until Elasticsearch and the reranker are idle, then run the pass again"
        )
    return result


# ---- cost ----------------------------------------------------------------------------------------


def mean_costs(records: Iterable[SystemRecord]) -> dict[str, dict[str, float]]:
    """Mean CNY per question of each question type, per system, from earlier records."""
    sums: dict[str, dict[str, list[float]]] = {s: {} for s in SYSTEMS}
    for r in records:
        sums[r.system].setdefault(r.qtype, []).append(r.usage.cny)
    return {s: {t: sum(v) / len(v) for t, v in by.items()} for s, by in sums.items()}


@dataclass
class Estimate:
    by_system: dict[str, float] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    judge_calls: int = 0
    judge: float = 0.0
    subtotal: float = 0.0
    margin: float = MARGIN
    total: float = 0.0
    """Subtotal with the margin: the number checked against the stop line."""

    def lines(self) -> list[str]:
        out = ["estimated cost of this pass (CNY):"]
        for system, cny in self.by_system.items():
            kinds = ", ".join(f"{t} {n}" for t, n in self.counts.get(system, {}).items())
            out.append(f"  {system}: {cny:.4f} ({kinds or 'nothing left to run'})")
        out.append(f"  judge: {self.judge:.4f} ({self.judge_calls} calls)")
        out.append(
            f"  subtotal {self.subtotal:.4f}; with {self.margin:.0%} margin {self.total:.4f}"
        )
        return out


def _mean_for(means: Mapping[str, float], qtype: str) -> float | None:
    if qtype in means:
        return means[qtype]
    if FALLBACK_TYPE in means:
        return means[FALLBACK_TYPE]
    return sum(means.values()) / len(means) if means else None


def estimate_pass(
    todo: Mapping[str, Sequence[EvalItem]],
    means: Mapping[str, Mapping[str, float]],
    *,
    judge_calls: int,
    judge_cny_per_call: float = JUDGE_CNY_PER_CALL,
    margin: float = MARGIN,
) -> Estimate:
    """What running `todo` (the questions still to run, per system) will cost: each question at
    the mean of its type from earlier records, a type without records at the two-hop mean, plus the
    judge's calls, plus the margin."""
    est = Estimate(margin=margin, judge_calls=judge_calls)
    for system, items in todo.items():
        total = 0.0
        counts: dict[str, int] = {}
        for item in items:
            price = _mean_for(means.get(system, {}), item.qtype)
            if price is None:
                raise UsageError(
                    f"cannot estimate the cost of {system}: no earlier records to take a mean "
                    "from (give --pass1-old / --pass1-new)"
                )
            total += price
            counts[item.qtype] = counts.get(item.qtype, 0) + 1
        est.by_system[system] = total
        est.counts[system] = counts
    est.judge = judge_calls * judge_cny_per_call
    est.subtotal = sum(est.by_system.values()) + est.judge
    est.total = est.subtotal * (1 + margin)
    return est


def check_budget(est: Estimate, *, spent: float, stop: float) -> None:
    """Refuse a pass that would take the ledger past the stop line (CLAUDE.md, models and money)."""
    if spent + est.total > stop + 1e-9:
        raise UsageError(
            f"spent {spent:.4f} + estimated {est.total:.4f} CNY (with margin) would pass the stop "
            f"line {stop:.2f} CNY; not started. Raise the line (owner) or run fewer questions"
        )
