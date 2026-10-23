"""The numbers of the repeated runs (spec EVAL-3, plan section 9). Pure functions of the passes: no
model, no search, no network.

Fixed before the runs:

- a question is *correct* in a pass when the judge scored the answer 3; an ambiguous question is
  correct when the system asked back (`asked_back`, a rule, no judge). A deadline-cut answer, a
  model timeout and a run error are not correct; they are also listed apart (never counted as
  refusals).
  No-answer questions are not part of the repeated list.
- *pass^k*: correct in all `k` passes (is it stable). *pass@k*: correct in at least one pass (can it
  do it at all). Both with the 95% Wilson interval.
- the two systems are compared on pass^k with the exact paired McNemar test (a = standard RAG,
  b = the graph).
- *noise*: for one system, the questions whose result differs between two adjacent passes. The
  question of the plan, "is the difference between the two systems larger than the difference
  between a system's own passes?", is answered by comparing the gap (graph minus standard RAG, in
  questions with pass^k) with the largest such count of either system. This is the plan's own
  wording and a strict yardstick: flips that cancel each other still count.
- only questions with a decidable result in every pass of both systems enter the numbers; the others
  are listed as incomplete.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from rag.eval.agentic.analysis import CORRECT, PairedBinary, paired_binary
from rag.eval.agentic.records import SYSTEMS, retry_reason
from rag.eval.agentic.repeat.passes import Cell, PassData
from rag.eval.agentic.repeat.sample import LIST_TYPES, OLD_COUNTS, QuestionList
from rag.eval.answer.record import Proportion
from rag.eval.hard.clarify import asked_back

OLD_GROUP = "old"
ALL_GROUP = "all"
GROUP_ORDER: tuple[str, ...] = (*LIST_TYPES, OLD_GROUP, ALL_GROUP)


def group_types(group: str) -> tuple[str, ...]:
    if group == OLD_GROUP:
        return tuple(OLD_COUNTS)
    if group == ALL_GROUP:
        return LIST_TYPES
    return (group,)


class SystemGroup(BaseModel):
    pass_all: Proportion
    pass_any: Proportion
    per_pass: list[Proportion]
    """Correct in each single pass, as a one-run report would have shown it."""


class Flips(BaseModel):
    """Questions of one system whose result differs between pass `from_pass` and the next."""

    system: str
    from_pass: int
    changed: int
    to_right: int
    to_wrong: int
    changed_clean: int
    """`changed` without the questions that had a cut, failed or timed-out record in either pass."""
    qids: list[str] = Field(default_factory=list)


class GroupResult(BaseModel):
    group: str
    n: int
    systems: dict[str, SystemGroup]
    paired: PairedBinary
    """McNemar on pass^k: a = standard RAG, b = the graph."""
    flips: list[Flips]
    max_flips: int
    mean_flips: float
    gap: int
    """Questions with pass^k: graph minus standard RAG."""
    gap_per_pass: list[int]
    gap_larger_than_noise: bool


class Incident(BaseModel):
    """A record `--retry-errors` would run again, in a pass of a sampled question."""

    system: str
    pass_no: int
    qid: str
    qtype: str
    reason: str
    """`deadline_truncated`, `generation_error` or `error` (see `records.retry_reason`)."""


class Incomplete(BaseModel):
    qid: str
    system: str
    pass_no: int
    kind: str
    """`missing` (no record) or `not_judged` (answered, no judge score)."""

    @property
    def reason(self) -> str:
        if self.kind == "not_judged":
            return f"{self.system} answer in pass {self.pass_no} is not judged"
        return f"{self.system} has no record in pass {self.pass_no}"


class RepeatAnalysis(BaseModel):
    passes: int
    questions: int
    """Questions in the numbers (a result in every pass, for both systems)."""
    incomplete: list[Incomplete] = Field(default_factory=list)
    groups: dict[str, GroupResult] = Field(default_factory=dict)
    incidents: list[Incident] = Field(default_factory=list)
    stable: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    """`standard`, `agentic`, `both` -> type -> ids of the questions correct in every pass."""
    pass_cny: dict[str, list[float]] = Field(default_factory=dict)
    """Per system, per pass: CNY of the sampled questions' records plus their judge calls."""


def is_correct(qtype: str, cell: Cell) -> bool | None:
    """Correct in this pass; `None` when it cannot be decided (answer not judged)."""
    if qtype == "ambiguous":
        return asked_back(cell.record)
    if cell.score is None:
        return None
    return cell.score >= CORRECT


def _outcomes(
    passes: Sequence[PassData], qlist: QuestionList
) -> tuple[dict[str, dict[str, list[bool]]], list[Incomplete]]:
    """Per system and question the correct flag of every pass, for the decidable questions."""
    qtypes = qlist.qtype_of()
    flags: dict[str, dict[str, list[bool]]] = {s: {} for s in SYSTEMS}
    incomplete: list[Incomplete] = []
    for qid in qlist.all_qids():
        problem: Incomplete | None = None
        row: dict[str, list[bool]] = {s: [] for s in SYSTEMS}
        for system in SYSTEMS:
            for number, data in enumerate(passes, start=1):
                found = data.get(system, {}).get(qid)
                if found is None:
                    problem = Incomplete(qid=qid, system=system, pass_no=number, kind="missing")
                    break
                ok = is_correct(qtypes[qid], found)
                if ok is None:
                    problem = Incomplete(qid=qid, system=system, pass_no=number, kind="not_judged")
                    break
                row[system].append(ok)
            if problem:
                break
        if problem:
            incomplete.append(problem)
            continue
        for system in SYSTEMS:
            flags[system][qid] = row[system]
    return flags, incomplete


def _incidents(passes: Sequence[PassData], qlist: QuestionList) -> list[Incident]:
    qtypes = qlist.qtype_of()
    found: list[Incident] = []
    for number, data in enumerate(passes, start=1):
        for system in SYSTEMS:
            for qid in qlist.all_qids():
                cell = data.get(system, {}).get(qid)
                reason = retry_reason(cell.record) if cell else None
                if reason:
                    found.append(
                        Incident(
                            system=system, pass_no=number, qid=qid, qtype=qtypes[qid], reason=reason
                        )
                    )
    return found


def _group(
    group: str,
    qids: list[str],
    flags: dict[str, dict[str, list[bool]]],
    bad: dict[str, dict[str, list[bool]]],
    k: int,
) -> GroupResult:
    n = len(qids)
    systems: dict[str, SystemGroup] = {}
    for system in SYSTEMS:
        rows = [flags[system][q] for q in qids]
        systems[system] = SystemGroup(
            pass_all=Proportion.of(sum(all(r) for r in rows), n),
            pass_any=Proportion.of(sum(any(r) for r in rows), n),
            per_pass=[Proportion.of(sum(r[p] for r in rows), n) for p in range(k)],
        )
    paired = paired_binary(
        f"pass^{k}",
        {q: all(flags["standard"][q]) for q in qids},
        {q: all(flags["agentic"][q]) for q in qids},
    )
    flips: list[Flips] = []
    for system in SYSTEMS:
        for p in range(k - 1):
            moved = [q for q in qids if flags[system][q][p] != flags[system][q][p + 1]]
            flips.append(
                Flips(
                    system=system,
                    from_pass=p + 1,
                    changed=len(moved),
                    to_right=sum(1 for q in moved if flags[system][q][p + 1]),
                    to_wrong=sum(1 for q in moved if flags[system][q][p]),
                    changed_clean=sum(
                        1 for q in moved if not (bad[system][q][p] or bad[system][q][p + 1])
                    ),
                    qids=moved,
                )
            )
    counts = [f.changed for f in flips]
    gap = systems["agentic"].pass_all.count - systems["standard"].pass_all.count
    return GroupResult(
        group=group,
        n=n,
        systems=systems,
        paired=paired,
        flips=flips,
        max_flips=max(counts) if counts else 0,
        mean_flips=sum(counts) / len(counts) if counts else 0.0,
        gap=gap,
        gap_per_pass=[
            systems["agentic"].per_pass[p].count - systems["standard"].per_pass[p].count
            for p in range(k)
        ],
        gap_larger_than_noise=abs(gap) > (max(counts) if counts else 0),
    )


def analyze(passes: Sequence[PassData], qlist: QuestionList) -> RepeatAnalysis:
    """The repeated-run numbers of `passes` (pass 1 first) over the questions of `qlist`."""
    k = len(passes)
    if k < 2:
        raise ValueError("the repeated-run analysis needs at least two passes")
    flags, incomplete = _outcomes(passes, qlist)
    # Per pass: did this record fail for a reason that is not the question's own?
    bad = {
        s: {
            q: [bool((c := data.get(s, {}).get(q)) and retry_reason(c.record)) for data in passes]
            for q in flags[s]
        }
        for s in SYSTEMS
    }
    qtypes = qlist.qtype_of()
    universe = list(flags["standard"])
    result = RepeatAnalysis(passes=k, questions=len(universe), incomplete=incomplete)
    for group in GROUP_ORDER:
        members = set(group_types(group))
        qids = [q for q in universe if qtypes[q] in members]
        if qids:
            result.groups[group] = _group(group, qids, flags, bad, k)
    result.incidents = _incidents(passes, qlist)
    result.stable = _stable(flags, qlist, universe)
    result.pass_cny = {
        system: [
            sum(
                c.record.usage.cny + c.judge_cny
                for q, c in data.get(system, {}).items()
                if q in qtypes
            )
            for data in passes
        ]
        for system in SYSTEMS
    }
    return result


def _stable(
    flags: dict[str, dict[str, list[bool]]], qlist: QuestionList, universe: list[str]
) -> dict[str, dict[str, list[str]]]:
    members = set(universe)
    out: dict[str, dict[str, list[str]]] = {"standard": {}, "agentic": {}, "both": {}}
    for qtype in LIST_TYPES:
        ids = [q for q in qlist.qids.get(qtype, []) if q in members]
        for system in SYSTEMS:
            out[system][qtype] = [q for q in ids if all(flags[system][q])]
        out["both"][qtype] = [
            q for q in ids if all(flags["standard"][q]) and all(flags["agentic"][q])
        ]
    return out


def stable_payload(result: RepeatAnalysis, qlist: QuestionList) -> dict[str, Any]:
    """The file step 4 reads: questions correct in every pass, per system and type."""
    return {
        "version": 1,
        "passes": result.passes,
        "question_list_seed": qlist.seed,
        "question_list_date": qlist.date,
        "universe": result.questions,
        "note": "correct in every pass: judge score 3 (ambiguous: asked back)",
        "standard": result.stable["standard"],
        "agentic": result.stable["agentic"],
        "both": result.stable["both"],
    }
