"""Before and after on a subset of questions (spec P3-06): the P3-05 records of standard RAG (A)
and of the graph before the fixes (B before), against a new run of the graph (B after) on the
same questions.

Pure functions of records and judge scores; the subset is whatever questions the new run has. The
questions fall into groups by what happened to them in P3-05 (`groups`), so a table row says which
finding it tests:

    crashed        the P3-05 graph run ended in an error (finding 2: a later round's search failed)
    another_round  answerable questions whose first-round verdict said "needs another round"
                   (finding 3: most of them were cut off by the deadline)
    no_answer      the other no-answer questions (finding 1: they reached the answer model)
    single_hop     the other single-hop questions (finding 4: cost and time of rewrite and judge)

What the tables can and cannot show is for the report to say; here are only the counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from rag.eval.agentic.analysis import DEADLINE_MS, TARGET_P95_MS, gold_hit
from rag.eval.agentic.records import SystemRecord
from rag.eval.retrieval.runner import latency_summary

GROUPS: tuple[str, ...] = ("crashed", "another_round", "no_answer", "single_hop", "other")
SYSTEM_LABELS: tuple[str, ...] = ("A standard", "B before", "B after")


def first_judge_verdict(record: SystemRecord) -> str | None:
    for event in record.events:
        if event.step == "judge" and event.round == 1:
            return str(event.data.get("verdict"))
    return None


def group_of(before: SystemRecord) -> str:
    """The group of a question from its P3-05 graph record (see the module doc)."""
    if before.status == "error":
        return "crashed"
    if before.answerable and first_judge_verdict(before) not in (None, "enough"):
        return "another_round"
    if before.qtype == "no_answer":
        return "no_answer"
    if before.qtype == "single_hop":
        return "single_hop"
    return "other"


def groups(before: Mapping[str, SystemRecord], qids: Sequence[str]) -> dict[str, list[str]]:
    """The question ids of `qids` by group, in `GROUPS` order (empty groups left out)."""
    found: dict[str, list[str]] = {name: [] for name in GROUPS}
    for qid in qids:
        found[group_of(before[qid])].append(qid)
    return {name: ids for name, ids in found.items() if ids}


def reached_answer_model(record: SystemRecord) -> bool:
    """The answer model was called (graph: the `online` profile; standard RAG: any call)."""
    online = record.usage_by_profile.get("online")
    if online is not None:
        return online.calls > 0
    return record.system == "standard" and record.usage.calls > 0


class Row(BaseModel):
    group: str
    label: str
    n: int
    errors: int = 0
    answered: int = 0
    refused: int = 0
    reached_answer_model: int = 0
    rounds_mean: float = 0.0
    two_or_more_rounds: int = 0
    deadline_stops: int = 0
    retrieval_errors: int = 0
    rerank_timeouts: int = 0
    judge_calls_mean: float = 0.0
    cheap_calls_mean: float = 0.0
    online_calls_mean: float = 0.0
    cny_mean: float = 0.0
    cny_total: float = 0.0
    p50_s: float = 0.0
    p95_s: float = 0.0
    max_s: float = 0.0
    over_15s: int = 0
    over_30s: int = 0
    correct: int = 0
    scored: int = 0
    notes: dict[str, int] = Field(default_factory=dict)


def _calls(record: SystemRecord, profile: str) -> int:
    usage = record.usage_by_profile.get(profile)
    return usage.calls if usage else 0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def row_of(
    group: str, label: str, records: Sequence[SystemRecord], scores: Mapping[str, int]
) -> Row:
    """One table row: the records of one system in one group."""
    lat = latency_summary([r.total_ms for r in records])
    judged = [scores[r.qid] for r in records if r.qid in scores]
    return Row(
        group=group,
        label=label,
        n=len(records),
        errors=sum(1 for r in records if r.status == "error"),
        answered=sum(1 for r in records if r.answered),
        refused=sum(1 for r in records if not r.answered),
        reached_answer_model=sum(1 for r in records if reached_answer_model(r)),
        rounds_mean=_mean([r.rounds for r in records]),
        two_or_more_rounds=sum(1 for r in records if r.rounds >= 2),
        deadline_stops=sum(1 for r in records if r.stop_reason == "deadline"),
        retrieval_errors=sum(
            1 for r in records if any(c.startswith("retrieval_") for c in r.error_categories)
        ),
        rerank_timeouts=sum(1 for r in records if r.rerank_timeout),
        cheap_calls_mean=_mean([_calls(r, "cheap") for r in records]),
        online_calls_mean=_mean([_calls(r, "online") for r in records]),
        cny_mean=_mean([r.usage.cny for r in records]),
        cny_total=sum(r.usage.cny for r in records),
        p50_s=lat.p50_ms / 1000.0,
        p95_s=lat.p95_ms / 1000.0,
        max_s=lat.max_ms / 1000.0,
        over_15s=sum(1 for r in records if r.total_ms > TARGET_P95_MS),
        over_30s=sum(1 for r in records if r.total_ms > DEADLINE_MS),
        correct=sum(1 for s in judged if s == 3),
        scored=len(judged),
    )


def table(rows: Sequence[Row]) -> str:
    head = (
        "| group | system | n | errors | answered | refused | reached answer model | "
        "rounds (mean) | 2+ rounds | stopped by deadline | retrieval errors | rerank timeouts | "
        "cheap calls | answer calls | CNY / question | P50 s | P95 s | max s | > 15 s | "
        "correct (judge 3) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in rows:
        correct = f"{r.correct}/{r.scored}" if r.scored else "-"
        lines.append(
            f"| {r.group} | {r.label} | {r.n} | {r.errors} | {r.answered} | {r.refused} | "
            f"{r.reached_answer_model} | {r.rounds_mean:.2f} | {r.two_or_more_rounds} | "
            f"{r.deadline_stops} | {r.retrieval_errors} | {r.rerank_timeouts} | "
            f"{r.cheap_calls_mean:.2f} | {r.online_calls_mean:.2f} | {r.cny_mean:.4f} | "
            f"{r.p50_s:.1f} | {r.p95_s:.1f} | {r.max_s:.1f} | {r.over_15s} | {correct} |"
        )
    return "\n".join(lines)


def question_table(
    qids: Sequence[str],
    systems: Mapping[str, Mapping[str, SystemRecord]],
    scores: Mapping[str, Mapping[str, int]],
) -> str:
    """One line per question: what each system did (status, rounds, stop reason, answer model
    reached or not, seconds, CNY, judge score, gold page in any round)."""
    head = (
        "| question | type | " + " | ".join(f"{label}" for label in SYSTEM_LABELS) + " |\n"
        "|---|---|" + "---|" * len(SYSTEM_LABELS)
    )
    lines = [head]
    for qid in qids:
        cells: list[str] = []
        qtype = ""
        for label in SYSTEM_LABELS:
            record = systems[label].get(qid)
            if record is None:
                cells.append("-")
                continue
            qtype = record.qtype
            score = scores.get(label, {}).get(qid)
            pages = {p for info in record.round_info for p in info.pages} or set(
                record.retrieved_pages
            )
            parts = [
                record.status,
                f"r{record.rounds}" + (f" {record.stop_reason}" if record.stop_reason else ""),
                "ans-model" if reached_answer_model(record) else "no ans-model",
                f"{record.total_ms / 1000:.1f}s",
                f"{record.usage.cny:.4f}",
            ]
            if score is not None:
                parts.append(f"judge {score}")
            if record.relevant_pages:
                parts.append("gold " + ("yes" if gold_hit(record, pages) else "no"))
            cells.append(", ".join(parts))
        lines.append(f"| {qid} | {qtype} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_subset(
    standard: Mapping[str, SystemRecord],
    before: Mapping[str, SystemRecord],
    after: Mapping[str, SystemRecord],
    scores: Mapping[str, Mapping[str, int]],
) -> str:
    """The Markdown report: per group and overall, then per question. `scores` has the judge's
    first-pass score by label (`SYSTEM_LABELS`) and question id (answerable questions only)."""
    qids = [q for q in after if q in before]
    by_label = {"A standard": standard, "B before": before, "B after": after}
    rows: list[Row] = []
    sections = {**groups(before, qids), "all": list(qids)}
    for name, ids in sections.items():
        for label in SYSTEM_LABELS:
            mine = [by_label[label][q] for q in ids if q in by_label[label]]
            if mine:
                rows.append(row_of(name, label, mine, scores.get(label, {})))
    verdicts = Counter(first_judge_verdict(after[q]) or "none" for q in qids)
    skipped = sum(
        int(event.data.get("skipped", 0))
        for q in qids
        for event in after[q].events
        if event.step == "judge"
    )
    out = [
        f"Questions in both runs: {len(qids)}; groups: "
        + ", ".join(f"{name} {len(ids)}" for name, ids in groups(before, qids).items()),
        "",
        table(rows),
        "",
        f"First-round verdicts of B after: {dict(verdicts)}; sub-questions taken as grounded by "
        f"their rerank score (judge skipped): {skipped}.",
        "",
        question_table(qids, by_label, scores),
    ]
    return "\n".join(out)
