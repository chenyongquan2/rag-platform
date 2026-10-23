"""The numbers of the Agentic RAG comparison (spec P3-05, design s13-5, s14-4). Pure functions of
the records and the judge lines: no model, no search, no network.

Fixed before the run (see the spec):

- an answerable question is *correct* when the judge's 1-3 score is 3 (adequate: 2 or 3); the mean
  score is the HF percentage `(score - 1) / 2`;
- a no-answer question is *refused* when no answer was given (`answered` false): not found, a parse
  failure, an error, or the graph asking back (counted apart as a clarification); it is *refused
  cleanly* when, besides, the run did not fail and the answer was not cut off by the total deadline
  (spec EVAL-0: such a record says `not_found` but the answer step never ran, so it is counted apart
  as "truncated", not as a refusal the system chose).

Paired tests over the questions both systems have: exact McNemar on the binary metrics, a
sign-flip permutation test on the scores, Wilson intervals on every rate, and the smallest
difference `n` questions could show. A difference that is not significant is reported as
"direction only".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from rag.eval.agentic.records import JudgeLine, SystemName, SystemRecord, is_deadline_truncated
from rag.eval.agentic.sets import NOT_ANSWERABLE, QTYPES
from rag.eval.answer.record import Proportion
from rag.eval.hard.clarify import asked_back
from rag.eval.retrieval.runner import latency_summary
from rag.eval.stats import (
    mcnemar_exact,
    min_detectable_difference,
    paired_changes,
    paired_permutation_test,
)

CORRECT = 3
ADEQUATE = 2
ALPHA = 0.05
DEADLINE_MS = 30_000
TARGET_P95_MS = 15_000
ANSWERABLE = "answerable"
"""Key of the answerable types pooled (the three of P3-05, plus `hard` when it was run)."""
ANSWERABLE_TYPES = tuple(t for t in QTYPES if t not in NOT_ANSWERABLE)


# ---- helpers ------------------------------------------------------------------------------------


def gold_hit(record: SystemRecord, pages: Iterable[str]) -> bool:
    """Is the question's gold page among `pages`? A two-hop question needs all its pages."""
    found = set(pages)
    wanted = record.relevant_pages
    if not wanted:
        return False
    return (
        all(p in found for p in wanted)
        if record.requires == "all"
        else any(p in found for p in wanted)
    )


def round_pages(record: SystemRecord, round_no: int) -> list[str]:
    return list(dict.fromkeys(p for r in record.round_info if r.round == round_no for p in r.pages))


def by_system(records: Iterable[SystemRecord]) -> dict[SystemName, dict[str, SystemRecord]]:
    out: dict[SystemName, dict[str, SystemRecord]] = {"standard": {}, "agentic": {}}
    for r in records:
        out[r.system][r.qid] = r
    return out


def judge_scores(lines: Iterable[JudgeLine], system: str, run: int = 1) -> dict[str, int]:
    """Score per question of one judge pass (a line without a score is left out)."""
    return {
        x.qid: x.score for x in lines if x.system == system and x.run == run and x.score is not None
    }


def _failed(record: SystemRecord) -> bool:
    """The run ended in an error: no answer because something broke, not because it refused."""
    return (
        record.status == "error"
        or record.generation_error is not None
        or record.answer_parse_failed
    )


def clean_refusal(record: SystemRecord) -> bool:
    """No answer was given and nothing broke or ran out of time on the way: the system decided to
    refuse. A deadline-cut answer (`is_deadline_truncated`) is not one (spec EVAL-0)."""
    return not record.answered and not _failed(record) and not is_deadline_truncated(record)


def _in_types(record: SystemRecord, qtype: str) -> bool:
    return record.answerable if qtype == ANSWERABLE else record.qtype == qtype


# ---- paired tests -------------------------------------------------------------------------------


class PairedBinary(BaseModel):
    """A binary metric of system A and system B on the same questions."""

    metric: str
    n: int
    a: Proportion
    b: Proportion
    improved: int
    """B yes, A no."""
    worsened: int
    """A yes, B no."""
    both: int
    neither: int
    p_value: float
    """Exact two-sided McNemar."""
    difference: float
    """Rate of B minus rate of A."""
    min_detectable: float | None
    """The smallest net change `n` questions can show at p <= 0.05 (best case: nothing worse)."""
    verdict: str


def paired_binary(metric: str, a: Mapping[str, bool], b: Mapping[str, bool]) -> PairedBinary:
    qids = sorted(set(a) & set(b))
    improved = sum(1 for q in qids if b[q] and not a[q])
    worsened = sum(1 for q in qids if a[q] and not b[q])
    both = sum(1 for q in qids if a[q] and b[q])
    n = len(qids)
    a_count = sum(1 for q in qids if a[q])
    b_count = sum(1 for q in qids if b[q])
    p = mcnemar_exact(improved, worsened)
    difference = (b_count - a_count) / n if n else 0.0
    if p <= ALPHA:
        verdict = "B higher (p <= 0.05)" if difference > 0 else "B lower (p <= 0.05)"
    elif improved == worsened:
        verdict = "no difference"
    else:
        direction = "B higher" if improved > worsened else "B lower"
        verdict = f"no reliable difference (direction: {direction})"
    smallest = min_detectable_difference(n) if n else None
    return PairedBinary(
        metric=metric,
        n=n,
        a=Proportion.of(a_count, n),
        b=Proportion.of(b_count, n),
        improved=improved,
        worsened=worsened,
        both=both,
        neither=n - improved - worsened - both,
        p_value=p,
        difference=difference,
        min_detectable=smallest.difference if smallest else None,
        verdict=verdict,
    )


class PairedScores(BaseModel):
    n: int
    a_pct: float
    b_pct: float
    diff_pct: float
    """HF percentage `(score - 1) / 2 * 100`, mean over the questions; B minus A."""
    improved: int
    worsened: int
    unchanged: int
    p_value: float
    """Paired sign-flip permutation test of the mean difference."""
    exact: bool
    iterations: int


def paired_scores(a: Mapping[str, int], b: Mapping[str, int]) -> PairedScores:
    qids = sorted(set(a) & set(b))
    xa = [(a[q] - 1) / 2 for q in qids]
    xb = [(b[q] - 1) / 2 for q in qids]
    changes = paired_changes(xa, xb)
    test = paired_permutation_test(xa, xb)
    n = len(qids)
    return PairedScores(
        n=n,
        a_pct=100.0 * sum(xa) / n if n else 0.0,
        b_pct=100.0 * sum(xb) / n if n else 0.0,
        diff_pct=100.0 * test.mean_diff,
        improved=changes.improved,
        worsened=changes.worsened,
        unchanged=changes.unchanged,
        p_value=test.p_value,
        exact=test.exact,
        iterations=test.iterations,
    )


class TypeSummary(BaseModel):
    qtype: str
    n: int
    """Questions compared (answerable: scored by the judge for both systems)."""
    unscored: int = 0
    """Answerable questions both systems have but the judge left without a score for one."""
    correct: PairedBinary | None = None
    adequate: PairedBinary | None = None
    score: PairedScores | None = None
    refusal: PairedBinary | None = None
    clean_refusal: PairedBinary | None = None
    """Refused without a crash or a deadline cut: the refusals of `refusal` minus the runs that
    ended in an error and the answers cut off by the total deadline (`truncated`)."""
    errors: dict[str, int] = Field(default_factory=dict)
    """No-answer questions whose run ended in an error (retrieval error, gateway timeout, parse
    failure): they gave no answer, so `refusal` counts them, and `clean_refusal` does not."""
    clarifications: dict[str, int] = Field(default_factory=dict)
    """No-answer or ambiguous questions where the graph asked back (counted as refusals)."""
    truncated: dict[str, int] = Field(default_factory=dict)
    """Questions of this type (answerable or not) whose answer was cut off by the total deadline
    (`is_deadline_truncated`), per system; listed apart, never a clean refusal (spec EVAL-0)."""


def per_type_summary(
    records: Sequence[SystemRecord], lines: Sequence[JudgeLine], *, run: int = 1
) -> dict[str, TypeSummary]:
    """One summary per question type present, plus `answerable` (the three answerable types)."""
    systems = by_system(records)
    scores = {s: judge_scores(lines, s, run) for s in ("standard", "agentic")}
    out: dict[str, TypeSummary] = {}
    for qtype in (*QTYPES, ANSWERABLE):
        common = [
            q
            for q, r in systems["standard"].items()
            if q in systems["agentic"] and _in_types(r, qtype)
        ]
        if not common:
            continue
        truncated = {
            s: sum(1 for q in common if is_deadline_truncated(systems[s][q]))  # type: ignore[index]
            for s in ("standard", "agentic")
        }
        if qtype == "no_answer":
            refused = {
                s: {q: not systems[s][q].answered for q in common}  # type: ignore[index]
                for s in ("standard", "agentic")
            }
            clean = {
                s: {q: clean_refusal(systems[s][q]) for q in common}  # type: ignore[index]
                for s in ("standard", "agentic")
            }
            out[qtype] = TypeSummary(
                qtype=qtype,
                n=len(common),
                refusal=paired_binary("refused", refused["standard"], refused["agentic"]),
                clean_refusal=paired_binary("refused cleanly", clean["standard"], clean["agentic"]),
                errors={
                    s: sum(1 for q in common if _failed(systems[s][q]))  # type: ignore[index]
                    for s in ("standard", "agentic")
                },
                clarifications={
                    s: sum(1 for q in common if systems[s][q].clarification)  # type: ignore[index]
                    for s in ("standard", "agentic")
                },
                truncated=truncated,
            )
            continue
        if qtype == "ambiguous":  # no reference answer, no judge: the rule check counts asking back
            out[qtype] = TypeSummary(
                qtype=qtype,
                n=len(common),
                clarifications={
                    s: sum(1 for q in common if asked_back(systems[s][q]))  # type: ignore[index]
                    for s in ("standard", "agentic")
                },
                truncated=truncated,
            )
            continue
        scored = [q for q in common if q in scores["standard"] and q in scores["agentic"]]
        a = {q: scores["standard"][q] for q in scored}
        b = {q: scores["agentic"][q] for q in scored}
        summary = TypeSummary(
            qtype=qtype, n=len(scored), unscored=len(common) - len(scored), truncated=truncated
        )
        if scored:
            summary.correct = paired_binary(
                "correct",
                {q: s >= CORRECT for q, s in a.items()},
                {q: s >= CORRECT for q, s in b.items()},
            )
            summary.adequate = paired_binary(
                "adequate",
                {q: s >= ADEQUATE for q, s in a.items()},
                {q: s >= ADEQUATE for q, s in b.items()},
            )
            summary.score = paired_scores(a, b)
        out[qtype] = summary
    return out


class MinDetectRow(BaseModel):
    name: str
    n: int
    best_case: float | None
    """Smallest net difference of a binary metric that reaches p <= 0.05 when nothing got worse."""
    with_two_worse: float | None
    """The same when two questions got worse."""


def min_detectable_table(sizes: Mapping[str, int]) -> list[MinDetectRow]:
    rows = []
    for name, n in sizes.items():
        best = min_detectable_difference(n)
        two = min_detectable_difference(n, worsened=2) if n > 2 else None
        rows.append(
            MinDetectRow(
                name=name,
                n=n,
                best_case=best.difference if best else None,
                with_two_worse=two.difference if two else None,
            )
        )
    return rows


# ---- the loop inside the graph ------------------------------------------------------------------


class LoopTypeStats(BaseModel):
    """What the graph did on the questions of one type."""

    n: int
    clarifications: int = 0
    rounds: dict[int, int] = Field(default_factory=dict)
    """Questions by number of retrieval rounds (0: none, e.g. a clarification)."""
    ran_round2: int = 0
    ran_round3: int = 0
    first_verdicts: dict[str, int] = Field(default_factory=dict)
    """The judge's verdict after round one."""
    final_sufficiency: dict[str, int] = Field(default_factory=dict)
    """The judge's last verdict of each question."""
    searched_answerable: int = 0
    """Answerable questions that were searched at least once (the base of the next counts)."""
    hit_round1: int = 0
    hit_all_rounds: int = 0
    hit_context: int = 0
    round1_missed: int = 0
    rescued: int = 0
    """Gold page not in round one, retrieved in a later round."""
    rescued_into_context: int = 0
    lost_to_context: int = 0
    """Gold page retrieved in some round but not among the fragments the answer was written from."""


class JudgeLabel(BaseModel):
    """The in-graph judge's round-one verdict against the label "gold page in round one"
    (P3-03's label). Positive = the judge said `enough`."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    accuracy: Proportion = Field(default_factory=lambda: Proportion.of(0, 0))


class LoopReport(BaseModel):
    by_type: dict[str, LoopTypeStats]
    judge_vs_label: JudgeLabel
    false_clarification: dict[str, Proportion]
    """Clarifications among the graph's answerable questions, by type."""
    clarified_questions: list[tuple[str, str]]
    """(qid, question) of answerable questions the graph asked back about."""


def _first_verdict(record: SystemRecord) -> str | None:
    for e in record.events:
        if e.step == "judge":
            return str(e.data.get("verdict"))
    return None


def _last_verdict(record: SystemRecord) -> str | None:
    verdicts = [str(e.data.get("verdict")) for e in record.events if e.step == "judge"]
    return verdicts[-1] if verdicts else None


def loop_report(records: Sequence[SystemRecord]) -> LoopReport:
    agentic = [r for r in records if r.system == "agentic"]
    by_type: dict[str, LoopTypeStats] = {}
    label = JudgeLabel()
    for r in agentic:
        label_hit = gold_hit(r, round_pages(r, 1))
        verdict = _first_verdict(r)
        if verdict is not None:
            stop = verdict == "enough"
            if stop and label_hit:
                label.tp += 1
            elif stop:
                label.fp += 1
            elif label_hit:
                label.fn += 1
            else:
                label.tn += 1
    label.accuracy = Proportion.of(label.tp + label.tn, label.tp + label.fp + label.fn + label.tn)
    for qtype in (*QTYPES, ANSWERABLE):
        group = [r for r in agentic if _in_types(r, qtype)]
        if not group:
            continue
        s = LoopTypeStats(n=len(group))
        rounds: Counter[int] = Counter()
        first: Counter[str] = Counter()
        last: Counter[str] = Counter()
        for r in group:
            s.clarifications += int(r.clarification)
            rounds[r.rounds] += 1
            s.ran_round2 += int(r.rounds >= 2)
            s.ran_round3 += int(r.rounds >= 3)
            if (v := _first_verdict(r)) is not None:
                first[v] += 1
            if (v := _last_verdict(r)) is not None:
                last[v] += 1
            if not (r.answerable and r.round_info):
                continue
            s.searched_answerable += 1
            hit1 = gold_hit(r, round_pages(r, 1))
            hit_all = gold_hit(r, r.retrieved_pages)
            hit_ctx = gold_hit(r, r.context_pages)
            s.hit_round1 += int(hit1)
            s.hit_all_rounds += int(hit_all)
            s.hit_context += int(hit_ctx)
            if not hit1:
                s.round1_missed += 1
                s.rescued += int(hit_all)
                s.rescued_into_context += int(hit_ctx)
            s.lost_to_context += int(hit_all and not hit_ctx)
        s.rounds = dict(sorted(rounds.items()))
        s.first_verdicts = dict(first)
        s.final_sufficiency = dict(last)
        by_type[qtype] = s
    false_clar = {
        qtype: Proportion.of(
            sum(int(r.clarification) for r in agentic if _in_types(r, qtype)),
            sum(1 for r in agentic if _in_types(r, qtype)),
        )
        for qtype in (*ANSWERABLE_TYPES, ANSWERABLE)
        if any(_in_types(r, qtype) for r in agentic)
    }
    return LoopReport(
        by_type=by_type,
        judge_vs_label=label,
        false_clarification=false_clar,
        clarified_questions=[
            (r.qid, r.question) for r in agentic if r.answerable and r.clarification
        ],
    )


def rewrite_effect(records: Sequence[SystemRecord]) -> dict[str, PairedBinary]:
    """Gold page among the chunks of the raw question (standard RAG's one search) against the
    chunks of round one after the graph's rewrite, on the questions the graph searched."""
    systems = by_system(records)
    out: dict[str, PairedBinary] = {}
    for qtype in (*ANSWERABLE_TYPES, ANSWERABLE):
        qids = [
            q
            for q, r in systems["agentic"].items()
            if _in_types(r, qtype) and r.round_info and q in systems["standard"]
        ]
        if not qids:
            continue
        raw = {
            q: gold_hit(systems["standard"][q], systems["standard"][q].retrieved_pages)
            for q in qids
        }
        rewritten = {
            q: gold_hit(systems["agentic"][q], round_pages(systems["agentic"][q], 1)) for q in qids
        }
        out[qtype] = paired_binary("gold page retrieved", raw, rewritten)
    return out


# ---- cost, latency, limits ----------------------------------------------------------------------


class CostRow(BaseModel):
    system: str
    qtype: str
    n: int
    mean_chat_calls: float
    mean_cheap_calls: float
    mean_online_calls: float
    mean_embedding_calls: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_cny: float
    total_cny: float
    mean_rounds: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float
    over_15s: int
    over_30s: int
    mean_steps_ms: dict[str, float] = Field(default_factory=dict)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _calls(record: SystemRecord, table: str, key: str) -> int:
    usage = (record.usage_by_kind if table == "kind" else record.usage_by_profile).get(key)
    return usage.calls if usage else 0


def cost_latency(records: Sequence[SystemRecord]) -> list[CostRow]:
    rows: list[CostRow] = []
    for system in ("standard", "agentic"):
        mine = [r for r in records if r.system == system]
        for qtype in (*QTYPES, "all"):
            group = mine if qtype == "all" else [r for r in mine if r.qtype == qtype]
            if not group:
                continue
            lat = latency_summary([r.total_ms for r in group])
            steps: dict[str, list[float]] = {}
            for r in group:
                for k, v in r.steps_ms.items():
                    steps.setdefault(k, []).append(v)
            rows.append(
                CostRow(
                    system=system,
                    qtype=qtype,
                    n=len(group),
                    mean_chat_calls=_mean([_calls(r, "kind", "chat") for r in group]),
                    mean_cheap_calls=_mean([_calls(r, "profile", "cheap") for r in group]),
                    mean_online_calls=_mean([_calls(r, "profile", "online") for r in group]),
                    mean_embedding_calls=_mean([_calls(r, "kind", "embedding") for r in group]),
                    mean_prompt_tokens=_mean([r.usage.prompt_tokens for r in group]),
                    mean_completion_tokens=_mean([r.usage.completion_tokens for r in group]),
                    mean_cny=_mean([r.usage.cny for r in group]),
                    total_cny=sum(r.usage.cny for r in group),
                    mean_rounds=_mean([r.rounds for r in group]),
                    p50_ms=lat.p50_ms,
                    p95_ms=lat.p95_ms,
                    mean_ms=lat.mean_ms,
                    max_ms=lat.max_ms,
                    over_15s=sum(1 for r in group if r.total_ms > TARGET_P95_MS),
                    over_30s=sum(1 for r in group if r.total_ms > DEADLINE_MS),
                    mean_steps_ms={k: _mean(v) for k, v in steps.items()},
                )
            )
    return rows


class LimitCase(BaseModel):
    qid: str
    qtype: str
    rounds: int
    stop_reason: str | None
    total_ms: float
    score: int | None
    """The judge's first-pass score (answerable questions)."""
    answered: bool
    truncated: bool = False
    """The answer was cut off by the deadline (no answer step ran; spec EVAL-0)."""


class LimitsReport(BaseModel):
    stop_reasons: dict[str, int]
    coverage_unconfirmed: int
    degraded: int
    rerank_timeout: int
    fallbacks: dict[str, int]
    error_categories: dict[str, int]
    cases: list[LimitCase]
    """Graph runs that stopped at a limit (a stop reason or `coverage_unconfirmed`)."""
    limited_answerable: int
    limited_correct: int
    over_deadline_ms: int
    """Runs that took longer than the 30 s default total deadline."""
    truncated: list[LimitCase] = Field(default_factory=list)
    """Graph runs whose answer was cut off by the deadline (`is_deadline_truncated`), listed apart
    from refusals (spec EVAL-0)."""


def limits_report(
    records: Sequence[SystemRecord], lines: Sequence[JudgeLine], *, run: int = 1
) -> LimitsReport:
    agentic = [r for r in records if r.system == "agentic"]
    scores = judge_scores(lines, "agentic", run)
    cases = [
        LimitCase(
            qid=r.qid,
            qtype=r.qtype,
            rounds=r.rounds,
            stop_reason=r.stop_reason,
            total_ms=r.total_ms,
            score=scores.get(r.qid) if r.answerable else None,
            answered=r.answered,
            truncated=is_deadline_truncated(r),
        )
        for r in agentic
        if r.stop_reason or r.coverage_unconfirmed
    ]
    limited = [c for c in cases if c.score is not None]
    return LimitsReport(
        stop_reasons=dict(Counter(r.stop_reason for r in agentic if r.stop_reason)),
        coverage_unconfirmed=sum(int(r.coverage_unconfirmed) for r in agentic),
        degraded=sum(int(r.degraded) for r in agentic),
        rerank_timeout=sum(int(r.rerank_timeout) for r in agentic),
        fallbacks=dict(Counter(f for r in agentic for f in r.fallbacks)),
        error_categories=dict(Counter(c for r in agentic for c in r.error_categories)),
        cases=cases,
        limited_answerable=len(limited),
        limited_correct=sum(1 for c in limited if (c.score or 0) >= CORRECT),
        over_deadline_ms=sum(1 for r in agentic if r.total_ms > DEADLINE_MS),
        truncated=[c for c in cases if c.truncated],
    )


# ---- judge stability ----------------------------------------------------------------------------


class JudgeStability(BaseModel):
    system: str
    n: int
    """Questions the judge model scored in both passes."""
    n_rule_excluded: int
    n_missing: int
    score_flips: Proportion
    correct_flips: Proportion
    mean_abs_diff: float | None
    max_abs_diff: int | None


def judge_stability(lines: Sequence[JudgeLine], system: str) -> JudgeStability:
    first = {x.qid: x for x in lines if x.system == system and x.run == 1}
    second = {x.qid: x for x in lines if x.system == system and x.run == 2}
    pairs: list[tuple[int, int]] = []
    rule = missing = 0
    for qid, x in first.items():
        y = second.get(qid)
        if y is None or x.score is None or y.score is None:
            missing += 1
        elif x.by_rule or y.by_rule:
            rule += 1
        else:
            pairs.append((x.score, y.score))
    diffs = [abs(x - y) for x, y in pairs]
    return JudgeStability(
        system=system,
        n=len(pairs),
        n_rule_excluded=rule,
        n_missing=missing,
        score_flips=Proportion.of(sum(1 for x, y in pairs if x != y), len(pairs)),
        correct_flips=Proportion.of(
            sum(1 for x, y in pairs if (x >= CORRECT) != (y >= CORRECT)), len(pairs)
        ),
        mean_abs_diff=_mean(diffs) if diffs else None,
        max_abs_diff=max(diffs) if diffs else None,
    )
