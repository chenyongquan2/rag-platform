"""Labels and numbers of the judge-accuracy report (spec P3-03; design s13-4, s14-4). Pure
functions of the records: no model, no search, no network.

The label (design s13-4): a question is "enough on round one" when its gold page is already in
what round one returned, so no human has to mark it. Questions that name several pages treat
them as alternatives (the same text on two paths), unless `requires` is `all` (a two-hop question
needs both pages). A no-answer question is never enough.

Three policies are compared on the same records (each says, per question, "round one was enough,
stop" or "not enough, run another round"):

    judge             the verdict is `enough` (rules, then the small model)
    rules_only        some chunk passed the rerank threshold (no model)
    fixed_two_rounds  never stop after round one: always run a second round (the fallback the
                      design prefers when the judge is too inaccurate)

A wrong "stop" (`missed_corrections`: label not enough, policy enough) skips a round that was
needed; a wrong "go on" (`wasted_rounds`: label enough, policy not enough) pays for a round that
was not. Rounds are a range: every question is searched once, and a question the policy sends on
runs at least one more round and, in the real loop (judged again after every round), up to
`max_rounds`; the fixed policy runs exactly two. `fixed_two_rounds` reports no accuracy: a
policy that never stops scores exactly the share of "not enough" labels by construction, so the
number says nothing about judging; compare policies on wasted rounds, missed corrections and
rounds, and the judge with the rules alone by the paired test (`paired_policies`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag.eval.judge_accuracy.records import JudgeFile, JudgeRecord
from rag.eval.questions import LabeledQuestion
from rag.eval.stats import mcnemar_exact, wilson_interval

Predict = Callable[[JudgeRecord], bool]
"""True: the policy stops after round one (it calls what was found enough)."""

POLICIES: dict[str, Predict] = {
    "judge": lambda r: r.verdict == "enough",
    "rules_only": lambda r: r.n_chunks > 0,
    "fixed_two_rounds": lambda r: False,
}


def label_enough(question: LabeledQuestion, returned_pages: Sequence[str]) -> bool:
    """Is the question's gold page among the pages round one returned?"""
    if not question.relevant_pages:
        return False
    found = set(returned_pages)
    wanted = question.relevant_pages
    if question.meta.get("requires") == "all":
        return all(page in found for page in wanted)
    return any(page in found for page in wanted)


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    value: float
    low: float
    high: float


def rate(successes: int, n: int) -> Rate:
    """`successes` of `n` with its Wilson 95% interval ((0, 1) when `n` is 0)."""
    low, high = wilson_interval(successes, n)
    return Rate(successes, n, successes / n if n else 0.0, low, high)


@dataclass(frozen=True)
class Confusion:
    """Positive = the policy calls round one enough."""

    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class Summary:
    n: int
    confusion: Confusion
    accuracy: Rate
    precision: Rate | None
    """Of the questions it called enough, the share that was. None when it never says enough."""
    recall: Rate | None
    """Of the questions that were enough, the share it called so. None without any."""
    specificity: Rate | None
    """Of the questions that were not enough, the share it sent on to another round."""


def confusion(records: Sequence[JudgeRecord], predict: Predict) -> Confusion:
    tp = fp = fn = tn = 0
    for r in records:
        said = predict(r)
        if r.label_enough:
            tp, fn = (tp + 1, fn) if said else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if said else (fp, tn + 1)
    return Confusion(tp, fp, fn, tn)


def summarize(records: Sequence[JudgeRecord], predict: Predict) -> Summary:
    c = confusion(records, predict)
    n = len(records)
    return Summary(
        n=n,
        confusion=c,
        accuracy=rate(c.tp + c.tn, n),
        precision=rate(c.tp, c.tp + c.fp) if c.tp + c.fp else None,
        recall=rate(c.tp, c.tp + c.fn) if c.tp + c.fn else None,
        specificity=rate(c.tn, c.tn + c.fp) if c.tn + c.fp else None,
    )


def by_type(records: Sequence[JudgeRecord], predict: Predict) -> dict[str, Summary]:
    """One summary per question type, in order of first appearance."""
    types = list(dict.fromkeys(r.qtype for r in records))
    return {t: summarize([r for r in records if r.qtype == t], predict) for t in types}


def by_set(records: Sequence[JudgeRecord], predict: Predict) -> dict[str, Summary]:
    names = list(dict.fromkeys(r.set for r in records))
    return {s: summarize([r for r in records if r.set == s], predict) for s in names}


def tier_table(records: Sequence[JudgeRecord]) -> dict[str, dict[str, int]]:
    """The judge's three verdicts against the label: how many questions of each tier were
    enough on round one and how many were not."""
    table = {tier: {"enough_label": 0, "not_enough_label": 0} for tier in TIERS}
    for r in records:
        table[r.verdict]["enough_label" if r.label_enough else "not_enough_label"] += 1
    return table


TIERS = ("enough", "partly_enough", "not_enough")


@dataclass(frozen=True)
class PolicyRow:
    policy: str
    accuracy: Rate | None
    """None for a policy that never stops after round one: it scores exactly the share of
    "not enough" labels, which says nothing about judging."""
    wasted_rounds: int
    missed_corrections: int
    rounds_min: int
    rounds_max: int
    """Retrieval rounds in all. A question a policy sends on runs round two at least and, in
    the real loop (the judge decides again after each round), up to `max_rounds`; the fixed
    policy runs exactly two."""


def policy_rows(records: Sequence[JudgeRecord], *, max_rounds: int = 3) -> list[PolicyRow]:
    rows = []
    n = len(records)
    for name, predict in POLICIES.items():
        c = confusion(records, predict)
        sent_on = n - (c.tp + c.fp)
        fixed = name == "fixed_two_rounds"
        rows.append(
            PolicyRow(
                policy=name,
                accuracy=None if fixed else rate(c.tp + c.tn, n),
                wasted_rounds=c.fn,
                missed_corrections=c.fp,
                rounds_min=n + sent_on,
                rounds_max=n + sent_on * (1 if fixed else max(1, max_rounds - 1)),
            )
        )
    return rows


@dataclass(frozen=True)
class PairedPolicies:
    """Two policies on the same questions: where one is right and the other wrong."""

    a: str
    b: str
    a_only: int
    b_only: int
    both_right: int
    both_wrong: int
    p_value: float
    """Exact McNemar test of `a_only` against `b_only` (design s14-4)."""


def paired_policies(records: Sequence[JudgeRecord], a: str, b: str) -> PairedPolicies:
    right_a = [POLICIES[a](r) == r.label_enough for r in records]
    right_b = [POLICIES[b](r) == r.label_enough for r in records]
    a_only = sum(1 for x, y in zip(right_a, right_b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(right_a, right_b, strict=True) if y and not x)
    both = sum(1 for x, y in zip(right_a, right_b, strict=True) if x and y)
    return PairedPolicies(
        a,
        b,
        a_only,
        b_only,
        both,
        len(records) - a_only - b_only - both,
        mcnemar_exact(a_only, b_only),
    )


@dataclass(frozen=True)
class RescueSummary:
    """Among the answerable questions that were not enough on round one: how many a later round
    found (cumulative: round three counts what round two found)."""

    n: int
    round2: Rate
    round3: Rate


def rescue_summary(records: Sequence[JudgeRecord]) -> RescueSummary | None:
    done = [r.rescue for r in records if r.rescue is not None]
    if not done:
        return None
    round2 = sum(1 for d in done if d.round2_found)
    round3 = sum(1 for d in done if d.round2_found or d.round3_found)
    return RescueSummary(len(done), rate(round2, len(done)), rate(round3, len(done)))


# ---- report text --------------------------------------------------------------------------------


def _pct(r: Rate | None) -> str:
    if r is None:
        return "n/a"
    return f"{r.value * 100:.1f}% ({r.successes}/{r.n}; {r.low * 100:.1f}-{r.high * 100:.1f})"


def _rounds(row: PolicyRow) -> str:
    if row.rounds_min == row.rounds_max:
        return str(row.rounds_min)
    return f"{row.rounds_min} to {row.rounds_max}"


def _paired_line(p: PairedPolicies) -> str:
    return (
        f"{p.a} right and {p.b} wrong on {p.a_only} questions, the reverse on {p.b_only}; both "
        f"right on {p.both_right}, both wrong on {p.both_wrong}. Exact McNemar test (design "
        f"s14-4): p = {p.p_value:.2g}."
    )


def _summary_row(name: str, s: Summary) -> str:
    c = s.confusion
    return (
        f"| {name} | {s.n} | {_pct(s.accuracy)} | {_pct(s.precision)} | {_pct(s.recall)} "
        f"| {c.tp} / {c.fp} / {c.fn} / {c.tn} |"
    )


_HEADER = (
    "| {first} | questions | accuracy | precision of enough | recall of enough "
    "| TP / FP / FN / TN |\n|---|---|---|---|---|---|"
)


def render_report(file: JudgeFile) -> str:
    """The Markdown report of a record file (all numbers recomputed from the rows)."""
    records = file.records
    judge = POLICIES["judge"]
    lines = [
        "# Judge accuracy",
        "",
        f"kb {file.kb}; rerank threshold {file.threshold}; k {file.k}; judge profile "
        f"{file.judge_profile or 'n/a'}; {len(records)} questions"
        + ("; PARTIAL (stopped at the spend limit)" if file.partial else "")
        + f"; spent {file.spent_cny:.4f} CNY.",
        "",
        "Positive = the judge says round one is enough. Label (from the question sets, no human): "
        "the gold page is among the chunks round one returned; a no-answer question is never "
        "enough. Intervals are Wilson 95%.",
        "",
        "## The judge",
        "",
        _HEADER.format(first="all"),
        _summary_row("all", summarize(records, judge)),
        "",
        "## By question set",
        "",
        _HEADER.format(first="set"),
        *(_summary_row(name, s) for name, s in by_set(records, judge).items()),
        "",
        "## By question type",
        "",
        _HEADER.format(first="type"),
        *(_summary_row(name, s) for name, s in by_type(records, judge).items()),
        "",
        "## The three tiers against the label",
        "",
        "| verdict | enough on round one | not enough on round one |",
        "|---|---|---|",
        *(
            f"| {tier} | {row['enough_label']} | {row['not_enough_label']} |"
            for tier, row in tier_table(records).items()
        ),
        "",
        "## The judge against the fallbacks",
        "",
        "`wasted rounds`: round one was enough but the policy went on; `missed corrections`: round "
        "one was not enough but the policy stopped; `rounds`: retrieval rounds run in all, as a "
        f"range (a question sent on runs round two at least, up to {file.max_rounds} rounds in "
        "the real loop). `fixed_two_rounds` never stops, so by construction its accuracy is the "
        f"label share of 'not enough' ({sum(1 for r in records if not r.label_enough)}/"
        f"{len(records)}); it is not reported as an accuracy.",
        "",
        "| policy | accuracy | wasted rounds | missed corrections | rounds |",
        "|---|---|---|---|---|",
        *(
            f"| {row.policy} | {_pct(row.accuracy)} | {row.wasted_rounds} "
            f"| {row.missed_corrections} | {_rounds(row)} |"
            for row in policy_rows(records, max_rounds=file.max_rounds)
        ),
        "",
        "## The judge against the rules alone (paired, same questions)",
        "",
        _paired_line(paired_policies(records, "judge", "rules_only")),
    ]
    sensitivity = sum(1 for r in records if r.label_in_rerank_top_k != r.label_enough)
    lines += [
        "",
        f"Without the threshold {sensitivity} labels would differ (gold page reranked in the top "
        f"{file.k} but under the threshold).",
    ]
    rescue = rescue_summary(records)
    if rescue is not None:
        lines += [
            "",
            "## Would another round have found the page?",
            "",
            f"Of {rescue.n} answerable questions that were not enough on round one: round two "
            f"(several phrasings) found the gold page for {_pct(rescue.round2)}; rounds two and "
            f"three together (then a HyDE passage) for {_pct(rescue.round3)}.",
        ]
    wrong = [r for r in records if judge(r) != r.label_enough]
    lines += [
        "",
        "## Where the judge and the label disagree",
        "",
        *(
            f"- {r.set}/{r.qid} ({r.qtype}): judge {r.verdict}"
            f"{' [' + r.fallback + ']' if r.fallback else ''}, label "
            f"{'enough' if r.label_enough else 'not enough'}"
            for r in wrong
        ),
    ]
    if not wrong:
        lines.append("- none")
    return "\n".join(lines) + "\n"
