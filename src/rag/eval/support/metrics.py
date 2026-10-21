"""Citation precision and recall, paired comparison and judge stability (spec P1-06, s11-2, s14-4).

- Citation precision: of the (sentence, cited fragment) pairs with a verdict, the share whose
  fragment supports the sentence. `partly` is not support; `precision_lenient` counts it.
- Citation recall: of the sentences with a known status, the share with at least one supporting
  cited fragment. An uncited sentence is known and not supported (level 1 already counts it).
- Unsupported-sentence rate: sentences no cited fragment supports even partly (uncited ones too).
- `unknown` (the judge failed) is left out of every denominator and counted apart.

Pairs and sentences of one answer are not independent, so their Wilson intervals are too
narrow; the per-answer rate (`all_supported`) has independent units.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from rag.eval.answer.record import Proportion
from rag.eval.stats import mcnemar_exact
from rag.eval.support.records import SupportLine


class SupportSummary(BaseModel):
    answers: int
    """Judged answers (skipped items are not answers here)."""
    skipped: int
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    pairs: int
    pairs_unknown: int
    precision: Proportion
    partly: Proportion
    precision_lenient: Proportion
    sentences: int
    sentences_unknown: int
    uncited: int
    recall: Proportion
    unsupported: Proportion
    all_supported: Proportion
    answers_unknown: int
    calls: int = 0
    cny: float = 0.0


def summarize(lines: Iterable[SupportLine]) -> SupportSummary:
    judged = [x for x in lines if x.result is not None]
    skipped = [x for x in lines if x.result is None]
    reasons: dict[str, int] = {}
    for x in skipped:
        reason = x.item.skip_reason or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    labels = [p.label for x in judged for s in x.result.sentences for p in s.pairs]  # type: ignore[union-attr]
    known = [label for label in labels if label != "unknown"]
    sentences = [s for x in judged for s in x.result.sentences]  # type: ignore[union-attr]
    statuses = [s.status for s in sentences if s.status != "unknown"]
    answers = [x.result.all_supported for x in judged]  # type: ignore[union-attr]
    known_answers = [a for a in answers if a is not None]
    return SupportSummary(
        answers=len(judged),
        skipped=len(skipped),
        skip_reasons=dict(sorted(reasons.items())),
        pairs=len(known),
        pairs_unknown=len(labels) - len(known),
        precision=Proportion.of(known.count("supported"), len(known)),
        partly=Proportion.of(known.count("partly"), len(known)),
        precision_lenient=Proportion.of(
            known.count("supported") + known.count("partly"), len(known)
        ),
        sentences=len(statuses),
        sentences_unknown=len(sentences) - len(statuses),
        uncited=sum(not s.cited for s in sentences),
        recall=Proportion.of(statuses.count("supported"), len(statuses)),
        unsupported=Proportion.of(statuses.count("not_supported"), len(statuses)),
        all_supported=Proportion.of(sum(known_answers), len(known_answers)),
        answers_unknown=len(answers) - len(known_answers),
        calls=sum(x.usage.calls for x in lines),
        cny=round(sum(x.usage.cny for x in lines), 6),
    )


def by_group(
    lines: Sequence[SupportLine],
    keys: Sequence[Callable[[SupportLine], str]] = (
        lambda x: x.system,
        lambda x: f"{x.system} / {x.qtype}",
    ),
) -> dict[str, SupportSummary]:
    """Summaries per system and per system and question type."""
    groups: dict[str, list[SupportLine]] = {}
    for key in keys:
        for line in lines:
            groups.setdefault(key(line), []).append(line)
    return {name: summarize(group) for name, group in sorted(groups.items())}


class PairedSupport(BaseModel):
    """'Every sentence supported' of the same questions under two systems (exact McNemar)."""

    n: int
    both: int
    a_only: int
    b_only: int
    neither: int
    unknown: int
    """Questions where one side is unknown (left out)."""
    unpaired: int
    """Questions judged on one side only."""
    p_value: float
    changed: list[dict[str, Any]] = Field(default_factory=list)


def paired_all_supported(a: Sequence[SupportLine], b: Sequence[SupportLine]) -> PairedSupport:
    """`b_only`: system B has every sentence supported and A does not (an improvement of B)."""
    first = {x.qid: x.result.all_supported for x in a if x.result is not None}
    second = {x.qid: x.result.all_supported for x in b if x.result is not None}
    shared = sorted(set(first) & set(second))
    both = a_only = b_only = neither = unknown = 0
    changed: list[dict[str, Any]] = []
    for qid in shared:
        x, y = first[qid], second[qid]
        if x is None or y is None:
            unknown += 1
        elif x and y:
            both += 1
        elif x:
            a_only += 1
            changed.append({"qid": qid, "a": True, "b": False})
        elif y:
            b_only += 1
            changed.append({"qid": qid, "a": False, "b": True})
        else:
            neither += 1
    return PairedSupport(
        n=both + a_only + b_only + neither,
        both=both,
        a_only=a_only,
        b_only=b_only,
        neither=neither,
        unknown=unknown,
        unpaired=len(set(first) ^ set(second)),
        p_value=mcnemar_exact(b_only, a_only),
        changed=changed,
    )


class SupportStability(BaseModel):
    """Two judge passes over the same items (design s14-3: how many verdicts move)."""

    pairs: Proportion
    """Pairs whose label differs, of the pairs known in both passes."""
    pairs_unknown: int
    sentences: Proportion
    """Sentences whose status differs, of the sentences known in both passes."""
    answers: Proportion
    """Answers whose 'every sentence supported' differs, of those known in both."""
    changed: list[dict[str, Any]] = Field(default_factory=list)


def stability(first: Sequence[SupportLine], second: Sequence[SupportLine]) -> SupportStability:
    one = {(x.system, x.qid): x for x in first if x.result is not None}
    two = {(x.system, x.qid): x for x in second if x.result is not None}
    pair_n = pair_flips = pair_unknown = 0
    sent_n = sent_flips = 0
    ans_n = ans_flips = 0
    changed: list[dict[str, Any]] = []
    for key in sorted(set(one) & set(two)):
        r1, r2 = one[key].result, two[key].result
        assert r1 is not None and r2 is not None
        labels2 = {(p.sentence, p.chunk_id): p.label for s in r2.sentences for p in s.pairs}
        for s in r1.sentences:
            for p in s.pairs:
                other = labels2.get((p.sentence, p.chunk_id), "unknown")
                if "unknown" in (p.label, other):
                    pair_unknown += 1
                    continue
                pair_n += 1
                if p.label != other:
                    pair_flips += 1
                    changed.append(
                        {
                            "system": key[0],
                            "qid": key[1],
                            "sentence": p.sentence,
                            "chunk_id": p.chunk_id,
                            "run1": p.label,
                            "run2": other,
                        }
                    )
        status2 = {s.index: s.status for s in r2.sentences}
        for s in r1.sentences:
            other_status = status2.get(s.index, "unknown")
            if "unknown" in (s.status, other_status):
                continue
            sent_n += 1
            sent_flips += s.status != other_status
        if r1.all_supported is not None and r2.all_supported is not None:
            ans_n += 1
            ans_flips += r1.all_supported != r2.all_supported
    return SupportStability(
        pairs=Proportion.of(pair_flips, pair_n),
        pairs_unknown=pair_unknown,
        sentences=Proportion.of(sent_flips, sent_n),
        answers=Proportion.of(ans_flips, ans_n),
        changed=changed,
    )


__all__ = [
    "PairedSupport",
    "SupportStability",
    "SupportSummary",
    "by_group",
    "paired_all_supported",
    "stability",
    "summarize",
]
