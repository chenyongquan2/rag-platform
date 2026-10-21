"""The shadow report: what the semantic cache would have done (design s12-4, s14-4; spec P5-01).

Input: the JSONL records of `ShadowRecorder` (one per L1 miss with a complete result) and,
optionally, question sets with gold pages. Output: the would-hit rate, the wrong-hit rate among
the would-hits, each with its Wilson 95% interval, what the entity guard rejected, the distance
ranges, the lookup time, and the decision rule.

- would-hit rate = records with status `would_hit` / lookups (records that had a vector: a
  `no_vector` record is not a lookup).
- A would-hit is WRONG when the stored question's top chunks would have given the caller a worse
  result than the fresh search did: the top-1 chunk differs (`top1_differs`), or a gold page that
  is in the fresh result's pages is not in the stored one's (`gold_lost`, needs the question's
  gold pages). Without a label only the first test applies.
- The decision rule (design s12-4: "a hit rate too low: do not turn L2 on"; s15-1: alert when
  the upper end of the wrong-hit interval is above 1%): turn L2 on only when the upper end of
  the Wilson interval of the wrong-hit rate is at most 1% AND at least `MIN_WOULD_HIT_RATE` of
  the lookups would have hit (my threshold: below it the cache saves too little to be worth the
  risk).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.cache.semantic import question_hash
from rag.eval.stats import wilson_interval

MAX_WRONG_HIT_UPPER = 0.01
MIN_WOULD_HIT_RATE = 0.05


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Records of the shadow files; blank and broken lines are skipped."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
    return records


def load_gold(paths: Iterable[str | Path]) -> dict[str, set[str]]:
    """Gold pages by `question_hash`, from question sets (`rag.eval.questions` format)."""
    gold: dict[str, set[str]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pages = data.get("relevant_pages") or []
                if pages and data.get("question"):
                    gold.setdefault(question_hash(data["question"]), set()).update(pages)
    return gold


def wrong_reasons(record: Mapping[str, Any], gold: Mapping[str, set[str]]) -> list[str]:
    """Why a would-hit is wrong (empty: it is not)."""
    reasons: list[str] = []
    if record.get("same_top1") is False:
        reasons.append("top1_differs")
    pages = gold.get(str(record.get("question_hash")), set())
    fresh = set(record.get("fresh_top_docs") or [])
    cached = set(record.get("cached_top_docs") or [])
    if any(p in fresh and p not in cached for p in pages):
        reasons.append("gold_lost")
    return reasons


def _rate(n: int, of: int) -> dict[str, Any]:
    low, high = wilson_interval(n, of)
    return {
        "n": n,
        "of": of,
        "rate": n / of if of else None,
        "wilson95": (low, high),
    }


def _sliced(
    records: Sequence[Mapping[str, Any]], max_distance: float | None
) -> list[dict[str, Any]]:
    """The records as they would read with a threshold of `max_distance`: a would-hit farther
    away than that was a candidate that a smaller threshold would not have taken (`too_far`)."""
    if max_distance is None:
        return [dict(r) for r in records]
    out: list[dict[str, Any]] = []
    for r in records:
        if r.get("status") == "would_hit" and float(r.get("distance") or 0.0) > max_distance:
            r = {**r, "status": "too_far", "nearest_distance": r.get("distance")}
        out.append(dict(r))
    return out


def shadow_report(
    records: Sequence[Mapping[str, Any]],
    gold: Mapping[str, set[str]] | None = None,
    *,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """The report; with `max_distance` as if the semantic cache had been run with that smaller
    threshold (the records keep every candidate within the recording threshold)."""
    gold = gold or {}
    records = _sliced(records, max_distance)
    status = Counter(str(r.get("status")) for r in records)
    lookups = [r for r in records if r.get("status") != "no_vector"]
    hits = [r for r in records if r.get("status") == "would_hit"]
    would_hit = _rate(len(hits), len(lookups))

    reasons: Counter[str] = Counter()
    wrong = 0
    labelled = 0
    for hit in hits:
        found = wrong_reasons(hit, gold)
        reasons.update(found)
        wrong += bool(found)
        labelled += str(hit.get("question_hash")) in gold
    wrong_hit = {**_rate(wrong, len(hits)), "by_reason": dict(reasons), "labelled": labelled}

    rejected = [r for r in records if r.get("status") == "guard_rejected"]
    guard = {
        "rejected": len(rejected),
        "rate_of_lookups": len(rejected) / len(lookups) if lookups else None,
        "by_reason": dict(Counter(str(r.get("guard_reason")) for r in rejected)),
    }
    hit_distances = [float(r["distance"]) for r in hits if r.get("distance") is not None]
    miss_distances = [
        float(r["nearest_distance"])
        for r in records
        if r.get("status") == "too_far" and r.get("nearest_distance") is not None
    ]
    times = [float(r["lookup_ms"]) for r in records if r.get("lookup_ms") is not None]
    ordered = sorted(times)
    report: dict[str, Any] = {
        "records": len(records),
        "lookups": len(lookups),
        "status": dict(status),
        "would_hit": would_hit,
        "wrong_hit": wrong_hit,
        "guard": guard,
        "distance": {
            "would_hit_max": max(hit_distances) if hit_distances else None,
            "nearest_miss_min": min(miss_distances) if miss_distances else None,
        },
        "lookup_ms": {
            "median": statistics.median(times) if times else None,
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None,
        },
    }
    report["decision"] = _decision(would_hit, wrong_hit)
    report["max_distance"] = max_distance
    return report


def _decision(would_hit: Mapping[str, Any], wrong_hit: Mapping[str, Any]) -> dict[str, Any]:
    rate = would_hit["rate"]
    if not wrong_hit["of"]:
        return {
            "turn_on": False,
            "wrong_hit_upper": None,
            "would_hit_rate": rate,
            "why": "no would-hit in the records: nothing to judge, keep shadow mode",
        }
    upper = wrong_hit["wilson95"][1]
    if upper > MAX_WRONG_HIT_UPPER:
        why = (
            f"the upper end of the wrong-hit interval is {upper:.1%}, above "
            f"{MAX_WRONG_HIT_UPPER:.0%}: keep shadow mode"
        )
        turn_on = False
    elif rate is None or rate < MIN_WOULD_HIT_RATE:
        why = f"would-hit rate {rate:.1%} is below {MIN_WOULD_HIT_RATE:.0%}: too little to gain"
        turn_on = False
    else:
        why = "the wrong-hit interval is within 1% and the would-hit rate is worth having"
        turn_on = True
    return {"turn_on": turn_on, "wrong_hit_upper": upper, "would_hit_rate": rate, "why": why}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_report(report: Mapping[str, Any]) -> str:
    """The report as plain text."""
    hit, wrong = report["would_hit"], report["wrong_hit"]
    lines = [
        f"records {report['records']}, lookups {report['lookups']} "
        f"(a record without a question vector is not a lookup)",
        "status: " + ", ".join(f"{k} {v}" for k, v in sorted(report["status"].items())),
        f"would hit: {hit['n']} of {hit['of']} = {_pct(hit['rate'])}, "
        f"Wilson 95% {_pct(hit['wilson95'][0])} to {_pct(hit['wilson95'][1])}",
        f"wrong among would-hits: {wrong['n']} of {wrong['of']} = {_pct(wrong['rate'])}, "
        f"Wilson 95% {_pct(wrong['wilson95'][0])} to {_pct(wrong['wilson95'][1])} "
        f"({wrong['labelled']} labelled; reasons {wrong['by_reason'] or 'none'})",
        f"entity guard rejected {report['guard']['rejected']} candidates within distance: "
        f"{report['guard']['by_reason'] or 'none'}",
        f"lookup time: median {report['lookup_ms']['median']} ms, "
        f"p95 {report['lookup_ms']['p95']} ms",
        f"decision: {'turn on' if report['decision']['turn_on'] else 'keep shadow'}: "
        f"{report['decision']['why']}",
    ]
    return "\n".join(lines)
