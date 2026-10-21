"""Evaluate the semantic cache's guard on the counter-example pairs (design s12-4; spec P5-01).

For every counter-example whose two requests differ in the question only (`pairs.text_pairs`),
the stored question `a` is a candidate for the new question `b`. It would be taken when the
cosine distance of their vectors is within the threshold AND the entity guard passes
(`pairs.guard_accepts`). That is a false accept. Reported: the rate with and without the guard,
per category, with Wilson 95% intervals.

The threshold is chosen on the dev part only (`choose_threshold`: the largest threshold on the
grid with no guarded false accept among the non-policy dev counter-examples; the benefit side is
how many of the real paraphrases, `paraphrase_pairs`, it still accepts); the holdout part is then
evaluated once with that threshold.

Vectors: question vectors of the pairs come from a local SQLite cache file; texts the L0 query
cache in Redis already holds are read from there (a plain GET, nothing written), the rest go to
the embedding gateway (a few hundred short texts; `estimate_cost` says what it costs first).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.cache.pairs import GUARDS, POLICY_CATEGORIES, Pair, guard_accepts, text_pairs
from rag.cache.semantic import STRICT_FACTOR
from rag.eval.stats import wilson_interval

THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 26)]
"""The grid: cosine distances 0.02 to 0.50."""


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb) if na and nb else 1.0


def collect_texts(pairs: Sequence[Pair], extra: Sequence[tuple[str, str]] = ()) -> list[str]:
    """The distinct texts whose vectors the evaluation needs, in a fixed order."""
    texts: dict[str, None] = {}
    for p in text_pairs(pairs):
        texts[p.a] = None
        texts[p.b] = None
    for a, b in extra:
        texts[a] = None
        texts[b] = None
    return list(texts)


def estimate_cost(
    texts: Sequence[str], *, cny_per_m_tokens: float = 0.5, chars_per_token: float = 3.0
) -> dict[str, float]:
    """A pessimistic estimate: `chars_per_token` below the usual 4 for English, and every text
    as if it were not cached anywhere."""
    chars = sum(len(t) for t in texts)
    tokens = chars / chars_per_token
    return {
        "texts": len(texts),
        "tokens": round(tokens),
        "cny": round(tokens / 1_000_000 * cny_per_m_tokens, 5),
    }


def paraphrase_pairs(question_dir: str | Path) -> list[tuple[str, str]]:
    """`(question, its hand-written paraphrase)` from `colloquial.jsonl`: 15 casual rewrites of
    public questions with the same gold page (the only real paraphrases in the question sets)."""
    root = Path(question_dir)
    source: dict[str, str] = {}
    with (root / "huggingface_doc_qa_eval.jsonl").open(encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                source[data["qid"]] = data["question"]
    out: list[tuple[str, str]] = []
    with (root / "colloquial.jsonl").open(encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                original = source.get(data.get("meta", {}).get("source_qid", ""))
                if original:
                    out.append((original, data["question"]))
    return out


def _rate(n: int, of: int) -> dict[str, Any]:
    return {"n": n, "of": of, "rate": n / of if of else None, "wilson95": wilson_interval(n, of)}


def false_accepts(
    pairs: Sequence[Pair],
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
    *,
    guard: str = "full",
    strict_factor: float = STRICT_FACTOR,
) -> list[Pair]:
    """The text counter-examples the semantic cache would take for each other at `threshold`
    (`guard`: see `pairs.GUARDS`)."""
    return [
        p
        for p in text_pairs(pairs)
        if guard_accepts(
            p,
            cosine_distance(vectors[p.a], vectors[p.b]),
            threshold,
            strict_factor,
            guard,
        )
    ]


def sweep(
    pairs: Sequence[Pair],
    vectors: Mapping[str, Sequence[float]],
    paraphrases: Sequence[tuple[str, str]],
    thresholds: Sequence[float] = THRESHOLDS,
    *,
    strict_factor: float = STRICT_FACTOR,
) -> list[dict[str, Any]]:
    """One row per threshold: false accepts of the non-policy text counter-examples with each
    guard (`none` = distance alone, `entity`, `full`), and how many of the paraphrases each
    guard accepts."""
    texts = [p for p in text_pairs(pairs) if p.category not in POLICY_CATEGORIES]
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        row: dict[str, Any] = {"threshold": t, "of": len(texts), "paraphrases": len(paraphrases)}
        for guard in GUARDS:
            row[f"false_accepts_{guard}"] = len(
                false_accepts(texts, vectors, t, guard=guard, strict_factor=strict_factor)
            )
            accepted = 0
            for original, rewrite in paraphrases:
                pair = Pair("x", "dev", "positive", "paraphrase", "x", original, rewrite)
                distance = cosine_distance(vectors[original], vectors[rewrite])
                accepted += guard_accepts(pair, distance, t, strict_factor, guard)
            row[f"paraphrases_accepted_{guard}"] = accepted
        rows.append(row)
    return rows


def choose_threshold(rows: Sequence[Mapping[str, Any]], guard: str = "full") -> float:
    """The largest threshold whose false accepts (with `guard`) are zero on the dev part. When
    even the smallest threshold of the grid has one, the smallest: nothing is safe, and the
    report says so."""
    zero = [r["threshold"] for r in rows if r[f"false_accepts_{guard}"] == 0]
    return float(max(zero)) if zero else float(min(r["threshold"] for r in rows))


def evaluate_split(
    pairs: Sequence[Pair],
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
    *,
    strict_factor: float = STRICT_FACTOR,
) -> dict[str, Any]:
    """False accepts of one part at `threshold`: overall (policy classes excluded) and by
    category, for each guard, with Wilson intervals; plus the distance of each category."""
    texts = [p for p in text_pairs(pairs) if p.category not in POLICY_CATEGORIES]
    accepted = {
        guard: {
            p.pair_id
            for p in false_accepts(
                texts, vectors, threshold, guard=guard, strict_factor=strict_factor
            )
        }
        for guard in GUARDS
    }
    by_cat: dict[str, list[Pair]] = defaultdict(list)
    for p in texts:
        by_cat[p.category].append(p)
    categories: dict[str, Any] = {}
    for category, items in sorted(by_cat.items()):
        distances = sorted(cosine_distance(vectors[p.a], vectors[p.b]) for p in items)
        categories[category] = {
            "pairs": len(items),
            **{
                guard: _rate(sum(p.pair_id in accepted[guard] for p in items), len(items))
                for guard in GUARDS
            },
            "distance_median": distances[len(distances) // 2],
            "distance_min": distances[0],
        }
    return {
        "threshold": threshold,
        "pairs": len(texts),
        **{guard: _rate(len(accepted[guard]), len(texts)) for guard in GUARDS},
        "by_category": categories,
        "full_pair_ids": sorted(accepted["full"]),
    }


def summary(pairs: Sequence[Pair]) -> dict[str, Any]:
    counters = [p for p in pairs if p.kind == "counter"]
    return {
        "counter": len(counters),
        "positive": sum(p.kind == "positive" for p in pairs),
        "by_split": {
            f"{split}:{kind}": n
            for (split, kind), n in sorted(Counter((p.split, p.kind) for p in pairs).items())
        },
    }


Embedder = Callable[[list[str]], dict[str, list[float]]]
