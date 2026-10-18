"""Seeded, stratified split of the kept questions into a dev set and a holdout set (s14-2).

Tuning looks only at dev; holdout is reported once at the end of a phase and needs at least 100
questions (design s14-2, s14-4). Near-duplicate questions (the same question asked from two
templated pages) form a group, and a group always lands in one set, so holdout never repeats a
dev question. The holdout size is `round(total * holdout_share)` questions, allocated to strata
in proportion to their sizes (a group counts in the stratum of its first id); inside a stratum the
groups are sorted, shuffled with the seed and go to holdout while they fit into the stratum's
quota (a group that would overshoot it goes to dev), so the result depends on the seed, the ids
and the groups only.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from rag.eval.synthetic.filters import STOPWORDS, tokens
from rag.eval.synthetic.labels import normalize_answer
from rag.eval.synthetic.sampler import allocate


def _content(question: str) -> set[str]:
    return {w for w in tokens(question) if w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_duplicate_groups(
    items: Sequence[tuple[str, str, str]],
    *,
    min_jaccard: float = 0.6,
    same_answer_jaccard: float = 0.3,
) -> dict[str, str]:
    """(qid, question, answer) -> {qid: group id}, the group id being its smallest qid.

    Two questions are near duplicates when the content words of the questions (no stop words)
    overlap by at least `min_jaccard` (Jaccard), or by at least `same_answer_jaccard` when the
    normalized answers are equal. Groups are the connected sets of near-duplicate pairs.
    """
    qids = [q for q, _, _ in items]
    parent = {q: q for q in qids}

    def find(q: str) -> str:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    words = [_content(question) for _, question, _ in items]
    answers = [normalize_answer(answer) for _, _, answer in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            overlap = _jaccard(words[i], words[j])
            same = bool(answers[i]) and answers[i] == answers[j]
            if overlap >= min_jaccard or (same and overlap >= same_answer_jaccard):
                a, b = find(qids[i]), find(qids[j])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = {q: find(q) for q in qids}
    smallest: dict[str, str] = {}
    for q in sorted(qids):
        smallest.setdefault(roots[q], q)
    return {q: smallest[roots[q]] for q in qids}


def split_dev_holdout(
    items: Sequence[tuple[str, str]],
    *,
    seed: int,
    holdout_share: float = 0.5,
    min_each: int = 100,
    groups: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """(dev ids, holdout ids) from (id, stratum) pairs, both sorted; `groups` (id -> group id)
    keeps each group in one set. Raises ValueError when either side would have fewer than
    `min_each` questions."""
    if not 0 < holdout_share < 1:
        raise ValueError("holdout_share must be between 0 and 1")
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    stratum_of = dict(items)
    group_of = {i: (groups or {}).get(i, i) for i in ids}
    members: dict[str, list[str]] = {}
    for qid in sorted(ids):
        members.setdefault(group_of[qid], []).append(qid)
    strata: dict[str, list[str]] = {}
    for group in sorted(members):
        strata.setdefault(stratum_of[members[group][0]], []).append(group)
    n_holdout = round(len(ids) * holdout_share)
    sizes = {k: sum(len(members[g]) for g in v) for k, v in sorted(strata.items())}
    quota = allocate(sizes, n_holdout)
    rng = random.Random(seed)
    dev: list[str] = []
    holdout: list[str] = []
    for stratum in sorted(strata):
        units = list(strata[stratum])
        rng.shuffle(units)
        taken = 0
        for group in units:
            if taken + len(members[group]) <= quota[stratum]:
                holdout += members[group]
                taken += len(members[group])
            else:
                dev += members[group]
    if len(holdout) < min_each or len(dev) < min_each:
        raise ValueError(
            f"{len(ids)} questions give holdout {len(holdout)} and dev {len(dev)}; "
            f"each needs at least {min_each}"
        )
    return sorted(dev), sorted(holdout)
