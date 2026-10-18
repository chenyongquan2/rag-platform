"""Seeded, stratified split of the kept questions into a dev set and a holdout set (s14-2).

Tuning looks only at dev; holdout is reported once at the end of a phase and needs at least 100
questions (design s14-2, s14-4). The holdout size is `round(total * holdout_share)`, allocated to
strata in proportion to their sizes; inside a stratum the ids are sorted, shuffled with the seed
and the first ones go to holdout, so the result depends on the seed and the ids only.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from rag.eval.synthetic.sampler import allocate


def split_dev_holdout(
    items: Sequence[tuple[str, str]],
    *,
    seed: int,
    holdout_share: float = 0.5,
    min_each: int = 100,
) -> tuple[list[str], list[str]]:
    """(dev ids, holdout ids) from (id, stratum) pairs, both sorted; raises ValueError when
    either side would have fewer than `min_each` questions."""
    if not 0 < holdout_share < 1:
        raise ValueError("holdout_share must be between 0 and 1")
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    strata: dict[str, list[str]] = {}
    for qid, stratum in items:
        strata.setdefault(stratum, []).append(qid)
    n_holdout = round(len(ids) * holdout_share)
    if n_holdout < min_each or len(ids) - n_holdout < min_each:
        raise ValueError(
            f"{len(ids)} questions give holdout {n_holdout} and dev {len(ids) - n_holdout}; "
            f"each needs at least {min_each}"
        )
    quota = allocate({k: len(v) for k, v in sorted(strata.items())}, n_holdout)
    rng = random.Random(seed)
    dev: list[str] = []
    holdout: list[str] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng.shuffle(members)
        holdout += members[: quota[stratum]]
        dev += members[quota[stratum] :]
    return sorted(dev), sorted(holdout)
