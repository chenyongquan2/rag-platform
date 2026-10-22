"""How big a difference between two eval runs is real (design s14-4).

- Binary per-question metrics (Hit@k, refused or not, citation passed or not): exact McNemar test,
  which looks only at the questions one run gets right and the other gets wrong.
- Continuous per-question metrics (MRR, nDCG, judge scores): paired sign-flip permutation test.
- Proportions (false refusal rate, wrong cache hits): Wilson interval.

No bootstrap: with few changed questions its interval is unreliable (65 questions, 2 better and
0 worse gives a bootstrap 95% interval of [0, 0.077] although the exact p is 0.5).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel

_BATCH = 65_536  # sign patterns evaluated per matrix product (bounds memory)

CHANGE_TOL = 1e-12
"""A per-question score changed only when it moved by more than this (float noise is not a
change). Shared by the permutation test, the change counts and the changed-question lists."""


def mcnemar_exact(improved: int, worsened: int) -> float:
    """Two-sided exact McNemar p-value (a sign test on the discordant questions).

    `improved`: questions run B gets right and run A wrong; `worsened`: the reverse. Questions
    both runs get right (or both wrong) do not enter the test.
    """
    if improved < 0 or worsened < 0:
        raise ValueError(f"counts must be >= 0, got {improved}, {worsened}")
    n = improved + worsened
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(improved, worsened) + 1))
    return min(1.0, 2 * tail / 2**n)


@dataclass(frozen=True)
class PermutationResult:
    mean_diff: float
    """Mean of `b - a` over all questions."""
    p_value: float
    iterations: int
    """Sign patterns evaluated: all `2**changed` when exact, else the random draws."""
    exact: bool
    n: int
    """Number of paired questions (including unchanged ones)."""


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], *, iterations: int = 10_000, seed: int = 0
) -> PermutationResult:
    """Two-sided paired permutation test of mean(b - a) = 0 by flipping the sign of each difference.

    Questions with no difference (within `CHANGE_TOL`) do not change under a flip, so only the
    `m` changed ones are flipped. When `2**m <= iterations` every sign pattern is enumerated and
    the p-value is exact; otherwise `iterations` random patterns are drawn from a generator
    seeded with `seed` and `p = (1 + as_extreme) / (1 + iterations)`, which is never 0.
    """
    da, db = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if da.ndim != 1 or da.shape != db.shape:
        raise ValueError(f"a and b must be paired 1-d sequences, got {da.shape} and {db.shape}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    n = len(da)
    diffs = db - da
    mean_diff = float(diffs.mean()) if n else 0.0
    changed = diffs[np.abs(diffs) > CHANGE_TOL]
    m = len(changed)
    if m == 0:
        return PermutationResult(mean_diff=mean_diff, p_value=1.0, iterations=0, exact=True, n=n)
    # Same n in every pattern, so comparing |sum| is comparing |mean|. The small slack keeps the
    # observed pattern itself counted despite float rounding in the matrix product.
    observed = abs(float(changed.sum()))
    threshold = observed - 1e-9 * max(1.0, float(np.abs(changed).sum()))

    if 2**m <= iterations:
        total = 2**m
        as_extreme = 0
        bits = np.arange(m, dtype=np.int64)
        for start in range(0, total, _BATCH):
            codes = np.arange(start, min(start + _BATCH, total), dtype=np.int64)
            signs = ((codes[:, None] >> bits) & 1) * 2.0 - 1.0
            as_extreme += int(np.count_nonzero(np.abs(signs @ changed) >= threshold))
        return PermutationResult(
            mean_diff=mean_diff, p_value=as_extreme / total, iterations=total, exact=True, n=n
        )

    rng = np.random.default_rng(seed)
    as_extreme = 0
    for start in range(0, iterations, _BATCH):
        size = min(_BATCH, iterations - start)
        signs = rng.integers(0, 2, size=(size, m)) * 2.0 - 1.0
        as_extreme += int(np.count_nonzero(np.abs(signs @ changed) >= threshold))
    return PermutationResult(
        mean_diff=mean_diff,
        p_value=(1 + as_extreme) / (1 + iterations),
        iterations=iterations,
        exact=False,
        n=n,
    )


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion; `(0, 1)` when `n == 0`."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes} of {n}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return (lo, hi)


class Proportion(BaseModel):
    """`count` of `n` with the 95% Wilson interval (design s14-4); rate None when n is 0."""

    count: int
    n: int
    rate: float | None
    low: float
    high: float

    @classmethod
    def of(cls, count: int, n: int) -> Proportion:
        low, high = wilson_interval(count, n)
        return cls(count=count, n=n, rate=count / n if n else None, low=low, high=high)

    def text(self) -> str:
        if self.rate is None:
            return f"{self.count}/{self.n}"
        return f"{self.count}/{self.n} = {self.rate:.1%} [{self.low:.1%}, {self.high:.1%}]"


@dataclass(frozen=True)
class PairedChanges:
    improved: int
    worsened: int
    unchanged: int


def paired_changes(
    a: Sequence[float], b: Sequence[float], *, tol: float = CHANGE_TOL
) -> PairedChanges:
    """Per question: did run B score higher, lower or the same as run A (within `tol`)?"""
    if len(a) != len(b):
        raise ValueError(f"a and b must have the same length, got {len(a)} and {len(b)}")
    improved = sum(1 for x, y in zip(a, b, strict=True) if y > x + tol)
    worsened = sum(1 for x, y in zip(a, b, strict=True) if y < x - tol)
    unchanged = len(a) - improved - worsened
    return PairedChanges(improved=improved, worsened=worsened, unchanged=unchanged)


@dataclass(frozen=True)
class MinDetectable:
    """Smallest net change of a binary metric that can reach `p <= alpha` with exact McNemar."""

    n: int
    improved: int
    worsened: int
    difference: float
    """`(improved - worsened) / n`: the change of the metric's mean, as a fraction."""
    p_value: float
    alpha: float


def min_detectable_difference(
    n: int, *, alpha: float = 0.05, worsened: int = 0
) -> MinDetectable | None:
    """How big a difference `n` questions can tell apart (design s14-4), for a binary metric.

    Assumes `worsened` questions got worse and finds the fewest improved questions for which
    the exact McNemar p-value is at most `alpha`. With `worsened=0` this is the best case: any
    smaller difference is never significant on this question count. `None` when even every
    remaining question improving is not enough.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for improved in range(worsened + 1, n - worsened + 1):
        p = mcnemar_exact(improved, worsened)
        if p <= alpha:
            return MinDetectable(
                n=n,
                improved=improved,
                worsened=worsened,
                difference=(improved - worsened) / n,
                p_value=p,
                alpha=alpha,
            )
    return None


def weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    low: int,
    high: int,
    weights: str = "quadratic",
) -> float | None:
    """Cohen's weighted kappa of two raters on the integer scale `low..high` (design s14-3).

    Disagreement weights are `((i - j) / (high - low)) ** 2` (quadratic) or `|i - j| / (high -
    low)` (linear), so a one-point miss costs less than a three-point one. Kappa is
    `1 - sum(w * observed) / sum(w * expected)`, the expected table being the product of the two
    raters' marginals. None when the expected disagreement is 0 (both raters give one score).
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have the same length, got {len(a)} and {len(b)}")
    if high <= low:
        raise ValueError(f"need low < high, got {low}..{high}")
    if weights not in ("quadratic", "linear"):
        raise ValueError(f"weights must be quadratic or linear, got {weights!r}")
    values = [*a, *b]
    if any(v < low or v > high or int(v) != v for v in values):
        raise ValueError(f"scores must be integers in {low}..{high}")
    if not a:
        return None
    k = high - low + 1
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[int(x) - low, int(y) - low] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    steps = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    w = steps**2 if weights == "quadratic" else steps
    disagreement = float((w * expected).sum())
    if disagreement <= 0:
        return None
    return 1.0 - float((w * observed).sum()) / disagreement


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (ties get average ranks); None when either side is constant."""
    if len(a) != len(b):
        raise ValueError(f"a and b must have the same length, got {len(a)} and {len(b)}")
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)
