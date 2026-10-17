"""Spec P0-10 AC8-AC11: tests and intervals used to compare eval runs (design s14-4)."""

from __future__ import annotations

import pytest

from rag.eval.stats import (
    PairedChanges,
    mcnemar_exact,
    min_detectable_difference,
    paired_changes,
    paired_permutation_test,
    wilson_interval,
)


def test_mcnemar_exact_known_tables() -> None:
    # Design s14-4: 2 better and 0 worse is p = 0.5; 5 better and 1 worse is about 0.22.
    assert mcnemar_exact(2, 0) == pytest.approx(0.5)
    assert mcnemar_exact(5, 1) == pytest.approx(0.21875)  # 2 * (1 + 6) / 64
    assert mcnemar_exact(1, 5) == mcnemar_exact(5, 1)
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0  # capped at 1
    # Large tables stay finite and exact enough (no float overflow in 2**n).
    assert 0.0 < mcnemar_exact(600, 500) < 0.01
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 0)


def test_permutation_exact_for_few_changes() -> None:
    a = [0.0] * 65
    b = [0.0] * 63 + [1.0, 1.0]
    result = paired_permutation_test(a, b, iterations=10_000, seed=0)
    assert result.exact is True
    assert result.n == 65
    assert result.mean_diff == pytest.approx(2 / 65)
    assert result.p_value == pytest.approx(0.5)  # 4 sign patterns, 2 as extreme

    # On 0/1 scores the exact sign-flip test is the exact McNemar test.
    a2 = [0.0] * 5 + [1.0] + [0.5] * 10
    b2 = [1.0] * 5 + [0.0] + [0.5] * 10
    exact = paired_permutation_test(a2, b2)
    assert exact.exact is True
    assert exact.iterations == 2**6
    assert exact.p_value == pytest.approx(mcnemar_exact(5, 1))

    same = paired_permutation_test([0.3, 0.7], [0.3, 0.7])
    assert same.p_value == 1.0
    assert same.mean_diff == 0.0
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.2], iterations=0)


def test_permutation_monte_carlo_is_seeded() -> None:
    a = [i / 40 for i in range(40)]
    b = [x + (0.05 if i % 3 else -0.04) for i, x in enumerate(a)]  # 40 changes: 2**40 patterns
    first = paired_permutation_test(a, b, iterations=2_000, seed=7)
    again = paired_permutation_test(a, b, iterations=2_000, seed=7)
    assert first.exact is False
    assert first.iterations == 2_000
    assert first == again
    assert 1 / 2_001 <= first.p_value <= 1.0
    assert first.p_value < 0.05  # 26 of 40 questions up by 0.05, 14 down by 0.04

    other = paired_permutation_test(a, b, iterations=2_000, seed=8)
    assert other.p_value == pytest.approx(first.p_value, abs=0.02)


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    assert wilson_interval(65, 65) == (pytest.approx(0.9442, abs=1e-4), 1.0)
    lo90, hi90 = wilson_interval(5, 10, confidence=0.90)
    assert lo < lo90 < hi90 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_paired_changes_counts() -> None:
    a = [1.0, 0.0, 0.5, 0.5, 1.0]
    b = [1.0, 1.0, 0.25, 0.5 + 1e-15, 0.0]
    assert paired_changes(a, b) == PairedChanges(improved=1, worsened=2, unchanged=2)
    with pytest.raises(ValueError):
        paired_changes([1.0], [])

    # The permutation test treats the same float noise as "unchanged": only q2 is flipped.
    perm = paired_permutation_test([*a[:2], 0.5], [*b[:2], 0.5 + 1e-15])
    assert perm.iterations == 2  # 2**1 sign patterns, not 2**2
    assert perm.p_value == 1.0


def test_min_detectable_difference() -> None:
    # Design s14-4: 65 questions tell apart about 8 to 10 points. Best case: 6 better, none
    # worse (p = 0.03125); 5 better and none worse is p = 0.0625.
    best = min_detectable_difference(65)
    assert best is not None
    assert (best.improved, best.worsened) == (6, 0)
    assert best.difference == pytest.approx(6 / 65)
    assert best.p_value == pytest.approx(0.03125)

    one_worse = min_detectable_difference(65, worsened=1)
    assert one_worse is not None
    assert (one_worse.improved, one_worse.worsened) == (8, 1)
    assert one_worse.difference == pytest.approx(7 / 65)

    assert min_detectable_difference(5) is None  # even 5 of 5 better is p = 0.0625
    assert min_detectable_difference(8, worsened=1) is None
