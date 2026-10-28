"""Combinatorial structure features of a single 6-ball draw (1-indexed balls)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np

SUM_MEAN = 138.0  # E[sum] of 6 balls drawn from 1..45 = 6 * 23
SUM_STD = math.sqrt(6 * (45**2 - 1) / 12 * 39 / 44)  # finite-population variance, ~29.95


def ac_value(nums: Sequence[int]) -> int:
    """Arithmetic complexity: distinct pairwise differences minus 5.

    Args:
        nums: 6 balls.

    Returns:
        AC value in [0, 10].
    """
    return len({abs(a - b) for a, b in combinations(nums, 2)}) - 5


def balance_stats(nums: Sequence[int]) -> np.ndarray:
    """Odd/low ratios, consecutive pairs and decade distribution.

    Args:
        nums: 6 balls, 1-indexed.

    Returns:
        [odd/6, (n<=22)/6, consecutive_pairs/5, decades/6 for 1-9,10-19,20-29,30-39,40-45]. Shape: [8]
    """
    s = sorted(nums)
    decades = np.bincount([n // 10 for n in s], minlength=5)[:5]  # Shape: [5]
    consecutive = sum(b - a == 1 for a, b in zip(s, s[1:]))
    head = [sum(n % 2 for n in s) / 6, sum(n <= 22 for n in s) / 6, consecutive / 5]
    return np.array(head + (decades / 6).tolist(), dtype=np.float32)


def sum_density(total: int) -> float:
    """Density of the draw sum under the exact 6-of-45 distribution, relative to its peak.

    Uses the analytic mean/std (no fitting, so no leakage).

    Args:
        total: Sum of the 6 balls.

    Returns:
        exp(-z^2 / 2), in (0, 1].
    """
    z = (total - SUM_MEAN) / SUM_STD
    return float(math.exp(-z * z / 2))


def last_digit_entropy(nums: Sequence[int]) -> float:
    """Normalized entropy of the 6 balls' last digits (1.0 = all different).

    Args:
        nums: 6 balls, 1-indexed.

    Returns:
        Entropy / log(6), in [0, 1].
    """
    counts = np.bincount([n % 10 for n in nums], minlength=10)
    p = counts[counts > 0] / len(nums)
    return float(-(p * np.log(p)).sum() / math.log(6))
