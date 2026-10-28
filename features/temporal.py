"""Temporal dynamics features: appearance gaps, range entropy, regime shift."""
from __future__ import annotations

import numpy as np


def gap_features(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Current lag since last appearance and its Z-score vs. historical gaps.

    Args:
        history: All draws before the target. Shape: [t, 45]

    Returns:
        (lag, z): draws since last appearance (``t`` if never) and
        ``(lag - mean(gaps)) / (std(gaps, ddof=1) + 1e-6)`` clipped to [-5, 5],
        0 when fewer than 2 gaps exist. Each Shape: [45]
    """
    t = len(history)
    lag = np.full(45, float(t), dtype=np.float32)  # Shape: [45]
    z = np.zeros(45, dtype=np.float32)  # Shape: [45]
    for i in range(45):
        idx = np.flatnonzero(history[:, i])
        if idx.size == 0:
            continue
        lag[i] = t - idx[-1]
        gaps = np.diff(idx)
        if gaps.size >= 2:
            z[i] = (lag[i] - gaps.mean()) / (gaps.std(ddof=1) + 1e-6)
    return lag, np.clip(z, -5.0, 5.0)


def range_entropy(window: np.ndarray) -> float:
    """Normalized Shannon entropy of ball counts over 5 ranges (1-9, ..., 37-45).

    Args:
        window: Binary draw matrix. Shape: [N, 45]

    Returns:
        Entropy / log(5), in [0, 1]; 0.0 when the window is empty.
    """
    counts = window.sum(0).reshape(5, 9).sum(1)  # Shape: [5]
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(5))


def regime_cosine(history: np.ndarray, short: int = 10, long: int = 52) -> float:
    """Cosine similarity of short- vs long-horizon ball frequency vectors.

    Args:
        history: Draws before the target. Shape: [t, 45]
        short: Recent horizon (draws).
        long: Long horizon (draws).

    Returns:
        Cosine in [0, 1]; 0.0 if either vector is zero.
    """
    a = history[-short:].sum(0)  # Shape: [45]
    b = history[-long:].sum(0)  # Shape: [45]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))
