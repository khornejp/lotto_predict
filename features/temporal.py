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


def multi_horizon_freq(history: np.ndarray, horizons: tuple[int, ...] = (5, 10, 26, 104)) -> np.ndarray:
    """Appearance rate of each ball over several recent horizons plus all history.

    Args:
        history: Draws before the target. Shape: [t, 45]
        horizons: Recent horizons (draws); shorter histories use what exists.

    Returns:
        Rates in [0, 1]; last column = whole history. Shape: [45, len(horizons) + 1]
    """
    cols = [history[-h:].mean(0) for h in horizons] + [history.mean(0)]  # each Shape: [45]
    return np.stack(cols, axis=1).astype(np.float32)


def ewma_rate(history: np.ndarray, halflife: float = 10.0) -> np.ndarray:
    """Exponentially weighted appearance rate (most recent draw has weight 1).

    Args:
        history: Draws before the target. Shape: [t, 45]
        halflife: Draws until a weight halves.

    Returns:
        Weighted rate in [0, 1]. Shape: [45]
    """
    age = np.arange(len(history))[::-1]  # Shape: [t]; 0 = latest
    w = 0.5 ** (age / halflife)  # Shape: [t]
    return (w @ history / w.sum()).astype(np.float32)


def binomial_z(history: np.ndarray) -> np.ndarray:
    """Z-score of each ball's total count against Binomial(t, 6/45).

    Args:
        history: Draws before the target. Shape: [t, 45]

    Returns:
        Z-scores clipped to [-5, 5]. Shape: [45]
    """
    t, p = len(history), 6 / 45
    z = (history.sum(0) - t * p) / np.sqrt(t * p * (1 - p))  # Shape: [45]
    return np.clip(z, -5.0, 5.0).astype(np.float32)
