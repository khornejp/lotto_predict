"""Graph features from the ball co-occurrence network over a draw window."""
from __future__ import annotations

import numpy as np


def cooccurrence_matrix(window: np.ndarray) -> np.ndarray:
    """Pairwise co-occurrence rate of balls over a window of draws.

    Args:
        window: Binary draw matrix. Shape: [W, 45]

    Returns:
        Symmetric matrix, zero diagonal; entry (i, j) = share of draws with both. Shape: [45, 45]
    """
    if len(window) == 0:
        return np.zeros((45, 45), dtype=np.float32)
    co = window.T @ window / len(window)  # Shape: [45, 45]
    np.fill_diagonal(co, 0.0)
    return co.astype(np.float32)


def centrality(co: np.ndarray) -> np.ndarray:
    """Weighted PageRank (scaled so uniform = 1) and unweighted degree centrality.

    Args:
        co: Co-occurrence matrix. Shape: [45, 45]

    Returns:
        Columns [pagerank * 45, degree / 44]. Shape: [45, 2]
    """
    out = np.zeros((45, 2), dtype=np.float32)  # Shape: [45, 2]
    if co.sum() == 0:
        out[:, 0] = 1.0
        return out
    out[:, 0] = _pagerank(co) * 45
    out[:, 1] = (co > 0).sum(1) / 44
    return out


def _pagerank(co: np.ndarray, alpha: float = 0.85, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Weighted PageRank by power iteration (networkx semantics, ~100x faster for 45 nodes).

    Args:
        co: Non-negative weight matrix. Shape: [45, 45]

    Returns:
        PageRank vector summing to 1. Shape: [45]
    """
    n = len(co)
    rowsum = co.sum(1, keepdims=True)  # Shape: [45, 1]
    p = np.where(rowsum > 0, co / np.where(rowsum > 0, rowsum, 1), 1.0 / n)  # dangling -> uniform; Shape: [45, 45]
    r = np.full(n, 1.0 / n)  # Shape: [45]
    for _ in range(max_iter):
        nxt = (1 - alpha) / n + alpha * (r @ p)  # Shape: [45]
        if np.abs(nxt - r).sum() < n * tol:
            return nxt
        r = nxt
    return r
