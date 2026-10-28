"""Graph features from the ball co-occurrence network over a draw window."""
from __future__ import annotations

import networkx as nx
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
    pr = nx.pagerank(nx.from_numpy_array(co), weight="weight")
    out[:, 0] = np.array([pr[i] for i in range(45)]) * 45
    out[:, 1] = (co > 0).sum(1) / 44
    return out
