"""Assemble one model input sample for target draw index ``t`` from draws ``< t`` only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import MAIN_COLS, draw_date
from features.calendar_feats import calendar_features
from features.cooccurrence import centrality, cooccurrence_matrix
from features.structural import ac_value, balance_stats, sum_density
from features.temporal import gap_features, range_entropy, regime_cosine

NODE_DIM = 5  # [log1p(lag), gap z, window freq, pagerank*45, degree]
GLOBAL_DIM = 16  # [ac/10, balance(8), sum density, range entropy, regime cosine, calendar(4)]
RECENT = 10  # short horizon for entropy / regime features


def build_sample(
    df: pd.DataFrame, multihot: np.ndarray, t: int, window: int
) -> dict[str, np.ndarray | None]:
    """Build the input tensors for predicting draw index ``t`` (0-based).

    Causality: only ``multihot[:t]`` and ``df`` row ``t-1`` feed the inputs; the
    target date (``t+1`` as a draw number) is known in advance.

    Args:
        df: Draws from ``load_draws``.
        multihot: Binary draw matrix of ``df``. Shape: [T, 45]
        t: Target draw index, ``window <= t <= T``; ``t == T`` is the next, undrawn draw.
        window: Number of past draws in the sequence input (>= 10).

    Returns:
        Dict with ``seq`` [window, 45], ``node`` [45, NODE_DIM], ``glob`` [GLOBAL_DIM]
        (float32) and ``target`` [6] int64 sorted 0-indexed balls, or None when ``t == T``.
    """
    assert window >= RECENT, f"window must be >= {RECENT}"
    assert window <= t <= len(multihot), f"t={t} outside [{window}, {len(multihot)}]"
    history = multihot[:t]  # Shape: [t, 45]
    seq = history[-window:]  # Shape: [window, 45]

    lag, z = gap_features(history)  # Shape: [45], [45]
    cent = centrality(cooccurrence_matrix(seq))  # Shape: [45, 2]
    node = np.column_stack([np.log1p(lag), z, seq.mean(0), cent]).astype(np.float32)  # Shape: [45, 5]

    prev = df.loc[t - 1, MAIN_COLS].to_numpy(dtype=int)  # Shape: [6], 1-indexed
    glob = np.concatenate([
        [ac_value(prev) / 10],
        balance_stats(prev),
        [sum_density(int(prev.sum())), range_entropy(history[-RECENT:]), regime_cosine(history, RECENT, window)],
        calendar_features(draw_date(t + 1)),
    ]).astype(np.float32)  # Shape: [16]

    target = None
    if t < len(multihot):
        target = np.sort(df.loc[t, MAIN_COLS].to_numpy(dtype=np.int64)) - 1  # Shape: [6]
    return {"seq": seq.astype(np.float32), "node": node, "glob": glob, "target": target}
