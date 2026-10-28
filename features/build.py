"""Assemble one model input sample for target draw index ``t`` from draws ``< t`` only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import MAIN_COLS, draw_date
from features.calendar_feats import calendar_features
from features.cooccurrence import centrality, cooccurrence_matrix
from features.structural import SUM_STD, ac_value, balance_stats, last_digit_entropy, sum_density
from features.temporal import binomial_z, ewma_rate, gap_features, multi_horizon_freq, range_entropy, regime_cosine

# Node columns: [log1p(lag), gap z, window freq, pagerank*45, degree, freq@5/10/26/104/all,
#                ewma rate, carry (in last draw), neighbour of last draw, co-occurrence with last draw,
#                bonus rate in window, binomial z, last digit/9]
NODE_DIM = 17
CARRY_COL = 11
# Global: [ac/10, balance(8), sum density, range entropy, regime cosine, calendar(4),
#          overlap(t-1, t-2)/6, sum change/std, last-digit entropy]
GLOBAL_DIM = 19
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
    co = cooccurrence_matrix(seq)  # Shape: [45, 45]
    cent = centrality(co)  # Shape: [45, 2]
    last = history[-1]  # Shape: [45]
    neighbour = np.clip(np.roll(last, 1) + np.roll(last, -1), 0, 1)  # ball i-1 or i+1 drawn last time
    neighbour[0], neighbour[-1] = last[1], last[-2]  # no wrap-around between 1 and 45
    bonus = np.bincount(df["bonus"].iloc[t - window:t].to_numpy() - 1, minlength=45) / window  # Shape: [45]
    node = np.column_stack([
        np.log1p(lag), z, seq.mean(0), cent,
        multi_horizon_freq(history), ewma_rate(history), last, neighbour, co @ last,
        bonus, binomial_z(history), (np.arange(1, 46) % 10) / 9,
    ]).astype(np.float32)  # Shape: [45, 17]

    prev = df.loc[t - 1, MAIN_COLS].to_numpy(dtype=int)  # Shape: [6], 1-indexed
    prev2 = df.loc[t - 2, MAIN_COLS].to_numpy(dtype=int)  # Shape: [6]
    glob = np.concatenate([
        [ac_value(prev) / 10],
        balance_stats(prev),
        [sum_density(int(prev.sum())), range_entropy(history[-RECENT:]), regime_cosine(history, RECENT, window)],
        calendar_features(draw_date(t + 1)),
        [len(set(prev) & set(prev2)) / 6, (prev.sum() - prev2.sum()) / SUM_STD, last_digit_entropy(prev)],
    ]).astype(np.float32)  # Shape: [19]

    target = None
    if t < len(multihot):
        target = np.sort(df.loc[t, MAIN_COLS].to_numpy(dtype=np.int64)) - 1  # Shape: [6]
    return {"seq": seq.astype(np.float32), "node": node, "glob": glob, "target": target}
