"""Shared pytest fixtures: synthetic draw history in the `load_draws` schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import draw_date

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def synthetic_draws(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Build `n` uniformly random valid draws.

    Args:
        n: Number of draws.
        seed: RNG seed.

    Returns:
        DataFrame with columns draw, n1..n6 (sorted, 1..45), bonus, date.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(1, n + 1):
        balls = rng.choice(45, size=7, replace=False) + 1  # Shape: [7]
        rows.append([d, *sorted(balls[:6].tolist()), int(balls[6])])
    df = pd.DataFrame(rows, columns=["draw", *MAIN_COLS, "bonus"])
    df["date"] = pd.to_datetime([draw_date(d) for d in df["draw"]])
    return df


@pytest.fixture
def draws_df() -> pd.DataFrame:
    """Fresh 120-draw synthetic history (mutable per test)."""
    return synthetic_draws()
