"""Shared pytest fixtures: synthetic draw history in the `load_draws` schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from data.loader import draw_date
from features.build import GLOBAL_DIM, NODE_DIM

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


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    """Seeded random batch, B=4, W=8, in the dataset's tensor layout."""
    g = torch.Generator().manual_seed(0)
    return {
        "seq": (torch.rand(4, 8, 45, generator=g) < 6 / 45).float(),  # Shape: [4, 8, 45]
        "node": torch.randn(4, 45, NODE_DIM, generator=g),  # Shape: [4, 45, NODE_DIM]
        "glob": torch.randn(4, GLOBAL_DIM, generator=g),  # Shape: [4, GLOBAL_DIM]
        "target": torch.rand(4, 45, generator=g).argsort(1)[:, :6].sort(1).values,  # Shape: [4, 6]
    }
