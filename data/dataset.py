"""PyTorch dataset of leakage-free samples, one per historical target draw."""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.loader import draws_to_multihot
from features.build import build_sample


class LottoDataset(Dataset):
    """Precomputed samples for target draw indices ``window .. T-1``.

    Args:
        df: Draws from ``load_draws``.
        window: Past draws per sample.
    """

    def __init__(self, df: pd.DataFrame, window: int = 52) -> None:
        multihot = draws_to_multihot(df)  # Shape: [T, 45]
        self.items: list[dict[str, torch.Tensor]] = []
        for t in range(window, len(df)):
            s = build_sample(df, multihot, t, window)
            self.items.append({
                "seq": torch.from_numpy(s["seq"]),  # Shape: [W, 45]
                "node": torch.from_numpy(s["node"]),  # Shape: [45, NODE_DIM]
                "glob": torch.from_numpy(s["glob"]),  # Shape: [GLOBAL_DIM]
                "target": torch.from_numpy(s["target"]),  # Shape: [6]
            })
        self.draw_numbers: list[int] = df["draw"].iloc[window:].tolist()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.items[i]


def split_indices(n: int, train: float = 0.8, val: float = 0.1) -> tuple[list[int], list[int], list[int]]:
    """Contiguous chronological train/val/test split (no shuffling across time).

    Args:
        n: Dataset length.
        train: Train fraction.
        val: Validation fraction; the rest is test.

    Returns:
        (train, val, test) index lists.
    """
    a, b = int(n * train), int(n * (train + val))
    return list(range(a)), list(range(a, b)), list(range(b, n))
