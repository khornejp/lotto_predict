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

    def normalize(self, stats: Scaler) -> None:
        """Standardise ``node`` and ``glob`` of every item in place with ``stats``."""
        self.items = [apply_scaler(it, stats) for it in self.items]


Scaler = dict[str, tuple[torch.Tensor, torch.Tensor]]  # key -> (mean, std) per feature column


def fit_scaler(ds: LottoDataset, indices: list[int]) -> Scaler:
    """Per-column mean/std of ``node`` and ``glob`` over ``indices`` only (pass the train split).

    Args:
        ds: Dataset (not yet normalised).
        indices: Items to fit on.

    Returns:
        ``{"node": (mean [NODE_DIM], std [NODE_DIM]), "glob": (mean [GLOBAL_DIM], std [GLOBAL_DIM])}``.
    """
    node = torch.stack([ds[i]["node"] for i in indices])  # Shape: [N, 45, NODE_DIM]
    glob = torch.stack([ds[i]["glob"] for i in indices])  # Shape: [N, GLOBAL_DIM]
    return {"node": (node.mean((0, 1)), node.std((0, 1))), "glob": (glob.mean(0), glob.std(0))}


def apply_scaler(sample: dict[str, torch.Tensor], stats: Scaler) -> dict[str, torch.Tensor]:
    """Return a copy of ``sample`` with ``node``/``glob`` standardised; constant columns are only centred.

    Args:
        sample: Item with ``node`` [..., 45, NODE_DIM] and ``glob`` [..., GLOBAL_DIM].
        stats: From ``fit_scaler``.

    Returns:
        New dict; other keys are shared, not copied.
    """
    out = dict(sample)
    for k in ("node", "glob"):
        mean, std = stats[k]
        out[k] = (sample[k] - mean) / torch.where(std > 1e-6, std, torch.ones_like(std))
    return out


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
