"""Backtest sampled tickets against held-out draws vs. the uniform-random baseline."""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

BASELINE_MEAN_MATCHES = 36 / 45  # E[matches] of a random ticket = 6 * 6/45
BASELINE_RATE_GE3 = float(stats.hypergeom(45, 6, 6).sf(2))  # P(>= 3 matches), ~0.0186


def count_matches(samples: Tensor, targets: Tensor) -> np.ndarray:
    """Number of winning balls in each ticket.

    Args:
        samples: Tickets. Shape: [N, S, 6]
        targets: Winning balls. Shape: [N, 6]

    Returns:
        Match counts. Shape: [N, S]
    """
    hit = samples.unsqueeze(-1) == targets[:, None, None, :]  # Shape: [N, S, 6, 6]
    return hit.any(-1).sum(-1).cpu().numpy()


def backtest(model: nn.Module, ds: Dataset, tickets: int = 5, seed: int = 0, device: str = "cpu") -> dict[str, float]:
    """Sample ``tickets`` per historical draw and compare matches to the random baseline.

    Args:
        model: Model exposing ``sample(batch, num_samples, generator)``.
        ds: Held-out samples (with targets).
        tickets: Tickets per draw.
        seed: Sampling seed.
        device: Device for sampling.

    Returns:
        Metrics with baselines and a one-sided one-sample t-test (H1: mean matches > 0.8).
    """
    model.to(device).eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    matches, unique = [], []
    for batch in DataLoader(ds, batch_size=64):
        batch = {k: v.to(device) for k, v in batch.items()}
        s = model.sample(batch, tickets, gen)  # Shape: [B, S, 6]
        matches.append(count_matches(s, batch["target"]))
        unique += [len({tuple(t) for t in row.tolist()}) / tickets for row in s]
    m = np.concatenate(matches).ravel()  # Shape: [N*S]
    t = stats.ttest_1samp(m, BASELINE_MEAN_MATCHES, alternative="greater")
    return {
        "n_draws": len(ds),
        "tickets": tickets,
        "mean_matches": float(m.mean()),
        "baseline_mean_matches": BASELINE_MEAN_MATCHES,
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
        "rate_ge3": float((m >= 3).mean()),
        "baseline_rate_ge3": BASELINE_RATE_GE3,
        "unique_ratio": float(np.mean(unique)),
    }
