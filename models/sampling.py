"""Sampling without replacement (Plackett-Luce) via the Gumbel top-k trick."""
from __future__ import annotations

import torch
from torch import Tensor


def gumbel_topk(
    logits: Tensor, k: int, banned: Tensor | None = None, generator: torch.Generator | None = None
) -> Tensor:
    """Draw ``k`` distinct balls per row, P proportional to Plackett-Luce over ``softmax(logits)``.

    Args:
        logits: Ball scores. Shape: [N, 45]
        k: Balls to draw.
        banned: Balls excluded from the draw. Shape: [N, 45]
        generator: RNG on the same device as ``logits``.

    Returns:
        Drawn ball indices (unsorted). Shape: [N, k]
    """
    u = torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-10, 1 - 1e-10)
    scores = logits - torch.log(-torch.log(u))  # Shape: [N, 45]
    if banned is not None:
        scores = scores.masked_fill(banned, float("-inf"))
    return scores.topk(k, dim=-1).indices  # Shape: [N, k]
