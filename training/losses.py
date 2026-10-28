"""Order-agnostic set cross-entropy shared by all three models."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def set_ce(logits: Tensor, positive: Tensor, banned: Tensor, weight: Tensor) -> Tensor:
    """Negative log-probability of drawing *any* still-missing winning ball, per step.

    ``loss_k = -log sum_{j in positive_k} softmax(logits_k masked by banned_k)_j``.
    Valid permutations of the winning set are never penalized.

    Args:
        logits: Per-step scores. Shape: [B, K, 45]
        positive: Balls counted as correct at each step. Shape: [B, K, 45]
        banned: Balls removed from the support (already drawn). Shape: [B, K, 45]
        weight: Per-step weight; 0 skips a step. Shape: [B, K]

    Returns:
        Weighted mean loss (scalar).
    """
    logp = F.log_softmax(logits.masked_fill(banned, float("-inf")), dim=-1)  # Shape: [B, K, 45]
    hit = logp.masked_fill(~positive, float("-inf")).logsumexp(-1)  # Shape: [B, K]
    step_loss = torch.where(positive.any(-1), -hit, torch.zeros_like(hit))  # Shape: [B, K]
    return (step_loss * weight).sum() / weight.sum()


def prefix_masks(target: Tensor) -> tuple[Tensor, Tensor]:
    """Autoregressive masks: step k may pick any of ``target[k:]``, never ``target[:k]``.

    Args:
        target: Winning balls in sampling order. Shape: [B, 6]

    Returns:
        (positive, banned). Each Shape: [B, 6, 45]
    """
    onehot = F.one_hot(target, 45).bool()  # Shape: [B, 6, 45]
    later = torch.triu(torch.ones(6, 6, dtype=torch.bool, device=target.device))  # [k, j]: j >= k
    positive = torch.einsum("kj,bjn->bkn", later.float(), onehot.float()) > 0  # Shape: [B, 6, 45]
    banned = torch.einsum("kj,bjn->bkn", (~later).float(), onehot.float()) > 0  # Shape: [B, 6, 45]
    return positive, banned
