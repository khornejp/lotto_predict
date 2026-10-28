"""Temporal Hypergraph NN: each past draw is a 6-ball hyperedge, recency-weighted."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.sampling import gumbel_topk
from training.losses import prefix_masks, set_ce


class HypergraphConv(nn.Module):
    """HGNN layer (Feng et al., 2019): X' = X + Drop(GELU(Dv^-1 H W De^-1 H^T X Theta)).

    Args:
        d_model: Node width.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.theta = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, h: Tensor, w: Tensor) -> Tensor:
        """Node -> hyperedge -> node message passing.

        Args:
            x: Node states. Shape: [B, 45, d]
            h: Incidence (ball i in draw e). Shape: [B, 45, W]
            w: Hyperedge weights. Shape: [W]

        Returns:
            Updated node states. Shape: [B, 45, d]
        """
        de = h.sum(1).clamp_min(1.0)  # Shape: [B, W]
        edges = h.transpose(1, 2) @ self.theta(x) / de.unsqueeze(-1) * w[:, None]  # Shape: [B, W, d]
        dv = (h @ w).clamp_min(1e-6)  # Shape: [B, 45]
        msg = h @ edges / dv.unsqueeze(-1)  # Shape: [B, 45, d]
        return x + self.drop(F.gelu(msg))


class TemporalHGNN(nn.Module):
    """Hypergraph over 45 ball nodes and W past-draw hyperedges -> per-ball logits.

    Edge weight ``gamma^(W-1-e)`` with learnable ``gamma`` favours recent draws. The
    set likelihood is Plackett-Luce over the per-ball logits.

    Args:
        window: Past draws per sample.
        node_dim: Node feature size.
        global_dim: Global context size.
        d_model: Node width.
        layers: Hypergraph conv layers.
        dropout: Dropout rate.
    """

    def __init__(
        self, window: int, node_dim: int, global_dim: int,
        d_model: int = 64, layers: int = 2, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, d_model)
        self.ball_emb = nn.Parameter(torch.randn(45, d_model) * 0.02)
        self.glob_proj = nn.Linear(global_dim, d_model)
        self.convs = nn.ModuleList(HypergraphConv(d_model, dropout) for _ in range(layers))
        self.readout = nn.Linear(d_model, 1)
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(0.97 / 0.03)))
        self.register_buffer("age", torch.arange(window - 1, -1, -1, dtype=torch.float32))  # Shape: [W]

    def ball_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Per-ball scores for the target draw.

        Args:
            batch: ``seq`` [B, W, 45], ``node`` [B, 45, F], ``glob`` [B, G].

        Returns:
            Logits. Shape: [B, 45]
        """
        h = batch["seq"].transpose(1, 2)  # Shape: [B, 45, W]
        w = torch.sigmoid(self.gamma_logit) ** self.age  # Shape: [W]
        # Global context enters before the nonlinear layers; added after them it would shift
        # every ball's logit equally and cancel in the softmax.
        x = self.node_proj(batch["node"]) + self.ball_emb + self.glob_proj(batch["glob"]).unsqueeze(1)  # Shape: [B, 45, d]
        for conv in self.convs:
            x = conv(x, h, w)
        return self.readout(x).squeeze(-1)  # Shape: [B, 45]

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Plackett-Luce set CE with one random draw order per row.

        Args:
            batch: Dataset batch incl. ``target`` [B, 6].

        Returns:
            Scalar loss.
        """
        target = batch["target"]
        perm = torch.rand(target.shape, device=target.device).argsort(1)
        target = target.gather(1, perm)  # Shape: [B, 6]
        logits = self.ball_logits(batch).unsqueeze(1).expand(-1, 6, -1)  # Shape: [B, 6, 45]
        return set_ce(logits, *prefix_masks(target), torch.ones(target.shape, device=target.device))

    @torch.no_grad()
    def sample(self, batch: dict[str, Tensor], num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw tickets from Plackett-Luce over the per-ball logits.

        Args:
            batch: Same keys as ``loss`` (``target`` unused).
            num_samples: Tickets per context (S).
            generator: RNG on the model's device.

        Returns:
            Sorted distinct 0-indexed balls. Shape: [B, S, 6]
        """
        logits = self.ball_logits(batch).repeat_interleave(num_samples, 0)  # Shape: [B*S, 45]
        return gumbel_topk(logits, 6, None, generator).view(-1, num_samples, 6).sort(-1).values  # Shape: [B, S, 6]
