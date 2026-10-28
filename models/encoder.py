"""Shared context encoder: one token per ball plus one global token."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ContextEncoder(nn.Module):
    """Transformer over 45 ball tokens + 1 global-context token.

    Ball token i = Linear(window history of ball i) + Linear(node features of i) + ball embedding.
    Small width limits capacity on ~1k near-i.i.d. training draws.

    Args:
        window: Past draws per sample (W).
        node_dim: Node feature size.
        global_dim: Global context size.
        d_model: Token width.
        nhead: Attention heads.
        layers: Encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self, window: int, node_dim: int, global_dim: int,
        d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(window, d_model)
        self.node_proj = nn.Linear(node_dim, d_model)
        self.glob_proj = nn.Linear(global_dim, d_model)
        self.ball_emb = nn.Parameter(torch.randn(45, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 128, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, seq: Tensor, node: Tensor, glob: Tensor) -> Tensor:
        """Encode one batch of contexts.

        Args:
            seq: Past draws. Shape: [B, W, 45]
            node: Per-ball features. Shape: [B, 45, node_dim]
            glob: Global features. Shape: [B, global_dim]

        Returns:
            Memory; tokens 0..44 are balls, 45 is global. Shape: [B, 46, d_model]
        """
        balls = self.seq_proj(seq.transpose(1, 2)) + self.node_proj(node) + self.ball_emb  # Shape: [B, 45, d]
        g = self.glob_proj(glob).unsqueeze(1)  # Shape: [B, 1, d]
        return self.encoder(torch.cat([balls, g], dim=1))  # Shape: [B, 46, d]
