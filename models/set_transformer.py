"""Autoregressive Set Transformer: P(set) = prod_k P(x_k | x_<k, context)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.encoder import ContextEncoder
from models.sampling import gumbel_topk
from training.losses import prefix_masks, set_ce


class ARSetTransformer(nn.Module):
    """Context encoder + causal pointer decoder that draws 6 balls one at a time.

    Each step points at the 45 ball tokens of the encoder memory, so already-drawn
    balls re-weight the next draw through attention. Targets are randomly permuted
    during training so the model learns P(set), not P(sorted sequence).

    Args:
        window: Past draws per sample.
        node_dim: Node feature size.
        global_dim: Global context size.
        d_model: Token width.
        nhead: Attention heads.
        layers: Encoder and decoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self, window: int, node_dim: int, global_dim: int,
        d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = ContextEncoder(window, node_dim, global_dim, d_model, nhead, layers, dropout)
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
        self.ball_emb_dec = nn.Embedding(45, d_model)
        self.step_emb = nn.Parameter(torch.randn(6, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, nhead, 128, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.scale = 1 / math.sqrt(d_model)

    def step_logits(self, memory: Tensor, prefix: Tensor) -> Tensor:
        """Next-ball logits after each prefix position.

        Args:
            memory: Encoder output. Shape: [B, 46, d]
            prefix: Balls drawn so far, in draw order (L in 0..6). Shape: [B, L]

        Returns:
            Logits for steps 0..min(L, 5). Shape: [B, min(L+1, 6), 45]
        """
        prev = prefix[:, :5]  # Shape: [B, L']
        tokens = torch.cat([self.bos.expand(len(memory), 1, -1), self.ball_emb_dec(prev)], dim=1)  # Shape: [B, L'+1, d]
        n = tokens.shape[1]
        tokens = tokens + self.step_emb[:n]
        causal = nn.Transformer.generate_square_subsequent_mask(n, device=memory.device)  # Shape: [n, n]
        h = self.decoder(tokens, memory, tgt_mask=causal, tgt_is_causal=True)  # Shape: [B, n, d]
        return h @ memory[:, :45].transpose(1, 2) * self.scale  # Shape: [B, n, 45]

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Order-agnostic set CE on a random permutation of each target.

        Args:
            batch: ``seq`` [B, W, 45], ``node`` [B, 45, F], ``glob`` [B, G], ``target`` [B, 6].

        Returns:
            Scalar loss.
        """
        memory = self.encoder(batch["seq"], batch["node"], batch["glob"])  # Shape: [B, 46, d]
        target = batch["target"]
        perm = torch.rand(target.shape, device=target.device).argsort(1)  # Shape: [B, 6]
        target = target.gather(1, perm)  # Shape: [B, 6]
        logits = self.step_logits(memory, target)  # Shape: [B, 6, 45]
        return set_ce(logits, *prefix_masks(target), torch.ones(target.shape, device=target.device))

    @torch.no_grad()
    def sample(self, batch: dict[str, Tensor], num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw ``num_samples`` tickets per context, ball by ball.

        Args:
            batch: Same keys as ``loss`` (``target`` unused).
            num_samples: Tickets per context (S).
            generator: RNG on the model's device.

        Returns:
            Sorted distinct 0-indexed balls. Shape: [B, S, 6]
        """
        memory = self.encoder(batch["seq"], batch["node"], batch["glob"])  # Shape: [B, 46, d]
        memory = memory.repeat_interleave(num_samples, 0)  # Shape: [B*S, 46, d]
        drawn = torch.empty(len(memory), 0, dtype=torch.long, device=memory.device)  # Shape: [B*S, 0]
        for _ in range(6):
            logits = self.step_logits(memory, drawn)[:, -1]  # Shape: [B*S, 45]
            banned = torch.zeros_like(logits, dtype=torch.bool).scatter(1, drawn, True)  # Shape: [B*S, 45]
            drawn = torch.cat([drawn, gumbel_topk(logits, 1, banned, generator)], dim=1)
        return drawn.view(-1, num_samples, 6).sort(-1).values  # Shape: [B, S, 6]
