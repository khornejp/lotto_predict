"""Absorbing-state discrete diffusion over the 6-ball set (SEDD/MDLM-style masked denoising)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.encoder import ContextEncoder
from models.sampling import gumbel_topk
from training.losses import set_ce


class SetDiffusion(nn.Module):
    """Denoise 6 unordered slots from all-[MASK] to a full set in a few parallel steps.

    Slots carry no positional embedding and attend bidirectionally, so the network is
    permutation-equivariant over slots (set semantics). All masked slots therefore share
    one distribution: P(remaining balls | revealed balls, context).

    Args:
        window: Past draws per sample.
        node_dim: Node feature size.
        global_dim: Global context size.
        d_model: Token width.
        nhead: Attention heads.
        layers: Encoder and denoiser layers.
        dropout: Dropout rate.
        steps: Balls revealed per denoising step; must sum to 6.
    """

    def __init__(
        self, window: int, node_dim: int, global_dim: int,
        d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2,
        steps: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()
        assert sum(steps) == 6, f"steps must sum to 6, got {steps}"
        self.steps = steps
        self.encoder = ContextEncoder(window, node_dim, global_dim, d_model, nhead, layers, dropout)
        self.mask_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.ball_emb = nn.Embedding(45, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, 128, dropout, batch_first=True)
        self.denoiser = nn.TransformerDecoder(layer, layers)
        self.scale = 1 / math.sqrt(d_model)

    def slot_logits(self, memory: Tensor, slots: Tensor, masked: Tensor) -> Tensor:
        """Per-slot ball logits given the partially revealed set.

        Args:
            memory: Encoder output. Shape: [B, 46, d]
            slots: Ball per slot (ignored where masked). Shape: [B, 6]
            masked: True where the slot is [MASK]. Shape: [B, 6]

        Returns:
            Logits. Shape: [B, 6, 45]
        """
        tokens = torch.where(masked.unsqueeze(-1), self.mask_emb, self.ball_emb(slots))  # Shape: [B, 6, d]
        h = self.denoiser(tokens, memory)  # Shape: [B, 6, d]
        return h @ memory[:, :45].transpose(1, 2) * self.scale  # Shape: [B, 6, 45]

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Masked set CE at a random mask rate r ~ U(0, 1) per row (>= 1 slot masked).

        Each masked slot may predict any masked target ball; revealed balls are banned.

        Args:
            batch: ``seq`` [B, W, 45], ``node`` [B, 45, F], ``glob`` [B, G], ``target`` [B, 6].

        Returns:
            Scalar loss.
        """
        target = batch["target"]  # Shape: [B, 6]
        b, dev = len(target), target.device
        memory = self.encoder(batch["seq"], batch["node"], batch["glob"])  # Shape: [B, 46, d]
        masked = torch.rand(b, 6, device=dev) < torch.rand(b, 1, device=dev)  # Shape: [B, 6]
        none = ~masked.any(1)
        masked[none, torch.randint(0, 6, (int(none.sum()),), device=dev)] = True
        onehot = F.one_hot(target, 45).bool()  # Shape: [B, 6, 45]
        hidden = (onehot & masked.unsqueeze(-1)).any(1)  # Shape: [B, 45]
        shown = (onehot & ~masked.unsqueeze(-1)).any(1)  # Shape: [B, 45]
        logits = self.slot_logits(memory, target, masked)  # Shape: [B, 6, 45]
        return set_ce(
            logits, hidden.unsqueeze(1).expand(-1, 6, -1), shown.unsqueeze(1).expand(-1, 6, -1), masked.float()
        )

    @torch.no_grad()
    def sample(self, batch: dict[str, Tensor], num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """Reveal ``steps[i]`` balls per step, without replacement, from all-[MASK].

        Args:
            batch: Same keys as ``loss`` (``target`` unused).
            num_samples: Tickets per context (S).
            generator: RNG on the model's device.

        Returns:
            Sorted distinct 0-indexed balls. Shape: [B, S, 6]
        """
        memory = self.encoder(batch["seq"], batch["node"], batch["glob"]).repeat_interleave(num_samples, 0)  # Shape: [B*S, 46, d]
        n, dev = len(memory), memory.device
        drawn = torch.empty(n, 0, dtype=torch.long, device=dev)  # Shape: [B*S, r]
        for k in self.steps:
            r = drawn.shape[1]
            slots = torch.cat([drawn, torch.zeros(n, 6 - r, dtype=torch.long, device=dev)], 1)  # Shape: [B*S, 6]
            masked = torch.arange(6, device=dev).expand(n, 6) >= r  # Shape: [B*S, 6]
            logits = self.slot_logits(memory, slots, masked)[:, -1]  # any masked slot; Shape: [B*S, 45]
            banned = torch.zeros(n, 45, dtype=torch.bool, device=dev).scatter(1, drawn, True)
            drawn = torch.cat([drawn, gumbel_topk(logits, k, banned, generator)], 1)
        return drawn.view(-1, num_samples, 6).sort(-1).values  # Shape: [B, S, 6]
