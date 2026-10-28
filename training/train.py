"""Training loop shared by all models (duck-typed ``loss(batch)``)."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """Optimisation settings. AdamW + weight decay + early stopping: the main defences against overfitting noise."""

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def _to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_model(model: nn.Module, train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig) -> dict[str, list[float]]:
    """Train with AdamW and early stopping on validation loss; restore the best weights.

    Args:
        model: Model exposing ``loss(batch) -> Tensor``.
        train_ds: Training samples.
        val_ds: Validation samples (chronologically after training).
        cfg: Settings.

    Returns:
        ``{"train_loss": [...], "val_loss": [...]}`` per epoch. The model is left in eval mode on ``cfg.device``.
    """
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(cfg.epochs):
        model.train()
        losses = []
        for batch in loader:
            loss = model.loss(_to(batch, cfg.device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        model.eval()
        # Losses sample a permutation/mask; fix that RNG so val loss is comparable across epochs.
        with torch.no_grad(), torch.random.fork_rng(devices=[] if cfg.device == "cpu" else None):
            torch.manual_seed(cfg.seed)
            val = float(np.mean([model.loss(_to(b, cfg.device)).item() for b in val_loader]))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(val)
        if val < best:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return hist
