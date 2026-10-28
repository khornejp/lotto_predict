"""Small regularisation-biased grid search, selected on validation loss only."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import product

from torch import nn
from torch.utils.data import Dataset

from training.train import TrainConfig, set_seed, train_model

# Biased toward small, strongly regularised models: the data are near-i.i.d., so capacity mostly fits noise.
GRID: dict[str, list] = {"d_model": [32, 64], "layers": [1, 2], "dropout": [0.2, 0.4], "weight_decay": [1e-2, 1e-1]}
TOLERANCE = 0.01  # nats of val loss treated as a tie


def select_config(trials: list[dict], tol: float = TOLERANCE) -> dict:
    """Pick the smallest model whose val loss is within ``tol`` of the best (1-SE-style rule).

    Args:
        trials: Dicts with ``val_loss``, ``n_params`` and the tried parameters.
        tol: Val-loss margin counted as equal.

    Returns:
        The chosen trial.
    """
    best = min(t["val_loss"] for t in trials)
    return min((t for t in trials if t["val_loss"] <= best + tol), key=lambda t: (t["n_params"], t["val_loss"]))


def tune(
    make_model: Callable[..., nn.Module], train_ds: Dataset, val_ds: Dataset,
    cfg: TrainConfig, grid: dict[str, list] = GRID,
) -> tuple[dict, list[dict]]:
    """Train one model per grid point and select by validation loss (test data never seen).

    Args:
        make_model: Builds a model from ``d_model``/``layers``/``dropout`` kwargs.
        train_ds: Training samples.
        val_ds: Validation samples.
        cfg: Base training config; ``weight_decay`` is overridden per trial.
        grid: Parameter grid; ``weight_decay`` goes to the optimiser, the rest to the model.

    Returns:
        (chosen parameters, all trials with ``val_loss`` and ``n_params``).
    """
    trials = []
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        set_seed(cfg.seed)
        model = make_model(**{k: v for k, v in params.items() if k != "weight_decay"})
        hist = train_model(model, train_ds, val_ds, replace(cfg, weight_decay=params["weight_decay"]))
        trials.append({**params, "val_loss": min(hist["val_loss"]),
                       "n_params": sum(p.numel() for p in model.parameters())})
    chosen = select_config(trials)
    return {k: chosen[k] for k in grid}, trials
