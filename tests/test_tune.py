from torch.utils.data import Subset

from data.dataset import LottoDataset, split_indices
from features.build import GLOBAL_DIM, NODE_DIM
from models.set_transformer import ARSetTransformer
from training.train import TrainConfig
from training.tune import select_config, tune


def test_select_prefers_smaller_model_within_tolerance():
    trials = [
        {"d_model": 64, "val_loss": 3.000, "n_params": 1000},
        {"d_model": 32, "val_loss": 3.008, "n_params": 300},
        {"d_model": 16, "val_loss": 3.050, "n_params": 100},
    ]
    assert select_config(trials, tol=0.01)["d_model"] == 32


def test_tune_runs_every_combo_and_returns_grid_params(draws_df):
    ds = LottoDataset(draws_df, 20); tr, va, _ = split_indices(len(ds))
    grid = {"d_model": [16, 32], "layers": [1], "dropout": [0.2], "weight_decay": [1e-2, 1e-1]}
    make = lambda **kw: ARSetTransformer(window=20, node_dim=NODE_DIM, global_dim=GLOBAL_DIM, **kw)
    best, trials = tune(make, Subset(ds, tr), Subset(ds, va), TrainConfig(epochs=1, device="cpu"), grid)
    assert len(trials) == 4 and set(best) == set(grid)
    assert best["d_model"] in grid["d_model"] and best["weight_decay"] in grid["weight_decay"]


def test_tune_logs_each_trial(draws_df, caplog):
    import logging
    ds = LottoDataset(draws_df, 20); tr, va, _ = split_indices(len(ds))
    grid = {"d_model": [16], "layers": [1], "dropout": [0.2], "weight_decay": [1e-2, 1e-1]}
    make = lambda **kw: ARSetTransformer(window=20, node_dim=NODE_DIM, global_dim=GLOBAL_DIM, **kw)
    with caplog.at_level(logging.INFO, logger="lotto"):
        tune(make, Subset(ds, tr), Subset(ds, va), TrainConfig(epochs=1, device="cpu"), grid)
    lines = [r.getMessage() for r in caplog.records]
    assert any("trial 1/2" in l for l in lines) and any("trial 2/2" in l for l in lines) and any("chosen" in l for l in lines)
