import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Subset

from data.dataset import LottoDataset, split_indices
from evaluation.backtest import backtest
from features.build import GLOBAL_DIM, NODE_DIM
from models.hypergraph import TemporalHGNN
from models.sampling import gumbel_topk
from training.train import TrainConfig, set_seed, train_model


class Uniform(nn.Module):
    def sample(self, batch, num_samples, generator=None):
        B = batch["seq"].shape[0]
        return gumbel_topk(torch.zeros(B * num_samples, 45), 6, None, generator).view(B, num_samples, 6).sort(-1).values


class Cheat(Uniform):
    def sample(self, batch, num_samples, generator=None):
        return batch["target"][:, None].expand(-1, num_samples, -1)


def test_backtest_uniform_matches_baseline(draws_df):
    r = backtest(Uniform(), LottoDataset(draws_df, 52), tickets=200, seed=0)
    assert r["mean_matches"] == pytest.approx(0.8, abs=0.05) and r["p_value"] > 0.01


def test_backtest_cheat_is_perfect(draws_df):
    r = backtest(Cheat(), LottoDataset(draws_df, 52), tickets=3)
    assert r["mean_matches"] == 6.0 and r["rate_ge3"] == 1.0


def test_train_model_runs_and_restores_best(draws_df):
    set_seed(0); ds = LottoDataset(draws_df, 52); tr, va, _ = split_indices(len(ds))
    m = TemporalHGNN(window=52, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)
    h = train_model(m, Subset(ds, tr), Subset(ds, va), TrainConfig(epochs=3, device="cpu"))
    assert len(h["val_loss"]) <= 3 and all(np.isfinite(h["val_loss"])) and not m.training


class Repeat(Uniform):
    """One random ticket per draw, repeated: extra copies add no evidence."""
    def sample(self, batch, num_samples, generator=None):
        return super().sample(batch, 1, generator).expand(-1, num_samples, -1)


def test_backtest_pvalue_not_inflated_by_ticket_copies(draws_df):
    ds = LottoDataset(draws_df, 52)
    one, many = backtest(Repeat(), ds, tickets=1, seed=3), backtest(Repeat(), ds, tickets=50, seed=3)
    assert many["p_value"] == pytest.approx(one["p_value"])
