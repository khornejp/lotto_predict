import numpy as np
import pytest
import torch

from features.build import GLOBAL_DIM, NODE_DIM
from models.encoder import ContextEncoder
from models.sampling import gumbel_topk
from training.losses import prefix_masks, set_ce


def test_set_ce_perfect_is_zero():
    logits = torch.full((1, 1, 45), -1e4); logits[0, 0, 3] = 1e4
    pos = torch.zeros(1, 1, 45, dtype=torch.bool); pos[0, 0, 3] = True
    assert set_ce(logits, pos, torch.zeros_like(pos), torch.ones(1, 1)) == pytest.approx(0, abs=1e-4)


def test_set_ce_uniform_value():
    t = torch.tensor([[0, 1, 2, 3, 4, 5]]); pos, ban = prefix_masks(t)
    loss = set_ce(torch.zeros(1, 6, 45), pos, ban, torch.ones(1, 6))
    expect = -np.mean([np.log((6 - k) / (45 - k)) for k in range(6)])
    assert loss.item() == pytest.approx(expect, rel=1e-5)


def test_set_ce_finite_for_unsorted_target():
    logits = torch.randn(1, 1, 45).expand(1, 6, 45)   # step-independent logits
    a, b = torch.tensor([[0, 5, 9, 20, 30, 44]]), torch.tensor([[44, 30, 20, 9, 5, 0]])
    la = set_ce(logits, *prefix_masks(a), torch.ones(1, 6))
    lb = set_ce(logits, *prefix_masks(b), torch.ones(1, 6))
    assert torch.isfinite(la) and torch.isfinite(lb)


def test_gumbel_topk_distinct_and_respects_ban():
    g = torch.Generator().manual_seed(0)
    ban = torch.zeros(500, 45, dtype=torch.bool); ban[:, :10] = True
    out = gumbel_topk(torch.zeros(500, 45), 6, ban, g)
    assert out.shape == (500, 6) and (out >= 10).all()
    assert all(len(set(r.tolist())) == 6 for r in out)


def test_encoder_shape(tiny_batch):
    enc = ContextEncoder(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)
    out = enc(tiny_batch["seq"], tiny_batch["node"], tiny_batch["glob"])
    assert out.shape == (4, 46, 64)


def test_set_ce_grad_finite_with_empty_step():
    logits = torch.randn(1, 2, 45, requires_grad=True)
    pos = torch.zeros(1, 2, 45, dtype=torch.bool); pos[0, 0, 3] = True   # step 1 has no positive
    set_ce(logits, pos, torch.zeros_like(pos), torch.tensor([[1.0, 0.0]])).backward()
    assert torch.isfinite(logits.grad).all()
