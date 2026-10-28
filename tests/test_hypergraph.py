import torch

from features.build import GLOBAL_DIM, NODE_DIM
from models.hypergraph import TemporalHGNN


def make(): return TemporalHGNN(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)


def test_hgnn_loss_backward(tiny_batch):
    m = make(); loss = m.loss(tiny_batch); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_hgnn_sample_valid(tiny_batch):
    s = make().eval().sample(tiny_batch, 50, torch.Generator().manual_seed(0))
    assert s.shape == (4, 50, 6) and s.min() >= 0 and s.max() <= 44
    assert (s.diff(dim=-1) > 0).all()     # sorted and distinct


def test_hgnn_overfits_tiny(tiny_batch):
    torch.manual_seed(0); m = make(); opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = m.loss(tiny_batch).item()
    for _ in range(150):
        opt.zero_grad(); l = m.loss(tiny_batch); l.backward(); opt.step()
    assert l.item() < 0.5 * first


def test_hgnn_isolated_ball_finite(tiny_batch):
    tiny_batch["seq"][:, :, 0] = 0          # ball 0 in no hyperedge → Dv = 0 path
    assert torch.isfinite(TemporalHGNN(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM).loss(tiny_batch))
