import numpy as np
import torch

from data.dataset import LottoDataset, split_indices
from data.loader import draws_to_multihot
from features.build import CARRY_COL, GLOBAL_DIM, NODE_DIM, build_sample


def test_build_sample_shapes(draws_df):
    s = build_sample(draws_df, draws_to_multihot(draws_df), t=60, window=52)
    assert s["seq"].shape == (52, 45) and s["node"].shape == (45, NODE_DIM) and s["glob"].shape == (GLOBAL_DIM,)
    assert s["target"].tolist() == sorted(draws_df.loc[60, ["n1","n2","n3","n4","n5","n6"]] - 1)
    assert all(np.isfinite(s[k]).all() for k in ("seq", "node", "glob"))


def test_build_sample_no_lookahead(draws_df):
    m = draws_to_multihot(draws_df); m2 = m.copy(); m2[60] = np.roll(m2[60], 1)
    a, b = build_sample(draws_df, m, 60, 52), build_sample(draws_df, m2, 60, 52)
    for k in ("seq", "node", "glob"):
        assert np.array_equal(a[k], b[k])


def test_build_sample_future_has_no_target(draws_df):
    m = draws_to_multihot(draws_df)
    s = build_sample(draws_df, m, t=len(m), window=52)
    assert s["target"] is None and s["seq"].shape == (52, 45)


def test_dataset_len_and_item(draws_df):
    ds = LottoDataset(draws_df, window=52)
    assert len(ds) == len(draws_df) - 52 and ds.draw_numbers[0] == 53
    assert ds[0]["target"].dtype == torch.int64 and ds[0]["seq"].dtype == torch.float32


def test_split_indices_chronological():
    tr, va, te = split_indices(100)
    assert tr == list(range(80)) and va == list(range(80, 90)) and te == list(range(90, 100))


def test_build_sample_ignores_target_row_in_df(draws_df):
    m = draws_to_multihot(draws_df); df2 = draws_df.copy()
    df2.loc[60, ["n1","n2","n3","n4","n5","n6","bonus"]] = [1, 2, 3, 4, 5, 6, 7]
    a, b = build_sample(draws_df, m, 60, 52), build_sample(df2, m, 60, 52)
    for k in ("seq", "node", "glob"):
        assert np.array_equal(a[k], b[k])


def test_new_node_features_recency(draws_df):
    m = draws_to_multihot(draws_df); s = build_sample(draws_df, m, 60, 52)
    last = draws_df.loc[59, ["n1","n2","n3","n4","n5","n6"]].to_numpy(dtype=int) - 1
    carry = s["node"][:, CARRY_COL]
    assert carry[last].tolist() == [1.0] * 6 and carry.sum() == 6
