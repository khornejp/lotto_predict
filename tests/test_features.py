from datetime import date

import numpy as np
import pytest

from data.loader import draws_to_multihot
from features.calendar_feats import calendar_features
from features.cooccurrence import centrality, cooccurrence_matrix
from features.structural import ac_value, balance_stats, sum_density
from features.temporal import gap_features, range_entropy, regime_cosine


def test_cooccurrence_two_rows():
    h = np.zeros((2, 45), np.float32); h[0, :6] = 1; h[1, [0, 6, 7, 8, 9, 10]] = 1
    co = cooccurrence_matrix(h)
    assert co[0, 1] == 0.5 and co[0, 6] == 0.5 and co[1, 2] == 0.5 and co[1, 6] == 0
    assert np.allclose(co, co.T) and np.all(np.diag(co) == 0)


def test_centrality_empty_graph():
    c = centrality(np.zeros((45, 45)))
    assert c.shape == (45, 2) and np.allclose(c[:, 0], 1.0) and np.all(c[:, 1] == 0)


def test_gap_features_values():
    h = np.zeros((7, 45), np.float32); h[[0, 2, 3], 0] = 1   # gaps [2,1], lag 4
    lag, z = gap_features(h)
    assert lag[0] == 4 and z[0] == pytest.approx(3.5355, abs=1e-3)
    assert lag[1] == 7 and z[1] == 0                          # never appeared


def test_gap_features_sparse_history_finite():
    lag, z = gap_features(np.zeros((1, 45), np.float32))
    assert np.isfinite(lag).all() and np.isfinite(z).all()


def test_range_entropy_bounds():
    one = np.zeros((3, 45), np.float32); one[:, :6] = 1
    even = np.zeros((1, 45), np.float32); even[0, [0, 9, 18, 27, 36]] = 1
    assert range_entropy(one) == 0.0 and range_entropy(even) == pytest.approx(1.0)


def test_regime_cosine_identical_is_one(draws_df):
    m = draws_to_multihot(draws_df)
    assert regime_cosine(m, short=10, long=10) == pytest.approx(1.0)


def test_ac_value():
    assert ac_value([1, 2, 3, 4, 5, 6]) == 0 and ac_value([1, 2, 4, 8, 16, 32]) == 10


def test_balance_stats():
    assert np.allclose(balance_stats([1, 2, 3, 4, 5, 6]), [0.5, 1, 1, 1, 0, 0, 0, 0])


def test_sum_density():
    assert sum_density(138) == pytest.approx(1.0) and sum_density(21) < 1e-3


def test_calendar_unit_circle():
    f = calendar_features(date(2022, 1, 29))
    assert f.shape == (4,) and f[0]**2 + f[1]**2 == pytest.approx(1) and f[2]**2 + f[3]**2 == pytest.approx(1)
