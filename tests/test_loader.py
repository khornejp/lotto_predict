from datetime import date

import numpy as np
import pytest

from data.loader import draw_date, draws_to_multihot, load_draws, save_draws


def test_load_draws_roundtrip(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    df = load_draws(p)
    assert list(df["draw"]) == list(range(1, len(draws_df) + 1))
    assert (df[["n1","n2","n3","n4","n5","n6"]].diff(axis=1).iloc[:, 1:] > 0).all().all()


def test_load_real_file():
    df = load_draws("lotto.xlsx")
    assert len(df) >= 1247 and df["draw"].iloc[-1] == len(df)
    assert list(df.loc[df.draw == 1247, ["n1","n2","n3","n4","n5","n6","bonus"]].iloc[0]) == [8,14,21,22,42,45,24]


@pytest.mark.parametrize("mutate,draw", [
    (lambda d: d.loc.__setitem__((d.index[4], "n2"), d.loc[d.index[4], "n1"]), 5),  # duplicate ball
    (lambda d: d.loc.__setitem__((d.index[4], "n6"), 46), 5),                        # out of range
    (lambda d: d.loc.__setitem__((d.index[4], "bonus"), d.loc[d.index[4], "n1"]), 5),# bonus == main
])
def test_load_draws_rejects_bad_row(tmp_path, draws_df, mutate, draw):
    mutate(draws_df); p = tmp_path / "b.xlsx"; save_draws(draws_df, p)
    with pytest.raises(ValueError, match=f"draw {draw}"):
        load_draws(p)


def test_load_draws_rejects_missing_draw(tmp_path, draws_df):
    p = tmp_path / "m.xlsx"; save_draws(draws_df.drop(index=9), p)
    with pytest.raises(ValueError, match="draw 10"):
        load_draws(p)


def test_multihot_six_per_row(draws_df):
    m = draws_to_multihot(draws_df)
    assert m.shape == (len(draws_df), 45) and m.dtype == np.float32
    assert (m.sum(1) == 6).all()


def test_draw_date():
    assert draw_date(1) == date(2002, 12, 7)
    assert draw_date(1000) == date(2022, 1, 29)


def test_load_draws_rejects_fractional_ball(tmp_path, draws_df):
    draws_df["n1"] = draws_df["n1"].astype(float); draws_df.loc[4, "n1"] = 1.5
    p = tmp_path / "f.xlsx"; save_draws(draws_df, p)
    with pytest.raises(ValueError, match="draw 5"):
        load_draws(p)


def test_load_draws_rejects_blank_cell(tmp_path, draws_df):
    draws_df["n3"] = draws_df["n3"].astype(float); draws_df.loc[7, "n3"] = np.nan
    p = tmp_path / "n.xlsx"; save_draws(draws_df, p)
    with pytest.raises(ValueError, match="draw 8"):
        load_draws(p)
