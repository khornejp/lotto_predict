import io

import pytest

from data.loader import load_draws, save_draws
from data.updater import UpdateError, append_draws, fetch_draw, update_draws


def fake_fetch(new):                     # new: list of row dicts
    it = iter(new)
    return lambda n: next(it, None)


def row(draw, nums, bonus):
    return {"draw": draw, **{f"n{i+1}": v for i, v in enumerate(nums)}, "bonus": bonus}


def test_update_appends_new_draws(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    added = update_draws(p, fake_fetch([row(121, [1,2,3,4,5,6], 7), row(122, [8,9,10,11,12,13], 14)]))
    assert added == 2 and len(load_draws(p)) == 122
    assert len(load_draws(p.with_suffix(".xlsx.bak"))) == 120


def test_update_up_to_date_writes_nothing(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); before = p.stat().st_mtime_ns
    assert update_draws(p, fake_fetch([])) == 0 and p.stat().st_mtime_ns == before


def test_update_network_error_warns_and_keeps_file(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    def boom(n): raise UpdateError("down")
    with pytest.warns(UserWarning, match="draw 121"):
        assert update_draws(p, boom) == 0
    assert len(load_draws(p)) == 120


def test_update_invalid_fetched_row_warns_and_keeps_good_rows(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    with pytest.warns(UserWarning, match="draw 122"):
        added = update_draws(p, fake_fetch([row(121, [1,2,3,4,5,6], 7), row(122, [1,1,3,4,5,6], 7)]))
    assert added == 1 and len(load_draws(p)) == 121


def test_update_locked_file_warns_and_cleans_tmp(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    def locked(*a): raise PermissionError("file in use")
    monkeypatch.setattr("data.loader.os.replace", locked)
    with pytest.warns(UserWarning, match="file in use"):
        assert update_draws(p, fake_fetch([row(121, [1,2,3,4,5,6], 7)])) == 0
    assert len(load_draws(p)) == 120 and not (tmp_path / "l.tmp.xlsx").exists()


def test_append_wrong_draw_number(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    with pytest.raises(ValueError, match="draw 121"):
        append_draws(p, [row(125, [1,2,3,4,5,6], 7)])


@pytest.mark.parametrize("body,expect", [
    (b'{"resultCode":null,"data":{"list":[{"ltEpsd":1248,"tm1WnNo":1,"tm2WnNo":2,"tm3WnNo":3,'
     b'"tm4WnNo":4,"tm5WnNo":5,"tm6WnNo":6,"bnsWnNo":7,"ltRflYmd":"20261031"}]}}',
     {"draw": 1248, "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "bonus": 7}),
    (b'{"resultCode":null,"resultMessage":null,"data":{"list":[]}}', None),
])
def test_fetch_draw_parses(monkeypatch, body, expect):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(body))
    assert fetch_draw(1248) == expect


@pytest.mark.parametrize("body", [b"<html>", b'{"data":{"list":[{"ltEpsd":1200}]}}'])
def test_fetch_draw_bad_body_raises(monkeypatch, body):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(UpdateError):
        fetch_draw(1248)
