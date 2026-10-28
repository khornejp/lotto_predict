from streamlit.testing.v1 import AppTest

from data.loader import load_draws, save_draws


def test_app_renders_last_draw(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception and any("120" in m.value for m in [*at.markdown, *at.header, *at.subheader])


def test_app_manual_add_rejects_bad_draw(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("../app.py").run(timeout=30)
    at.number_input(key="add_draw").set_value(125); at.button(key="add_submit").click().run(timeout=30)
    assert at.error and len(load_draws(p)) == 120


def test_app_run_shows_results(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("../app.py").run(timeout=30)
    at.number_input(key="epochs").set_value(1)
    at.number_input(key="window").set_value(20); at.checkbox(key="auto_update").uncheck()
    at.button(key="run").click().run(timeout=180)
    assert not at.exception and len(at.dataframe) == 2


def test_app_manual_add_requires_all_numbers(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("../app.py").run(timeout=30)
    at.button(key="add_submit").click().run(timeout=30)
    assert at.error and len(load_draws(p)) == 120


def test_app_manual_add_locked_file_shows_error(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    def locked(*a): raise PermissionError("file in use")
    monkeypatch.setattr("data.loader.os.replace", locked)
    at = AppTest.from_file("../app.py").run(timeout=30)
    for i, v in enumerate([1, 2, 3, 4, 5, 6]):
        at.number_input(key=f"add_n{i+1}").set_value(v)
    at.number_input(key="add_bonus").set_value(7); at.button(key="add_submit").click().run(timeout=30)
    assert not at.exception and at.error and len(load_draws(p)) == 120


def test_app_offers_no_model_choice(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    assert len(AppTest.from_file("../app.py").run(timeout=30).selectbox) == 0


def test_app_tune_checkbox_off_by_default(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    assert AppTest.from_file("../app.py").run(timeout=30).checkbox(key="tune").value is False
