from data.loader import load_draws, save_draws
from main import main


def test_main_smoke(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    out = main(["--data", str(p), "--model", "all", "--epochs", "1", "--window", "20", "--no-update",
                "--tickets", "2", "--out", str(tmp_path / "ck")])
    assert set(out["models"]) == {"ar", "sedd", "hgnn"} and isinstance(out["sample_seed"], int)
    for r in out["models"].values():
        assert len(r["next_draw"]) == 2
        assert all(len(set(t)) == 6 and 1 <= min(t) and max(t) <= 45 for t in r["next_draw"])
        assert "p_value" in r["backtest"] and len(r["history"]["val_loss"]) == 1
    assert (tmp_path / "ck" / "ar.pt").exists()


def test_main_manual_add(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    out = main(["--data", str(p), "--model", "hgnn", "--epochs", "1", "--window", "20", "--no-update",
                "--add", "121", "1", "2", "3", "4", "5", "6", "7", "--out", str(tmp_path / "ck")])
    assert out["last_draw"] == 121 and len(load_draws(p)) == 121
