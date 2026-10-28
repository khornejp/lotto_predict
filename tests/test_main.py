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


def test_main_default_model_is_ar(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    out = main(["--data", str(p), "--epochs", "1", "--window", "20", "--no-update", "--out", str(tmp_path / "ck")])
    assert set(out["models"]) == {"ar"}


def test_main_tune_saves_and_reuses_params(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); ck = tmp_path / "ck"
    args = ["--data", str(p), "--epochs", "1", "--window", "20", "--no-update", "--out", str(ck)]
    tuned = main(args + ["--tune"])
    assert (ck / "ar_params.json").exists() and len(tuned["models"]["ar"]["trials"]) == 16
    again = main(args)
    assert again["models"]["ar"]["params"] == tuned["models"]["ar"]["params"]


def test_predict_next_applies_train_scaler(draws_df):
    import torch
    from data.dataset import LottoDataset, fit_scaler, split_indices
    from main import predict_next

    class Spy(torch.nn.Module):
        def sample(self, batch, num_samples, generator=None):
            self.node = batch["node"]
            return torch.arange(6).expand(1, num_samples, 6)

    ds = LottoDataset(draws_df, 20); stats = fit_scaler(ds, split_indices(len(ds))[0])
    spy = Spy(); predict_next(spy, draws_df, 20, 1, 0, "cpu", stats)
    from data.dataset import apply_scaler
    from data.loader import draws_to_multihot
    from features.build import build_sample
    raw = build_sample(draws_df, draws_to_multihot(draws_df), len(draws_df), 20)
    expect = apply_scaler({k: torch.from_numpy(raw[k]).unsqueeze(0) for k in ("node", "glob")}, stats)["node"]
    assert torch.allclose(spy.node, expect)
