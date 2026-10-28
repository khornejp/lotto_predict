"""Pipeline entry point: update draws -> train -> backtest -> sample next-draw tickets.

Lottery draws are (near) i.i.d.; outputs are samples from a learned distribution and are
always reported next to the uniform-random baseline. This is not a deterministic predictor.

Usage:
    python main.py --model all --epochs 50
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import time
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

from data.dataset import LottoDataset, Scaler, apply_scaler, fit_scaler, split_indices
from data.loader import draws_to_multihot, load_draws
from data.updater import append_draws, update_draws
from evaluation.backtest import backtest
from features.build import GLOBAL_DIM, NODE_DIM, build_sample
from models.diffusion import SetDiffusion
from models.hypergraph import TemporalHGNN
from models.set_transformer import ARSetTransformer
from training.train import TrainConfig, set_seed, train_model
from training.tune import tune

log = logging.getLogger("lotto.main")

MODELS: dict[str, type[nn.Module]] = {"ar": ARSetTransformer, "sedd": SetDiffusion, "hgnn": TemporalHGNN}


def predict_next(
    model: nn.Module, df: pd.DataFrame, window: int, tickets: int, seed: int, device: str, stats: Scaler
) -> list[list[int]]:
    """Sample tickets for the next, not yet held, draw.

    Args:
        model: Trained model.
        df: All draws on file.
        window: Past draws per sample.
        tickets: Number of tickets.
        seed: Sampling seed.
        device: Device of ``model``.
        stats: Train-split feature scaler (the same one the model was trained with).

    Returns:
        ``tickets`` sorted lists of 6 balls, 1-indexed.
    """
    s = build_sample(df, draws_to_multihot(df), len(df), window)
    batch = apply_scaler({k: torch.from_numpy(s[k]).unsqueeze(0) for k in ("seq", "node", "glob")}, stats)  # B = 1
    batch = {k: v.to(device) for k, v in batch.items()}
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model.sample(batch, tickets, gen)[0] + 1  # Shape: [S, 6]
    return out.tolist()


def main(argv: list[str] | None = None) -> dict:
    """Run the full pipeline once.

    Args:
        argv: CLI arguments (``sys.argv[1:]`` when None).

    Returns:
        ``{"last_draw", "added", "update_warning", "sample_seed", "models": {name: {"params", "trials", "history", "backtest", "next_draw"}}}``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="lotto.xlsx")
    ap.add_argument("--model", choices=[*MODELS, "all"], default="ar")
    ap.add_argument("--window", type=int, default=52)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tickets", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42, help="training seed (reproducible experiments)")
    ap.add_argument("--sample-seed", type=int, default=None, help="ticket seed; random each run when omitted")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--add", nargs=8, type=int, metavar=("DRAW", "N1", "N2", "N3", "N4", "N5", "N6", "BONUS"))
    ap.add_argument("--no-update", action="store_true", help="skip fetching new draws")
    ap.add_argument("--tune", action="store_true", help="grid-search AR hyperparameters on val loss; saves {out}/ar_params.json")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    handlers = _attach_log(Path(args.out) / "train.log")
    try:
        return _run(args)
    finally:
        for h in handlers:
            logging.getLogger("lotto").removeHandler(h)
            h.close()


def _attach_log(path: Path) -> list[logging.Handler]:
    """Send ``lotto.*`` INFO logs to stderr and append them to ``path``."""
    logger = logging.getLogger("lotto")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr), logging.FileHandler(path, encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return handlers


def _run(args: argparse.Namespace) -> dict:
    """Pipeline body of ``main`` (logging already attached)."""
    if args.add:
        d, *nums, bonus = args.add
        append_draws(args.data, [{"draw": d, **{f"n{i+1}": v for i, v in enumerate(nums)}, "bonus": bonus}])
    added, update_warning = 0, None
    if not args.no_update:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            added = update_draws(args.data)
        update_warning = str(caught[0].message) if caught else None
    df = load_draws(args.data)
    print(f"added {added} draws (last draw {df['draw'].iloc[-1]})" + (f"; {update_warning}" if update_warning else ""))

    sample_seed = args.sample_seed if args.sample_seed is not None else secrets.randbits(32)
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    log.info("run: last draw %d, device %s, torch threads %d, model %s, tune %s",
             df["draw"].iloc[-1], cfg.device, torch.get_num_threads(), args.model, args.tune)
    t0 = time.perf_counter()
    ds = LottoDataset(df, args.window)
    tr, va, te = split_indices(len(ds))
    stats = fit_scaler(ds, tr)  # fit on train only: val/test/next-draw statistics must not leak in
    ds.normalize(stats)
    log.info("dataset: %d samples (train %d / val %d / test %d) built in %.1fs",
             len(ds), len(tr), len(va), len(te), time.perf_counter() - t0)
    results = {}
    for name in MODELS if args.model == "all" else [args.model]:
        make = partial(MODELS[name], window=args.window, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)
        params, trials = {}, None
        if name == "ar":  # tuning applies to the production model only
            params_file = Path(args.out) / "ar_params.json"
            if args.tune:
                params, trials = tune(make, Subset(ds, tr), Subset(ds, va), cfg)
                params_file.write_text(json.dumps(params))
            elif params_file.exists():
                params = json.loads(params_file.read_text())
        set_seed(args.seed)
        model = make(**{k: v for k, v in params.items() if k != "weight_decay"})
        model_cfg = replace(cfg, weight_decay=params.get("weight_decay", cfg.weight_decay))
        hist = train_model(model, Subset(ds, tr), Subset(ds, va), model_cfg)
        torch.save(model.state_dict(), Path(args.out) / f"{name}.pt")
        t0 = time.perf_counter()
        bt = backtest(model, Subset(ds, te), args.tickets, args.seed, cfg.device)
        log.info("backtest %s: mean matches %.3f vs %.3f, p %.3f (%.1fs)", name, bt["mean_matches"],
                 bt["baseline_mean_matches"], bt["p_value"], time.perf_counter() - t0)
        results[name] = {
            "params": params,
            "trials": trials,
            "history": hist,
            "backtest": bt,
            "next_draw": predict_next(model, df, args.window, args.tickets, sample_seed, cfg.device, stats),
        }
    out = {"last_draw": int(df["draw"].iloc[-1]), "added": added, "update_warning": update_warning,
           "sample_seed": sample_seed, "models": results}
    print(json.dumps({k: v for k, v in out.items() if k != "models"} | {
        "models": {n: {"params": r["params"], "backtest": r["backtest"], "next_draw": r["next_draw"]} for n, r in results.items()}
    }, indent=2))
    return out


if __name__ == "__main__":
    main()
