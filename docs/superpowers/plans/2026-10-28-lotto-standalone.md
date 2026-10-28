# Lotto Standalone End-to-End Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone pipeline that turns `lotto.xlsx` into leakage-free features, trains three joint-distribution models (Autoregressive Set Transformer, absorbing-state Discrete Diffusion, Temporal Hypergraph NN), and backtests their samples against the uniform-random baseline.

**Architecture:** `data/` loads and validates draws and builds a `Dataset` whose sample `t` uses only draws `< t`. `features/` holds pure NumPy feature functions. `models/` holds three `nn.Module`s that share one duck-typed interface (`loss(batch)`, `sample(batch, num_samples, generator)`). `training/` holds one shared set loss and one training loop. `evaluation/` scores sampled tickets against the actual draw. `main.py` runs on every invocation: it first appends new winning numbers to `lotto.xlsx` (auto fetch, or manual `--add`), retrains, and samples fresh tickets with a new seed each run. It prints JSON.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, NumPy 1.26, pandas 2.3, networkx 3.2 (PageRank), scipy 1.13 (t-test, hypergeometric), openpyxl, pytest.

**Spec:** `standalone_ml_models_and_features_for_lottery_prediction.md` (models §1, features §2, tensors §3) plus `AGENT.md` (coding standards).

## Global Constraints

- Interpreter: `C:/ProgramData/miniconda3/envs/Predict/python.exe` (conda env `Predict`). Every command below uses it. Do not use the base `python`.
- No dependency beyond what Predict already has, except `pytest` (dev only) and `streamlit` (UI, Task 11). No Lightning.
- Layout (AGENT.md §3): `data/`, `features/`, `models/`, `training/`, `evaluation/`, `main.py`, plus `tests/`. Keep `lotto.xlsx` in the repo root (default `--data` path).
- Every module: type hints, Google-style docstrings, and a shape comment on every tensor op and `forward` (`# Shape: [B, W, 45]`).
- Every entry point calls `set_seed(seed)` (torch, numpy, random, cuda).
- No dependency on the legacy CatRank + MC pipeline.
- Balls are 0-indexed (`0..44`) inside tensors and 1-indexed (`1..45`) in files and printed output. Convert only in `data/loader.py` and `main.py`.
- Causality: features for target draw index `t` use only rows `< t` of the draw history. The target draw's calendar date is allowed, because it is known before the draw.
- Never claim deterministic prediction. Report every model metric next to the uniform baseline.
- Device: `cuda` if available, else `cpu`.
- Network: stdlib `urllib` only, with default headers. Never spoof a browser or work around bot checks. Tests never touch the network.
- `lotto.xlsx` is user data. Write to it only through `append_draws`, which validates first, backs up to `lotto.xlsx.bak`, and then writes atomically.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Look-ahead leakage:** if draw `t` changes, sample `t`'s inputs must stay the same. Test: `test_build_sample_no_lookahead` (Task 3).
2. **Malformed spreadsheet:** a duplicate ball, a ball outside 1..45, a bonus equal to a main ball, or a missing draw number must raise `ValueError` naming the draw, not train silently. Test: `test_load_draws_rejects_*` (Task 1).
3. **Invalid tickets:** every model's `sample` must return 6 distinct balls in `0..44`, sorted, even after many sampling steps. Test: `test_*_sample_valid` (Tasks 5–7).
4. **Sparse early history:** a ball with fewer than 2 gaps, or a window with no co-occurrence, must give finite features (no NaN or inf). Test: `test_gap_features_sparse_history_finite`, `test_centrality_empty_graph` (Task 2).
5. **Next-draw prediction:** `t == T` has no target, so `build_sample` must still work and `predict_next` must return valid tickets. Test: `test_build_sample_future_has_no_target` (Task 3), `test_main_smoke` (Task 10).
6. **Auto update failing or returning junk:** a network error or HTML response must warn and keep training on the existing file. A bad fetched row must never corrupt `lotto.xlsx`, and a backup must exist before any write. Test: `test_update_network_error_warns_and_keeps_file`, `test_update_rejects_invalid_fetched_row` (Task 9).

---

## File Structure

```
pytest.ini                     # pythonpath = . ; testpaths = tests
.gitignore                     # __pycache__/, checkpoints/, .pytest_cache/
data/loader.py                 # load_draws, validate_draws, save_draws, draws_to_multihot, draw_date
data/updater.py                # fetch_draw, append_draws, update_draws (auto + manual add, .bak backup)
data/dataset.py                # LottoDataset, split_indices
features/cooccurrence.py       # cooccurrence_matrix, centrality
features/temporal.py           # gap_features, range_entropy, regime_cosine
features/structural.py         # ac_value, balance_stats, sum_density
features/calendar_feats.py     # calendar_features  (not "calendar.py": avoids stdlib-name confusion)
features/build.py              # build_sample, NODE_DIM, GLOBAL_DIM
models/encoder.py              # ContextEncoder (shared by AR + diffusion)
models/sampling.py             # gumbel_topk
models/set_transformer.py      # ARSetTransformer
models/diffusion.py            # SetDiffusion
models/hypergraph.py           # TemporalHGNN
training/losses.py             # set_ce, prefix_masks
training/train.py              # set_seed, TrainConfig, train_model
evaluation/backtest.py         # count_matches, backtest
main.py                        # CLI: update -> train -> backtest -> predict next draw
app.py                         # Streamlit UI over main() + manual entry
tests/conftest.py              # synthetic_draws fixture, tiny_batch fixture, write_xlsx helper
tests/test_*.py
```

Every package directory gets an empty `__init__.py`.

**Shared model interface:** Tasks 5–7 implement this interface and Tasks 8–9 rely on it. It is duck-typed, so there is no base class.
- `loss(batch: dict[str, Tensor]) -> Tensor` returns a scalar. The batch keys are `seq [B,W,45] float`, `node [B,45,NODE_DIM] float`, `glob [B,GLOBAL_DIM] float`, and `target [B,6] long`.
- `sample(batch, num_samples: int, generator: torch.Generator | None = None) -> LongTensor [B, num_samples, 6]` returns distinct balls in `0..44`, sorted ascending. It must not read `batch["target"]`.

---

### Task 1: Project scaffold + draw loader

**Files:**
- Create: `pytest.ini`, `.gitignore`, all `__init__.py`, `data/loader.py`, `tests/conftest.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Produces:
  - `load_draws(path: str | Path) -> pd.DataFrame` with columns `draw:int, n1..n6:int (sorted asc, 1..45), bonus:int, date:datetime64`, rows sorted by `draw` ascending, index `0..T-1`.
  - `draws_to_multihot(df: pd.DataFrame) -> np.ndarray` with shape `[T, 45]`, `float32`, and exactly 6 ones per row (bonus excluded).
  - `draw_date(draw_no: int) -> datetime.date`, computed as `date(2002,12,7) + timedelta(weeks=draw_no-1)`.
  - `validate_draws(df: pd.DataFrame) -> None` raises the `ValueError`s described in Step 4. `load_draws` calls it, and the Task 9 updater reuses it.
  - `save_draws(df: pd.DataFrame, path: str | Path) -> None` writes the real file layout: a header row `count,1,2,3,4,5,6,7`, then rows in descending draw order. It writes to `path.with_suffix(".tmp.xlsx")` first and then calls `os.replace`, so a crash cannot leave a half-written `lotto.xlsx`.
  - conftest: `synthetic_draws(n: int = 120, seed: int = 0) -> pd.DataFrame` (same schema as `load_draws`) as a plain function, and a fixture `draws_df` = `synthetic_draws()`. Tests write files with `save_draws`.

- [ ] **Step 1: Set up the env and repo**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pip install pytest` and `git init`.
Expected: pytest installs, and git reports an empty repo created.

- [ ] **Step 2: Write the failing tests** in `tests/test_loader.py`

```python
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
    (lambda d: d.__setitem__((d.index[4], "n2"), d.loc[d.index[4], "n1"]), 5),  # duplicate ball
    (lambda d: d.__setitem__((d.index[4], "n6"), 46), 5),                        # out of range
    (lambda d: d.__setitem__((d.index[4], "bonus"), d.loc[d.index[4], "n1"]), 5),# bonus == main
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.loader'`

- [ ] **Step 4: Implement `data/loader.py`**

Read the file with `pd.read_excel(path, header=0, engine="openpyxl")`. The explicit engine lets the `.xlsx.bak` backup load too. Rename `count→draw`, `1..6→n1..n6`, `7→bonus`, then sort each row's mains. Validate in this order and report the first failure as `ValueError(f"draw {n}: <reason>")`: values in 1..45, 6 distinct mains, bonus not among the mains, and `draw` equal to `1..T` with no gaps. A missing draw is reported by the first missing number. Add `date` with `draw_date`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_loader.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add . && git commit -m "feat: scaffold project and validated draw loader"
```

---

### Task 2: Feature functions

**Files:**
- Create: `features/cooccurrence.py`, `features/temporal.py`, `features/structural.py`, `features/calendar_feats.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `history` arrays shaped like `draws_to_multihot` output (`[t, 45]` float32 0/1).
- Produces:
  - `cooccurrence_matrix(window: np.ndarray) -> np.ndarray` with shape `[45,45]`, computed as `window.T @ window / len(window)` with the diagonal zeroed. An empty window gives all zeros.
  - `centrality(co: np.ndarray) -> np.ndarray` with shape `[45,2]`. Column 0 is weighted PageRank × 45 (networkx `pagerank(weight="weight")`; an all-zero `co` gives all ones). Column 1 is unweighted degree `(co > 0).sum(1) / 44`.
  - `gap_features(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(lag [45], z [45])`. For ball `i` appearing at rows `a_1<…<a_m`: `lag = t - a_m` (where `t = len(history)`), or `lag = t` if `m == 0`. `gaps = diff(a)`, and `z = (lag - mean(gaps)) / (std(gaps, ddof=1) + 1e-6)` clipped to `[-5, 5]`. `z = 0` when `len(gaps) < 2`.
  - `range_entropy(window: np.ndarray) -> float`: ball counts summed into 5 bins (`1–9, 10–18, 19–27, 28–36, 37–45`). Shannon entropy divided by `log(5)` gives a value in `[0, 1]`. Zero counts give 0.0.
  - `regime_cosine(history: np.ndarray, short: int = 10, long: int = 52) -> float`: cosine between the column sums of the last `short` rows and the last `long` rows. A zero vector gives 0.0.
  - `ac_value(nums: Sequence[int]) -> int`: the number of distinct pairwise absolute differences minus 5.
  - `balance_stats(nums: Sequence[int]) -> np.ndarray` with shape `[8]`: `[odd/6, (n<=22)/6, consecutive_pairs/5, decade counts/6 for 1–9, 10–19, 20–29, 30–39, 40–45]`. Input is 1-indexed.
  - `sum_density(total: int) -> float`: `exp(-z²/2)` with `z = (total - 138) / sqrt(6*(45**2-1)/12 * 39/44)`, the exact mean and std of a 6-of-45 draw. It needs no fit, so it cannot leak.
  - `calendar_features(d: date) -> np.ndarray` with shape `[4]`: `[sin, cos]` of `2π·(day-1)/days_in_month`, then `[sin, cos]` of `2π·(day_of_year-1)/365.25`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the four modules** with the signatures above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_features.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add features tests/test_features.py && git commit -m "feat: co-occurrence, temporal, structural, calendar features"
```

---

### Task 3: Sample builder + Dataset

**Files:**
- Create: `features/build.py`, `data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces:
  - `NODE_DIM = 5`, `GLOBAL_DIM = 16`.
  - `build_sample(df: pd.DataFrame, multihot: np.ndarray, t: int, window: int) -> dict[str, np.ndarray | None]`, valid for `window <= t <= T`. It asserts `window >= 10`, because `range_entropy` and `regime_cosine` read the last 10 rows. Keys:
    - `seq [window,45]` = `multihot[t-window:t]`
    - `node [45,5]` = `[log1p(lag), z, window freq = seq.mean(0), pagerank×45, degree]`. `lag` and `z` come from `gap_features(multihot[:t])`, and centrality from `cooccurrence_matrix(seq)`.
    - `glob [16]` = the previous draw `t-1`'s `[ac_value/10, *balance_stats, sum_density]`, then `range_entropy(multihot[t-10:t])`, then `regime_cosine(multihot[:t], 10, window)`, then `calendar_features(draw_date(t+1))`.
    - `target [6]` int64 0-indexed sorted, or `None` when `t == T`.
  - `LottoDataset(df: pd.DataFrame, window: int = 52)`, a `torch.utils.data.Dataset`. Item `i` maps to draw index `t = window + i`, so `len = T - window`. Items are precomputed in `__init__` as float32/int64 tensors with keys `seq, node, glob, target`. Attribute `draw_numbers: list[int]` holds the 1-indexed draw number of each item.
  - `split_indices(n: int, train: float = 0.8, val: float = 0.1) -> tuple[list[int], list[int], list[int]]`: contiguous, chronological, and not shuffled.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `build_sample`, `LottoDataset`, and `split_indices`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_dataset.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add features/build.py data/dataset.py tests/test_dataset.py && git commit -m "feat: leakage-free sample builder and dataset"
```

---

### Task 4: Set loss, Gumbel top-k, shared encoder

**Files:**
- Create: `training/losses.py`, `models/sampling.py`, `models/encoder.py`
- Modify: `tests/conftest.py` (add `tiny_batch`)
- Test: `tests/test_core.py`

**Interfaces:**
- Produces:
  - `set_ce(logits: Tensor[B,K,45], positive: BoolTensor[B,K,45], banned: BoolTensor[B,K,45], weight: Tensor[B,K]) -> Tensor` (scalar). Per step: `logp = log_softmax(logits.masked_fill(banned, -inf))`, and `loss_k = -logsumexp(logp[positive])`. This scores the probability of picking **any** still-missing winning ball, so it does not penalize valid permutations (AGENT.md §4). The result is `sum(loss_k * weight) / weight.sum()`.
  - `prefix_masks(target: LongTensor[B,6]) -> tuple[BoolTensor[B,6,45], BoolTensor[B,6,45]]`. Step `k` gets `positive = onehot(target[:, k:])` and `banned = onehot(target[:, :k])`.
  - `gumbel_topk(logits: Tensor[N,45], k: int, banned: BoolTensor[N,45] | None = None, generator: torch.Generator | None = None) -> LongTensor[N,k]`. This draws `k` balls without replacement under Plackett–Luce: `topk(logits + Gumbel noise)` with banned entries set to `-inf`. Uniform noise is clamped to `[1e-10, 1-1e-10]`.
  - `ContextEncoder(window: int, node_dim: int, global_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2)` with `forward(seq, node, glob) -> Tensor[B,46,d_model]`. Ball token `i` = `Linear(window→d)(seq[:,:,i]) + Linear(node_dim→d)(node[:,i]) + ball_emb[i]`. Token 45 = `Linear(global_dim→d)(glob)`. Apply `nn.TransformerEncoder` (`batch_first=True`, `dim_feedforward=128`) and return its output. The model is permutation-equivariant over balls apart from `ball_emb`. Small `d` limits overfitting on ~950 training draws.
  - conftest fixture `tiny_batch`: `B=4, W=8`, random `seq/node/glob` with `NODE_DIM`/`GLOBAL_DIM`, and `target` of 4 rows of 6 distinct sorted balls (seeded).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the three modules**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_core.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add training/losses.py models/sampling.py models/encoder.py tests && git commit -m "feat: order-agnostic set loss, gumbel top-k, context encoder"
```

---

### Task 5: Autoregressive Set Transformer

**Files:**
- Create: `models/set_transformer.py`
- Test: `tests/test_set_transformer.py`

**Interfaces:**
- Consumes: `ContextEncoder`, `set_ce`, `prefix_masks`, `gumbel_topk`.
- Produces: `ARSetTransformer(window: int, node_dim: int, global_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2)` implementing the shared model interface, plus `step_logits(memory, prefix) -> Tensor[B,6,45]`.
  - Decoder input token `0` = learned BOS, token `k` = `ball_emb_dec[prefix[:, k-1]] + step_emb[k]`. It runs through `nn.TransformerDecoder` with a causal `tgt_mask` over `memory`.
  - Pointer logits: `h @ memory[:, :45].transpose(1, 2) / sqrt(d)` → `[B,6,45]`.
  - `loss`: shuffle each row of `target` with a fresh random permutation (`argsort(rand)`) so the model learns P(set) rather than P(sorted sequence). Then return `set_ce(step_logits(memory, perm_target), *prefix_masks(perm_target), ones)`.
  - `sample`: encode once and `repeat_interleave` the memory `num_samples` times. Run 6 decoding steps, each with `gumbel_topk(k=1, banned=already chosen)`. Return sorted `[B,S,6]`. Beam search is not included (see skipped list).

- [ ] **Step 1: Write the failing tests**

```python
def make(): return ARSetTransformer(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)

def test_ar_loss_backward(tiny_batch):
    m = make(); loss = m.loss(tiny_batch); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters() if p.requires_grad)

def test_ar_sample_valid(tiny_batch):
    s = make().eval().sample(tiny_batch, 50, torch.Generator().manual_seed(0))
    assert s.shape == (4, 50, 6) and s.min() >= 0 and s.max() <= 44
    assert (s.diff(dim=-1) > 0).all()     # sorted and distinct

def test_ar_overfits_tiny(tiny_batch):
    torch.manual_seed(0); m = make(); opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = m.loss(tiny_batch).item()
    for _ in range(150):
        opt.zero_grad(); l = m.loss(tiny_batch); l.backward(); opt.step()
    assert l.item() < 0.5 * first
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_set_transformer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ARSetTransformer`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_set_transformer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add models/set_transformer.py tests/test_set_transformer.py && git commit -m "feat: autoregressive set transformer"
```

---

### Task 6: Discrete diffusion over sets (absorbing-state, SEDD/MDLM-style)

**Files:**
- Create: `models/diffusion.py`
- Test: `tests/test_diffusion.py`

**Interfaces:**
- Consumes: `ContextEncoder`, `set_ce`, `gumbel_topk`.
- Produces: `SetDiffusion(window, node_dim, global_dim, d_model=64, nhead=4, layers=2, dropout=0.2, steps: tuple[int, ...] = (2, 2, 2))` implementing the shared model interface.
  - Slot token = learned `mask_emb` if masked, else `ball_emb[target]`. Slots get **no positional embedding**, and a bidirectional `nn.TransformerDecoder` (no `tgt_mask`) runs over `memory`. Pointer logits work as in Task 5 → `[B,6,45]`. The model is permutation-equivariant over slots, which matches set semantics.
  - `loss`: per row, draw a mask rate `r ~ U(0,1)` and mask each slot with probability `r`, forcing at least one masked slot. Each masked slot gets `positive = onehot(all masked target balls)` and `banned = onehot(revealed balls)`, weighted 1. Revealed slots get weight 0. Return the `set_ce` result.
  - `sample`: start from all slots masked. For each `k` in `steps`, run the model, take the logits of any masked slot (they are identical by equivariance), and reveal `gumbel_topk(k, banned=revealed)`. `sum(steps) == 6` is asserted in `__init__`. Return sorted `[B,S,6]`.

- [ ] **Step 1: Write the failing tests**: copy the three tests from Task 5, with `SetDiffusion` in place of `ARSetTransformer` and names `test_diffusion_*`. Add this test:

```python
def test_diffusion_rejects_bad_steps():
    with pytest.raises(AssertionError):
        SetDiffusion(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM, steps=(3, 2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `SetDiffusion`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_diffusion.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add models/diffusion.py tests/test_diffusion.py && git commit -m "feat: absorbing-state discrete diffusion over sets"
```

---

### Task 7: Temporal Hypergraph NN

**Files:**
- Create: `models/hypergraph.py`
- Test: `tests/test_hypergraph.py`

**Interfaces:**
- Consumes: `set_ce`, `prefix_masks`, `gumbel_topk`. It does **not** use `ContextEncoder`: the hypergraph is its own inductive bias.
- Produces: `TemporalHGNN(window, node_dim, global_dim, d_model=64, layers=2, dropout=0.2)` implementing the shared model interface.
  - Incidence `H = seq.transpose(1, 2)` has shape `[B,45,W]`, with one hyperedge per past draw.
  - Edge weight `w_e = γ^(W-1-e)` uses a learnable `γ = sigmoid(θ)` with `θ` initialized at `logit(0.97)`. Recent draws therefore weigh more.
  - Layer (HGNN, Feng et al. 2019): `X' = X + Dropout(GELU(Dv⁻¹ H diag(w) De⁻¹ Hᵀ X Θ))`, with `De = 6` and `Dv = H @ w` clamped to at least `1e-6`. `X⁰ = Linear(node_dim→d)(node) + ball_emb`.
  - Readout: `logits [B,45] = Linear(d→1)(X + Linear(global_dim→d)(glob)[:, None])`.
  - `loss`: apply a random per-row permutation of `target`, then `set_ce(logits[:, None].expand(-1, 6, -1), *prefix_masks(perm), ones)`. This is the Plackett–Luce set likelihood with one sampled order.
  - `sample`: `gumbel_topk(logits.repeat_interleave(S, 0), 6)`, reshaped and sorted to `[B,S,6]`.

- [ ] **Step 1: Write the failing tests**: copy the three tests from Task 5, with `TemporalHGNN` in place of `ARSetTransformer` and names `test_hgnn_*`. Add this test:

```python
def test_hgnn_isolated_ball_finite(tiny_batch):
    tiny_batch["seq"][:, :, 0] = 0          # ball 0 in no hyperedge → Dv = 0 path
    assert torch.isfinite(TemporalHGNN(window=8, node_dim=NODE_DIM, global_dim=GLOBAL_DIM).loss(tiny_batch))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_hypergraph.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `TemporalHGNN`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_hypergraph.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add models/hypergraph.py tests/test_hypergraph.py && git commit -m "feat: temporal hypergraph neural network"
```

---

### Task 8: Training loop + backtest

**Files:**
- Create: `training/train.py`, `evaluation/backtest.py`
- Test: `tests/test_train_eval.py`

**Interfaces:**
- Consumes: the shared model interface and `LottoDataset`/`Subset`.
- Produces:
  - `set_seed(seed: int) -> None`
  - `@dataclass TrainConfig(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-2, patience: int = 8, grad_clip: float = 1.0, seed: int = 42, device: str = "cuda" if available else "cpu")`
  - `train_model(model: nn.Module, train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig) -> dict[str, list[float]]` returns `{"train_loss": [...], "val_loss": [...]}`. It uses AdamW, shuffles the train loader, clips gradients to `grad_clip`, and early-stops on val loss with `patience`. It restores the best `state_dict` into `model` before returning, and leaves the model in `eval()` mode.
  - `count_matches(samples: LongTensor[N,S,6], targets: LongTensor[N,6]) -> np.ndarray[N,S]`
  - `backtest(model, ds: Dataset, tickets: int = 5, seed: int = 0, device: str = "cpu") -> dict[str, float]` with keys:
    - `n_draws`, `tickets`
    - `mean_matches` and `baseline_mean_matches = 36/45 = 0.8`
    - `t_stat` and `p_value`: one-sample t-test of per-ticket matches against 0.8 (`scipy.stats.ttest_1samp(alternative="greater")`)
    - `rate_ge3` and `baseline_rate_ge3 = scipy.stats.hypergeom(45, 6, 6).sf(2)`
    - `unique_ratio`: distinct tickets / total tickets per draw, averaged

- [ ] **Step 1: Write the failing tests**

```python
class Uniform(nn.Module):
    def sample(self, batch, num_samples, generator=None):
        B = batch["seq"].shape[0]
        return gumbel_topk(torch.zeros(B * num_samples, 45), 6, None, generator).view(B, num_samples, 6).sort(-1).values

class Cheat(Uniform):
    def sample(self, batch, num_samples, generator=None):
        return batch["target"][:, None].expand(-1, num_samples, -1)

def test_backtest_uniform_matches_baseline(draws_df):
    r = backtest(Uniform(), LottoDataset(draws_df, 52), tickets=200, seed=0)
    assert r["mean_matches"] == pytest.approx(0.8, abs=0.05) and r["p_value"] > 0.01

def test_backtest_cheat_is_perfect(draws_df):
    r = backtest(Cheat(), LottoDataset(draws_df, 52), tickets=3)
    assert r["mean_matches"] == 6.0 and r["rate_ge3"] == 1.0

def test_train_model_runs_and_restores_best(draws_df):
    set_seed(0); ds = LottoDataset(draws_df, 52); tr, va, _ = split_indices(len(ds))
    m = TemporalHGNN(window=52, node_dim=NODE_DIM, global_dim=GLOBAL_DIM)
    h = train_model(m, Subset(ds, tr), Subset(ds, va), TrainConfig(epochs=3, device="cpu"))
    assert len(h["val_loss"]) <= 3 and all(np.isfinite(h["val_loss"])) and not m.training
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_train_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `training/train.py` and `evaluation/backtest.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_train_eval.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add training/train.py evaluation tests/test_train_eval.py && git commit -m "feat: training loop with early stopping and baseline backtest"
```

---

### Task 9: Winning-number updater (auto fetch + manual add)

**Files:**
- Create: `data/updater.py`
- Test: `tests/test_updater.py`

**Interfaces:**
- Consumes: `load_draws`, `validate_draws`, `save_draws`, `draw_date`.
- Produces:
  - `DEFAULT_API = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do?srchLtEpsd={n}"`. Verified 2026-10-28 with default urllib headers: draw 1247 returns `{"data":{"list":[{"ltEpsd":1247,"tm1WnNo":8,…,"tm6WnNo":45,"bnsWnNo":24,"ltRflYmd":"20261024",…}]}}`, which matches `lotto.xlsx`. A draw not held yet returns `{"data":{"list":[]}}`. The old `common.do?method=getLottoNumber` now returns HTML; do not use it.
  - `class UpdateError(RuntimeError)`
  - `fetch_draw(draw_no: int, url: str = DEFAULT_API, timeout: float = 10.0) -> dict[str, int] | None` uses stdlib `urllib.request.urlopen` with default headers. Do not spoof a browser or bypass bot checks. It parses the JSON and returns `{"draw", "n1".."n6", "bonus"}` built from `ltEpsd`, `tm1WnNo..tm6WnNo`, and `bnsWnNo` of `data.list[0]`. It returns `None` when `data.list` is empty (draw not held yet). It raises `UpdateError` on a network error, a non-JSON body, a missing key, or `ltEpsd != draw_no`.
  - `append_draws(path: str | Path, rows: list[dict[str, int]]) -> None` appends `rows` to the loaded draws, runs `validate_draws` on the result (so a wrong draw number or ball raises `ValueError` before anything is written), copies the file to `path.with_suffix(".xlsx.bak")`, and then calls `save_draws`.
  - `update_draws(path: str | Path, fetch: Callable[[int], dict | None] = fetch_draw, max_new: int = 20) -> int` fetches draws `last+1, last+2, …` until `None` or until `max_new` draws. On `UpdateError` it calls `warnings.warn(f"auto update failed at draw {n}: {e}")` and keeps the draws fetched so far. It appends them with `append_draws` when there are any, and returns the number added.

- [ ] **Step 1: Write the failing tests**

```python
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

def test_update_rejects_invalid_fetched_row(tmp_path, draws_df):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p)
    with pytest.raises(ValueError, match="draw 121"):
        update_draws(p, fake_fetch([row(121, [1,1,3,4,5,6], 7)]))
    assert len(load_draws(p)) == 120

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
```

`fetch_draw` must call `urllib.request.urlopen` through the module attribute (`urllib.request.urlopen(...)`), not through `from urllib.request import urlopen`. Otherwise the monkeypatch does not apply.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_updater.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `data/updater.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_updater.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add data/updater.py tests/test_updater.py && git commit -m "feat: winning-number updater with backup and manual add"
```

---

### Task 10: `main.py` pipeline

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `MODELS: dict[str, type] = {"ar": ARSetTransformer, "sedd": SetDiffusion, "hgnn": TemporalHGNN}`
  - `predict_next(model, df: pd.DataFrame, window: int, tickets: int, seed: int, device: str) -> list[list[int]]` returns 1-indexed sorted tickets for draw `T+1`, built with `build_sample(df, multihot, len(df), window)` batched to B=1.
  - `main(argv: list[str] | None = None) -> dict` takes these flags:
    - `--data lotto.xlsx`, `--model {ar,sedd,hgnn,all}` (default `all`), `--window 52`, `--epochs 50`, `--tickets 5`, `--seed 42` (training seed), `--out checkpoints`
    - `--add DRAW N1 N2 N3 N4 N5 N6 BONUS` (`nargs=8, type=int`, optional) for manual entry
    - `--no-update` to skip the network fetch
    - `--sample-seed` (default `None`, which becomes `secrets.randbits(32)`) for the ticket sampling seed
  - Order of operations:
    1. If `--add` is given, call `append_draws`.
    2. Unless `--no-update`, call `update_draws` and print `"added N draws (last draw L)"`.
    3. Run `load_draws`.
    4. For each model: `set_seed(seed)`, chronological 80/10/10 split, train, save `{out}/{name}.pt`, backtest the test split, then `predict_next` with `sample_seed`.
  - A fixed training seed keeps experiments reproducible. A fresh sample seed gives new tickets every run, and printing it lets any run be reproduced.
  - It returns `{"last_draw": int, "added": int, "update_warning": str | None, "sample_seed": int, "models": {name: {"history": {"train_loss": [...], "val_loss": [...]}, "backtest": {...}, "next_draw": [[...]]}}}`. It captures the `update_draws` warning with `warnings.catch_warnings(record=True)` into `update_warning` so the UI can show it. It prints the result as JSON and runs under `if __name__ == "__main__": main()`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement `main.py`**

- [ ] **Step 4: Run the full suite and two real runs**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest -v`
Expected: all tests pass (53 tests).

Run `C:/ProgramData/miniconda3/envs/Predict/python.exe main.py --model all --epochs 50` twice.
Expected:
- Each run prints `added N draws (last draw L)`, with N = 0 when nothing new has been drawn, and then continues.
- Each run prints JSON for the 3 models, with `mean_matches` near `baseline_mean_matches = 0.8` and a `p_value`.
- The two runs show different `sample_seed` values and different `next_draw` tickets.
- A large `p_value` is the expected honest result on near-i.i.d. data. A `p_value < 0.01` should be checked for leakage before anyone trusts it.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py && git commit -m "feat: end-to-end pipeline entry point"
```

---

### Task 11: Streamlit UI

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `main(argv) -> dict` (Task 10), `load_draws`, `draw_date` (Task 1), `append_draws` (Task 9).
- Produces: `app.py`, launched with `C:/ProgramData/miniconda3/envs/Predict/python.exe -m streamlit run app.py`. The data path comes from the env var `LOTTO_DATA`, default `lotto.xlsx`. The page has these sections, top to bottom:
  1. **Sidebar:** `selectbox` model `["all","ar","sedd","hgnn"]`, `number_input` epochs (default 50), tickets (default 5), window (default 52), and `checkbox` "자동 갱신" (default on; off adds `--no-update`).
  2. **Header:** the last draw number, its date (`draw_date`), and its 6 numbers + bonus.
  3. **Manual entry (`st.form`):** draw number (default `last+1`), 6 numbers and a bonus (`number_input`, 1–45). Submit calls `append_draws`. A `ValueError` shows as `st.error(str(e))` and the file stays unchanged. Success shows `st.success` and calls `st.rerun()`.
  4. **Run button "실행":** inside `st.spinner`, it calls `main([... --data LOTTO_DATA, sidebar values ...])` and stores the result in `st.session_state["result"]`. Training blocks the page during a run; the spinner covers that.
  5. **Results** (only when `result` exists):
     - "added N draws", or `st.warning(update_warning)`, plus `sample_seed`.
     - Tickets: one `st.dataframe` with rows = model × ticket and columns `n1..n6`, 1-indexed.
     - Backtest: one `st.dataframe` with rows = model and columns `mean_matches, baseline_mean_matches, rate_ge3, baseline_rate_ge3, p_value, unique_ratio`.
     - Learning curves: one `st.line_chart` per model of `train_loss` and `val_loss`.
     - A fixed `st.caption`: "로또 추첨은 독립 시행입니다. 기준선 대비 p_value가 작지 않으면 무작위 선택과 차이가 없습니다."

- [ ] **Step 1: Install streamlit**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pip install streamlit`
Expected: installs, and `python -c "import streamlit.testing.v1"` succeeds.

- [ ] **Step 2: Write the failing tests** using `streamlit.testing.v1.AppTest`

```python
def test_app_renders_last_draw(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("app.py").run(timeout=30)
    assert not at.exception and any("120" in m.value for m in at.markdown + at.header + at.subheader)

def test_app_manual_add_rejects_bad_draw(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("app.py").run(timeout=30)
    at.number_input(key="add_draw").set_value(125); at.button(key="add_submit").click().run(timeout=30)
    assert at.error and len(load_draws(p)) == 120

def test_app_run_shows_results(tmp_path, draws_df, monkeypatch):
    p = tmp_path / "l.xlsx"; save_draws(draws_df, p); monkeypatch.setenv("LOTTO_DATA", str(p))
    at = AppTest.from_file("app.py").run(timeout=30)
    at.selectbox(key="model").set_value("hgnn"); at.number_input(key="epochs").set_value(1)
    at.number_input(key="window").set_value(20); at.checkbox(key="auto_update").uncheck()
    at.button(key="run").click().run(timeout=180)
    assert not at.exception and len(at.dataframe) == 2
```

Widget keys are fixed by these tests: `model`, `epochs`, `tickets`, `window`, `auto_update`, `add_draw`, `add_n1..add_n6`, `add_bonus`, `add_submit` (the form submit button), and `run`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest tests/test_app.py -v`
Expected: FAIL (`app.py` does not exist)

- [ ] **Step 4: Implement `app.py`**

- [ ] **Step 5: Run the full suite, then open the UI**

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m pytest -v`
Expected: all tests pass (56 tests).

Run: `C:/ProgramData/miniconda3/envs/Predict/python.exe -m streamlit run app.py`
Expected: the browser shows draw 1247 (2026-10-24) in the header. Clicking "실행" shows tickets, the backtest table, and the learning curves.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py && git commit -m "feat: streamlit UI for update, run, manual entry, results"
```

---

## Deliberately skipped (from the spec; add later only if wanted)

- Beam search over top combinations (spec §1-1): sampling already gives tickets, and beam search would return the mode, which collapses ticket diversity.
- Holiday and seasonality flags (spec §2-④): no holiday calendar exists in the data. The day-of-year sin/cos covers coarse seasonality.
- Mamba encoder (spec §3 mentions it as an option).
- PyTorch Lightning, hyperparameter search, and model ensembling.
