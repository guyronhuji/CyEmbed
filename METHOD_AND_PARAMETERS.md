# CyEmbed: the method, and every parameter

A complete reference: what the model does, what each of the 41 config parameters controls, and
what to set it to for CyTOF versus scRNA-seq.

Companions: `CODEBASE_DESCRIPTION.md` (API surface), `SCRNA_SEQ_GUIDE.md` (scRNA-seq workflow and
traps), `../mcRBM/Desc/scrna_seq_guide.md` (the other model).

Values marked **[measured]** come from benchmarks in `tools/` and are cited. Everything else is
the code's default or reasoned from it — treated as a starting point, not a finding.

---

## 1. The method

Each cell is a **convex combination of K archetypes** — extreme molecular programs. The model is
an autoencoder with a simplex bottleneck.

**Encode.** An MLP maps a cell's feature vector `x_i ∈ R^M` to K logits `u_i`, then onto the
simplex:

```
u_i = EncoderMLP(x_i)                    # hidden_dims, ReLU, optional dropout
w_i = normalizer(u_i / tau)              # w_ik >= 0,  sum_k w_ik = 1
```

`w_i` is the object you care about: cell `i`'s **proportions** over the K archetypes. This is what
distinguishes archetypal analysis from NMF, whose loadings are unnormalised and confounded with
library size.

**Decode.** Two forms:

```
factorized:  h_i   = w_i @ Z                      # Z (K,d): archetype coords in latent space
             x̂_i   = h_i @ E.T + b                # E (M,d): a d-dim embedding PER FEATURE
             A_hat = Z @ E.T + b                  # archetype profiles in feature space

direct:      x̂_i   = w_i @ A                      # A (K,M) learned directly, NO bias term
             A_hat = A
```

**Per-patient offset** (optional), added in feature space after either decoder:

```
x̂_i += B_eff[s_i]        where  B_eff = B - B.mean(0)     # B (S,M), zero-sum across patients
```

**The geometry, which is the whole point.** Because `w` is on the simplex, every cell lies inside
the **convex hull of K points** — a (K−1)-dimensional object. Archetypes are the *vertices*;
cells interpolate between them. That is a claim about biology (cells lie on a continuum between
extreme states), not a factorisation convenience. It is why `d` is not a capacity knob (§3.2) and
why NMF cannot make the same statement.

**Loss.**

```
L = recon(x̂, x)
  + lambda_entropy  * mean_i H(w_i)                 # per-cell weight entropy
  + lambda_sep      * separation(Z or A)            # push archetypes apart
  + lambda_balance  * balance(mean_i w_i)           # guard against dead archetypes
  [+ beta_w * KL_w + beta_r * KL_r]                 # probabilistic only
```

**Probabilistic variant.** The encoder emits `mu_w, logvar_w`; weights are sampled via
reparameterisation, `u ~ N(mu_w, exp(logvar_w))`, and a KL to `N(0,I)` shrinks toward a
uniform-centred prior. Gives a per-cell posterior over composition.

---

## 2. Preprocessing (not in `run_config` — the data layer)

| Parameter | Where | CyTOF | scRNA-seq | Notes |
|---|---|---|---|---|
| `MarkerScaler.mode` | `fit_scaler(mode=…)` | `"zscore"` | **`"none"`** | SCT Pearson residuals are already ~N(0,1) per gene (measured `mean −0.0000, std 0.9858`). Z-scoring on top undoes the stabilisation. |
| — `"robust_zscore"` | | usable | **NEVER** | Broken on sparse data: any gene detected in <50% of cells has median 0 → MAD 0 → `scale` clamps to `eps=1e-8` → detections ×1e8 (`data.py:63-70`). Raises `FloatingPointError`. |
| `balanced_max_per_sample` | `fit_scaler` | ~2000 | n/a with `mode="none"` | Stops the scaler being dominated by your largest sample. Needs `sample_ids` too. |
| `val_fraction` | `split_train_val_indices` | 0.2 | 0.15–0.2 | |
| `stratify_labels` | `split_train_val_indices` | `sample_ids` | `sample_ids` | Splits *within* each group, so every patient is in both halves. **Required** if `use_sample_offset=True`, else `B` is untrained for held-out patients. |

Upstream of all of this: for scRNA-seq, feed SCTransform/analytic Pearson residuals and select
HVGs **before** `extract_matrix`. The package does no feature selection.

---

## 3. Model architecture

### 3.1 `K` — number of archetypes

The only real modelling choice. It is a hypothesis about how many extreme programs your biology
has.

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | sweep 4–10 | sweep 3–10 |

**How to select it** — benchmarked against a planted `K_TRUE=5` (`tools/select_k_and_d_scrna.py`)
**[measured]**:

| K | val_recon | dead | stability | w_recovery (oracle) |
|---|---|---|---|---|
| 3 | 1.2953 | 0.0 | 1.000 | 0.490 |
| 4 | 1.3291 | 0.0 | 0.564 | 0.426 |
| **5** | **1.1920** | 0.3 | 0.740 | **0.732** |
| 6 | 1.2949 | 1.3 | 0.702 | 0.410 |
| 7 | 1.5070 | 1.0 | **1.000** | 0.029 |
| 8 | 1.2261 | 2.7 | 0.679 | 0.430 |

- **Archetype redundancy — the strongest criterion.** `mean|off-diag|` of
  `cosine_similarity_matrix(A_hat)`: 0.209, 0.324, **0.078**, 1.000, 0.993, 0.985 for K=3..8.
  Sharp minimum at truth, and **1.000 means archetypes have become identical**. Above K_TRUE this
  model collapses rather than splitting (`lambda_balance` forcing uniform usage across more
  archetypes than the data supports) — which is why stability scores a perfect 1.000 at K=7: every
  seed reaches the *same* degenerate solution. Use `mean`, not `max` (max hits 1.000 at K=4 on one
  duplicate pair).
- **`val_recon` minimum** picks the truth. It is *not* monotone in K — early stopping plus the
  `lambda_*` package makes excess archetypes cost on held-out data. Verify non-monotonicity on
  your own data; if it falls to your largest K, it isn't selecting.
- **Dead archetypes** picks the truth. Use a *relative* threshold (`usage < 0.5/K`);
  `dead_archetypes_lt_1pct` is absolute and gets less strict as K grows.
- **Cross-seed stability fails** — picks K=7 with a perfect 1.000 while recovering nothing
  (`w_recovery 0.029`). Degenerate solutions are trivially reproducible. **Reject with it, never
  select with it.**

### 3.2 `d` — latent dimension (factorized decoder only)

**Default 8 in the notebooks. Use 16–32.** **[measured]**

| d | 2 | 3 | 5 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| w_recovery @ K=5, 2000 HVGs | 0.520 | 0.852 | 0.722 | 0.712 | **0.988** | 0.992 | 0.992 |

`rank(A_hat) ≤ min(K, d)`, so above `d = K` the rank constraint is **inactive** — d=16 and d=5
have identical expressiveness at K=5. The gap is pure optimisation: a wider product
parametrisation descends better. Plateau at **d≈16–32, not d=K**.

**What `d` does not control:** the dimension your cells live in. `h = w @ Z` with `w` on the
simplex means cells occupy a **(K−1)-dimensional** hull regardless of `d`. Unlike a VAE's latent,
`d` is not capacity.

**With `decoder_type="direct"`, `d` does nothing at all** (`latent_dim` is read only in the
factorized branches, `model.py:128-129`, `:263-268`) — but it *is* fingerprinted, so sweeping it
produces one run dir per value holding identical models.

| | CyTOF (~40 markers) | scRNA-seq (2000 HVGs) |
|---|---|---|
| Recommended | 8–16 | **16–32** |

### 3.3 `decoder_type` — `"factorized"` | `"direct"`

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | either | **`"factorized"`** |

**[measured]** Identical at 20 markers (w_recovery 0.993 each); at 2000 HVGs, factorized **0.818**
vs direct **0.575**. That gap is the reason to use it — **the mechanism is not established.** It is
not expressiveness (both `A_hat`s are rank ≤ K), and it is not simply "direct has no bias": because
`w` sums to 1, adding a constant vector to every row of `A` shifts every cell alike, so `A` *can*
carry a baseline. The likely cause is optimisation, consistent with §3.2 — a wider product
parametrisation trains better at identical rank, and direct is single-factor.

**`E` is *not* a reason to prefer factorized** — but it works fine. `E_g ≈ E_h` implies the genes
load alike, so closeness in `E` is real information. In theory `E` is identifiable only up to an
invertible `R`, and its `null(Z)` component is invisible to `x̂`; in practice neither bites.
**[measured]** on planted gene groups at 2000 HVGs: AUC **1.000 from `E`** and **1.000 from
`A_hat.T`**, with cross-seed agreement **0.764 vs 0.775** — equivalent. Weight decay pins the
factorisation near-balanced, so cosines are effectively invariant. Use either; `A_hat.T` is
marginally safer (exactly the loadings, no ambiguity, exists for `direct` too). The point is that
`E` adds nothing *over* `A_hat`, not that it is unreliable.

**[measured]** The real caveat is neither matrix: **gene modules are only ~0.77 correlated across
seeds** — the archetypes move between restarts. Take a consensus before believing a module.

### 3.4 `model_type` — `"deterministic"` | `"probabilistic"`

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | either | **`"deterministic"`** unless you use `logvar_w` |

**[measured]** w_recovery at 2000 HVGs, factorized:

| | no regularisers | with the `lambda_*` package |
|---|---|---|
| deterministic | 0.834 | **0.992** |
| probabilistic | 0.978 | 0.856 |

The KL and the explicit regularisers are **substitutes**. Without `lambda_*` the KL does the
regularising and helps; with them it over-regularises and costs you. Probabilistic's real value is
the per-cell posterior — which `prob_eval_mode="mean"` discards.

### 3.5 `hidden_dims` — encoder MLP widths

Default `(128, 64)`. Real configs use `[256, 64]` (matches `sct_gaussian_k_sweep.yaml`) or
`[256, 128]`.

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | `[128, 64]` | `[256, 64]` |

At 330 cells (BCK_44) a `[256,64]` encoder over 2000 genes is ~272k parameters on 330
observations. The simplex bottleneck is the only thing making that tractable.

### 3.6 `dropout` — encoder dropout, default `0.0`

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | 0.0 | 0.0–0.1 |

Applied to encoder hidden layers only.

---

## 4. Simplex mapping

### 4.1 `tau` — temperature, default `1.0`, must be > 0

Logits are divided by `tau` before the normaliser. **Lower → sharper/more one-hot weights; higher
→ more diffuse.** Real sweeps use `[0.7, 1.0]`.

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | sweep `[0.7, 1.0]` | sweep `[0.7, 1.0]` |

### 4.2 `logit_normalizer` — `"softmax"` | `"entmax"`, default `"softmax"`

### 4.3 `entmax_alpha` — default `1.5`, must be in `[1.0, 2.0]`

`1.0` → softmax; `1.5` → entmax15; `2.0` → sparsemax; otherwise `entmax_bisect`. Entmax yields
**sparser** weights than softmax (exact zeros are reachable).

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | `"entmax"`, 1.5 | `"entmax"`, 1.5 **[measured]** as part of the package that took w_recovery 0.834 → 0.992 |

Sets a derived key `simplex_impl_version` (3 for entmax, 1 for softmax) that enters the
fingerprint. **Caution:** entmax on MPS can produce NaNs; `train.py` guards and raises
`FloatingPointError`.

---

## 5. Loss

### 5.1 `recon_loss_type` — `"mse"` | `"huber"`, default `"mse"`

### 5.2 `huber_delta` — default `1.0`

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | `"mse"` | `"mse"`; `"huber"` is defensible |

Both are Gaussian likelihoods with constant variance. On Pearson residuals (heavy-tailed) Huber
downweights the tails and would help — but mcRBM's loss is hardcoded MSE, so use `"mse"` if you
intend to compare the two models.

### 5.3 `lambda_entropy` — default **`0.0`**

Penalises mean per-cell weight entropy. Positive → *lower* entropy → sharper weights.

### 5.4 `lambda_sep` — default **`0.0`**

Pushes archetypes apart (on `Z` for factorized, `A` for direct).

### 5.5 `separation_mode` — default `"cosine_sq"`

`"cosine_mean"` | `"cosine_abs"` | `"cosine_sq"` | `"rbf"`.

### 5.6 `rbf_gamma` — default `1.0`. Only used when `separation_mode="rbf"`.

### 5.7 `lambda_balance` — default **`0.0`**

Guards against dead archetypes by penalising mean usage away from uniform. **The most important
of the three** — it is what keeps archetypes alive at moderate K.

### 5.8 `balance_mode` — default `"l2_uniform"`

`"l2_uniform"` | `"kl_uniform"` | `"neg_entropy"`.

| | CyTOF | scRNA-seq |
|---|---|---|
| `lambda_entropy` | 1e-3 | **1e-3** |
| `lambda_sep` | 1e-3 | **1e-3** |
| `lambda_balance` | 5e-2 | **5e-2** |
| `separation_mode` | `"cosine_sq"` | `"cosine_sq"` |
| `balance_mode` | `"l2_uniform"` | `"l2_uniform"` |

**[measured]** This package took w_recovery 0.834 → 0.992 and val_recon 1.180 → 1.086 at 2000
HVGs. **All three default to 0.0** — if you don't set them, you have no regularisation at all,
and `val_recon` may stop working as a K-selector.

---

## 6. Per-patient offset

### 6.1 `use_sample_offset` — default `False`

Adds `B (S,M)`, a learned per-patient intercept in the decoder, centred zero-sum each forward
pass, warm-started at the empirical per-patient mean, and excluded from weight decay.

| | CyTOF | scRNA-seq |
|---|---|---|
| Recommended | `False` unless multi-sample | `False` **first**, then only if you see identity archetypes |

**[measured]** At 2000 HVGs: cross-patient spread 0.43 → 0.009, `B~shift` corr 0.999, w_recovery
0.037 → 0.818.

Requires `sample_ids`, and a **stratified** split so every patient appears in both halves. Removes
*additive* shifts only. It is a **soft** fix — it removes the incentive for `w` to encode patient,
not the ability. `A_hat` excludes `B` (it is the profile at the average patient).

**Do not reach for it reflexively.** Shared archetypes with patient-varying composition is the
*goal*. The failure worth correcting is narrow: an archetype with `dominant_fraction ≈ 1.0` in one
patient and ≈0 elsewhere. In PDX/patient tumours inter-patient variation substantially *is* the
biology. Diagnose with `summarize_by_group(W, sample_ids)` first.

Only enters the config fingerprint when `True`, so leaving it off preserves cached run dirs.

---

## 7. Probabilistic-only

All ignored when `model_type="deterministic"`.

| Parameter | Default | CyTOF | scRNA-seq | What it does |
|---|---|---|---|---|
| `use_residual_latent` | `False` | `False`/`True` | **`False`** | Per-cell latent absorbing what archetypes can't explain. On sparse data that's **dropout noise** — it defeats the simplex bottleneck's main protection. |
| `residual_dim` | `= d` | 8 | 8 | Only if the above is on. |
| `beta_w` | `1e-3` | 1e-3 | 1e-3 | KL weight on the archetype-logit posterior. Too high → posterior collapse to uniform. |
| `beta_r` | `1e-3` | 1e-3 | n/a | KL weight on the residual latent. |
| `kl_warmup_epochs` | `0` | 10 | 10 | Linearly ramps `beta_*` from 0. Prevents early collapse. |
| `logvar_min` | `-10.0` | −10 | −10 | Clamp on posterior log-variance. |
| `logvar_max` | `10.0` | 5.0 | 5.0 | Real configs use 5.0 — tighter than default. |
| `logvar_init_bias` | `-3.0` | −3 | −3 | Starts the posterior near-deterministic (small variance). |
| `prob_eval_mode` | `"mean"` | `"mean"` | `"mean"` | `"mean"` = posterior mean (stable, used for early stopping); `"sample"`; `"mc"` = average over `prob_eval_samples`. |
| `prob_eval_samples` | `1` | 3 | 3 | Only used by `"mc"`. |

If you use `prob_eval_mode="mean"` you are discarding the uncertainty that is the only reason to
run probabilistic at all (§3.4).

---

## 8. Optimisation

| Parameter | Default | CyTOF | scRNA-seq | Notes |
|---|---|---|---|---|
| `lr` | required | 1e-3 | 1e-3 | Adam. |
| `weight_decay` | **`0.0`** | 1e-5 | 1e-4 | Adam (not AdamW), so L2-into-gradient, diluted by Adam's normalisation. `B` is excluded via its own param group. |
| `epochs` | required | 1500 | 400–3000 | With early stopping this is a cap. |
| `batch_size` | required | 1024–2048 | 512–1024 | |
| `grad_clip_norm` | `None` | 5.0 | 5.0 | Global norm over all params. Useful with entmax. |

### Early stopping

| Parameter | Default | Recommended | Notes |
|---|---|---|---|
| `early_stopping` | `True` | `True` | Tracks best `val_recon`. |
| `patience` | `20` | 20–60 | Epochs without improvement before stopping. |
| `min_delta` | `0.0` | 1e-4 | Improvement threshold. |
| `restore_best_weights` | `True` | `True` | **The saved `W`/`B` come from `best_epoch`, not the last one.** |

**Non-convergence is the most common way to get a wrong answer here.** At a fixed 150–300 epochs
these models are still descending, which produced a spurious 4.5× "regression" that vanished at
convergence (`val_recon` 2.039 → 0.004). Always use early stopping with a generous cap.

---

## 9. Reproducibility and runtime

| Parameter | Default | CyTOF | scRNA-seq | Notes |
|---|---|---|---|---|
| `seed` | `0` | **sweep `[7,17,23]`** | **sweep `[7,17,23]`** | **[measured]** Two runs of an identical config gave `val_recon` 1.2586 vs 1.3291 — **5% variance**, against a ~6% margin for the winning K. On one seed you cannot distinguish K from noise. Put `"seed": [7,17,23]` in `SWEEP_GRID`; it works today, since `_config_fingerprint` includes `seed`. |
| `deterministic` | `True` | `True` | `True` | Torch determinism / cuDNN flag. Unrelated to `model_type`. |
| `device` | `"auto"` | **`"cpu"`** for small models | `"cpu"` or `"cuda"` | **[measured]** `"auto"` resolves CUDA → MPS → CPU. On Apple Silicon with small tensors, **CPU is 11.6× faster than MPS** (2.0s vs 23.3s for an identical 300-epoch run at 20 features) — kernel-launch overhead dominates. Benchmark before trusting `"auto"`. |

---

## 10. Logging and sweeps

None of these affect the model. The first four are excluded from the fingerprint, so changing them
does not invalidate cached runs.

| Parameter | Default | Notes |
|---|---|---|
| `print_every` | `10` | Prints at epoch 1, multiples, and the last epoch. |
| `progress_epoch` | `True` | tqdm per epoch. |
| `progress_sweep` | `True` | Read off `base_config`, not a `run_sweep` argument. |
| `skip_existing_runs` | `True` | Read off `base_config`. Reloads a matching-fingerprint run dir instead of retraining. |
| `run_name` | — | Forces a folder name. |

---

## 11. Starting configs

**CyTOF** (~40 markers, arcsinh-transformed):

```python
GLOBAL_CFG = {"seed": 7, "deterministic": True, "device": "auto"}
scaler, _ = fit_scaler(bundle.X, mode="zscore",
                       sample_ids=bundle.sample_ids, balanced_max_per_sample=2000)
BASE = {"epochs": 1500, "early_stopping": True, "patience": 20, "min_delta": 0.0,
        "restore_best_weights": True, "weight_decay": 1e-5, "dropout": 0.0,
        "logit_normalizer": "entmax", "entmax_alpha": 1.5, "grad_clip_norm": 5.0,
        "separation_mode": "cosine_sq", "balance_mode": "l2_uniform"}
GRID = {"model_type": ["deterministic"], "decoder_type": ["factorized"],
        "K": [4,5,6,7,8], "d": [8, 16], "hidden_dims": [[128,64]],
        "lr": [1e-3], "batch_size": [2048], "recon_loss_type": ["mse"],
        "lambda_entropy": [1e-3], "lambda_sep": [1e-3], "lambda_balance": [5e-2],
        "tau": [0.7, 1.0], "seed": [7, 17, 23]}
```

**scRNA-seq** (2000 HVGs, SCT Pearson residuals):

```python
GLOBAL_CFG = {"seed": 7, "deterministic": True, "device": "cpu"}
scaler, _ = fit_scaler(bundle.X, mode="none")        # residuals are already ~N(0,1)
BASE = {"epochs": 3000, "early_stopping": True, "patience": 30, "min_delta": 1e-4,
        "restore_best_weights": True, "weight_decay": 1e-4, "dropout": 0.0,
        "logit_normalizer": "entmax", "entmax_alpha": 1.5, "grad_clip_norm": 5.0,
        "separation_mode": "cosine_sq", "balance_mode": "l2_uniform"}
GRID = {"model_type": ["deterministic"], "decoder_type": ["factorized"],
        "K": [3,4,5,6,7,8], "d": [16, 32], "hidden_dims": [[256,64]],
        "lr": [1e-3], "batch_size": [512], "recon_loss_type": ["mse"],
        "lambda_entropy": [1e-3], "lambda_sep": [1e-3], "lambda_balance": [5e-2],
        "tau": [0.7, 1.0], "seed": [7, 17, 23],
        "use_sample_offset": [False]}      # turn on only after seeing identity archetypes
```

---

## 12. Diagnostics to run every time

| Check | How | Failure looks like |
|---|---|---|
| **Depth leakage** (scRNA) | Spearman each `W[:,k]` against total counts | Any archetype tracking depth ⇒ preprocessing failed; everything downstream is suspect |
| **Patient leakage** | `summarize_by_group(W, sample_ids)` | `dominant_fraction ≈ 1.0` in one patient, ≈0 elsewhere ⇒ identity archetype |
| **Dead archetypes** | `usage < 0.5/K` (relative!) | K too high |
| **Stability** | Hungarian-matched cosine of `A_hat` across seeds | Low ⇒ reject that K. High is *not* evidence for it |
| **Convergence** | `history.csv` `val_recon` still descending at the last epoch | Truncated, not converged — comparisons are meaningless |
| **Biology** | UCell signatures joined on `cell_id`, Spearman vs `W` | The only non-self-referential check |

---

## 13. The traps, in one place

1. **`robust_zscore` on sparse data** → ×1e8 blowup, `FloatingPointError` (`data.py:63-70`).
2. **`mode="zscore"` on Pearson residuals** → undoes the variance stabilisation.
3. **`sample_col` typos fail silently** → `sample_ids=None`, stratification and balanced scaling
   quietly disabled. Assert it resolved.
4. **`marker_names` is ignored** unless `source="obsm"`.
5. **All three `lambda_*` default to 0.0** — omit them and you have no regularisation.
6. **`weight_decay` defaults to 0.0.**
7. **`d` does nothing with `decoder_type="direct"`** but is still fingerprinted.
8. **`d ≈ K` is the wrong regime** — use 16–32.
9. **One seed cannot select K** — the noise is the same size as the signal.
10. **`device="auto"` picks MPS**, which can be 11.6× slower than CPU on small models.
11. **Fixed epochs ≠ converged.** Use early stopping; check the curve.
12. **Stability rewards reproducible collapse** — reject with it, don't select with it.
13. **`dead_archetypes_lt_1pct` is an absolute threshold**, so it under-detects at large K.
14. **`use_sample_offset` needs a stratified split**, or `B` is untrained for held-out patients.
