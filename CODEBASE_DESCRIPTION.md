# CyEmbed — Codebase Description

A compact, notebook-first PyTorch library for **archetype-based deconvolution of single-cell CyTOF profiles**. Each cell is modeled as a convex combination of K latent archetypes (extreme molecular programs). The library provides a deterministic variant and a probabilistic (variational) variant with an optional residual latent, plus a small framework for hyperparameter sweeps, metrics, and notebook analysis.

---

## 1. Repository layout

```
CyEmbed/                                  # project root (repo)
├── CyEmbed/                              # the Python package (see §2)
│   ├── __init__.py                       # public re-exports
│   ├── data.py                           # DataBundle, MarkerScaler, splits
│   ├── model.py                          # ArchetypeEmbeddingModel, ProbabilisticArchetypeModel
│   ├── losses.py                         # reconstruction + regularizers + KL
│   ├── train.py                          # train_one_run, run_sweep, RunResult
│   ├── analysis.py                       # load_run_outputs + metrics/utilities
│   ├── plotting.py                       # matplotlib/seaborn helpers
│   └── utils.py                          # seeding, device, I/O, config validation
├── 01_*_sweep.ipynb, 02_*_analysis.ipynb # paired notebooks per experiment
├── notebooks/<experiment>/…              # additional per-experiment notebooks
├── outputs/<experiment>/run_<type>_<fp>/ # per-run artifacts produced by train_one_run
├── Analysis/<experiment>/ …              # downstream analysis outputs
├── tools/                                # standalone report-generation scripts
│   ├── generate_cyembed_methods_pdf.py
│   ├── generate_batch6_report.py
│   ├── generate_pdac_chemo_archetype_report.py
│   └── generate_pdac_chemo_nature_style_report.py
└── CODEBASE_DESCRIPTION.md               # this file
```

Dependencies (inferred from code): `numpy`, `pandas`, `torch`, `entmax`, `matplotlib`, `seaborn`, optional `tqdm`, optional `umap-learn`, optional `anndata`/`scanpy`. Tools scripts use `reportlab`, `PIL`.

---

## 2. Package API (`CyEmbed/__init__.py`)

Public exports (all symbols re-exported from `CyEmbed`):

- **Models**: `ArchetypeEmbeddingModel`, `ProbabilisticArchetypeModel`
- **Data**: `DataBundle`, `MarkerScaler`, `extract_matrix`, `fit_scaler`, `preprocess_array`, `split_train_val_indices`
- **Training**: `build_sweep_configs`, `run_sweep`, `train_one_run`
- **Utils**: `collect_software_versions`, `make_run_id`, `resolve_device`, `set_seed`, `validate_run_config`
- **Analysis**: `archetype_marker_rankings`, `cosine_similarity_matrix`, `dominant_assignments`, `kl_history_columns`, `load_run_outputs`, `nearest_neighbors_from_similarity`, `per_marker_reconstruction_stats`, `posterior_mean_weights`, `residual_norms`, `residual_summary`, `summarize_by_group`, `weight_entropy`

---

## 3. Mathematical model

For each cell `x_i ∈ R^M` (M = markers), the model infers a simplex weight vector `w_i ∈ Δ^{K-1}` (K archetypes, `w_ik ≥ 0`, `Σ_k w_ik = 1`) and reconstructs the input. Two decoder types:

- **Factorized decoder**: archetypes live in a latent space of dim `d`.
  - `Z ∈ R^{K×d}`: archetype coordinates in latent space
  - `E ∈ R^{M×d}`: marker embeddings
  - `b ∈ R^M`: per-marker bias
  - `h_i = w_i Z`  (cell latent)
  - `x̂_i = h_i E^T + b`
  - `A_hat = Z E^T + b` (archetype profiles in marker space)
- **Direct decoder**: `A ∈ R^{K×M}` is learned directly; `x̂_i = w_i A`, `A_hat = A`.

### 3.1 Simplex mapping (`model.simplex_weights_from_logits`)

Logits `u ∈ R^K` → weights via `u/τ` then a normalizer:
- `logit_normalizer="softmax"`: standard softmax
- `logit_normalizer="entmax"` with `entmax_alpha ∈ [1.0, 2.0]`:
  - `α = 1.0` → softmax, `α = 1.5` → `entmax15`, `α = 2.0` → `sparsemax`, else `entmax_bisect`. Entmax variants yield sparser weights than softmax.

### 3.2 Deterministic model (`ArchetypeEmbeddingModel`)

- **Encoder**: `EncoderMLP`, MLP with ReLU+optional dropout on hidden dims, output dim K (archetype logits).
- Forward returns `{U, W, H, X_hat, A_hat}`.
- `archetype_separation_tensor()` returns `Z` (factorized) or `A` (direct) — used for separation penalty.

### 3.3 Probabilistic model (`ProbabilisticArchetypeModel`)

Variational (logistic-normal) version. An encoder trunk produces two heads:
- `mu_w, logvar_w ∈ R^K` — Gaussian posterior over archetype logits
- If `use_residual_latent=True`: `mu_r, logvar_r ∈ R^{residual_dim}` — per-cell residual latent

`logvar` is clamped to `[logvar_min, logvar_max]`, initialized with bias `logvar_init_bias` (default −3 → small initial variance).

Sampling uses the reparameterization trick:
- `u_sample ~ N(mu_w, diag(exp(logvar_w)))`
- `W = normalizer(u_sample/τ)`
- `W_mean = normalizer(mu_w/τ)` (posterior-mean plug-in)
- Optional residual `r_sample ~ N(mu_r, diag(exp(logvar_r)))`

Decoding with residual:
- Factorized: `h_main = w Z`, `h_total = h_main + r_proj`, where `r_proj = r` if `residual_dim == latent_dim` else `r @ P_r` (learned projection `P_r`). `x̂ = h_total E^T + b`.
- Direct: `x̂ = w A + r G`, with learned `G ∈ R^{residual_dim × M}`.

Forward also returns diagnostic scalars: `kl_w`, `kl_r`, `entropy` (mean per-cell categorical entropy), `sep` (cosine² off-diagonal mean of archetype rows), `balance` (L2 distance of mean usage from uniform).

KL to N(0,I): `gaussian_kl_standard_normal(mu, logvar)`.

---

## 4. Losses (`CyEmbed/losses.py`)

- `reconstruction_loss(x_hat, x, loss_type)` — `"mse"` or `"huber"`.
- `entropy_penalty(w)` — mean per-cell categorical entropy of weights.
- `separation_penalty(A, mode)` — push archetypes apart:
  - `"cosine_mean" | "cosine_abs" | "cosine_sq"`: off-diagonal cosine similarity (or its |·|, ²).
  - `"rbf"`: mean off-diagonal `exp(-γ ||A_i − A_j||²)`.
- `balance_penalty(w, mode)` — guard against dead archetypes:
  - `"l2_uniform"`: MSE of mean usage vs 1/K.
  - `"kl_uniform"`: KL(mean usage || uniform).
  - `"neg_entropy"`: `log K − H(mean usage)`.
- `total_loss(...)`: for deterministic model
  `recon + λ_entropy·entropy + λ_sep·sep + λ_balance·balance`.
- `total_variational_loss(...)`: for probabilistic model, adds `β_w·kl_w + β_r·kl_r` (β's may be warmed up externally).

Returns `(total, parts_dict)` where `parts_dict` keys are logged every epoch.

---

## 5. Data handling (`CyEmbed/data.py`)

- `DataBundle`: dataclass wrapping `X (N×M, float32)`, `marker_names`, `cell_ids`, optional `sample_ids`, `cluster_ids`. Shapes are validated.
- `MarkerScaler`: per-marker scaler, `mode ∈ {"none", "zscore", "robust_zscore"}`. Robust uses median and `1.4826 * MAD`. Supports `fit`, `transform`, `inverse_transform`, `to_dict`/`from_dict` (JSON-serializable).
- `extract_matrix(...)`: builds a `DataBundle` from any of: AnnData (`source ∈ {"X","layer","obsm"}`, columns from `var_names`, cell ids from `obs_names`, optional `sample_col`/`cluster_col` from `obs`), a pandas DataFrame, or a raw ndarray.
- `balanced_downsample_indices(labels, max_per_group, random_state)`: balanced random sampling across groups.
- `fit_scaler(x, mode, sample_ids=None, balanced_max_per_sample=None, random_state=0)`: fits a `MarkerScaler`, optionally on a balanced subset across samples; returns `(scaler, fit_idx)`.
- `preprocess_array(x, scaler)`: applies scaler (or passes through).
- `split_train_val_indices(n_cells, val_fraction, seed, stratify_labels=None)`: shuffled or stratified train/val split; returns sorted index arrays.

---

## 6. Utilities (`CyEmbed/utils.py`)

- `set_seed(seed, deterministic=True)`: seeds Python/NumPy/Torch (+ CUDA), and enables deterministic algorithms with `warn_only=True`.
- `resolve_device(device)`: priority CUDA → MPS → CPU, or honor explicit `"cuda"`/`"mps"`/`"cpu"`/`"auto"`.
- `make_run_id(prefix)`: timestamped id `"{prefix}_YYYYMMDD_HHMMSS"`.
- `ensure_dir(path)`.
- `NumpyJSONEncoder`, `save_json`, `load_json` (supports numpy scalars/arrays).
- `flatten_dict(values, prefix="")`: `{a: {b: 1}} → {"a.b": 1}`.
- `collect_software_versions()`: Python + numpy/pandas/torch/matplotlib/anndata/scanpy versions.
- `validate_run_config(config)`: canonicalizes/validates fields used by training. Enforces:
  - `model_type ∈ {"deterministic", "probabilistic"}`
  - `decoder_type ∈ {"factorized", "direct"}`
  - `tau > 0`
  - `logit_normalizer ∈ {"softmax", "entmax"}`, `entmax_alpha ∈ [1.0, 2.0]`
  - Sets `simplex_impl_version` (3 if entmax else 1)
  - For probabilistic: defaults + validates `use_residual_latent`, `residual_dim`, `beta_w`, `beta_r`, `kl_warmup_epochs`, `prob_eval_mode ∈ {"mean","sample","mc"}`, `prob_eval_samples ≥ 1`.

---

## 7. Training (`CyEmbed/train.py`)

### 7.1 `train_one_run(*, x_train, x_val, x_full, marker_names, cell_ids, run_config, output_root, ...) -> RunResult`

Trains a single config end-to-end and writes a complete artifact directory. Important behaviors:

- `run_config` is validated via `validate_run_config`; seed + device resolved.
- Builds the appropriate model via `_build_model` using keys:
  `model_type, decoder_type, K, d, hidden_dims, tau, logit_normalizer, entmax_alpha, dropout`, and (probabilistic) `use_residual_latent, residual_dim, logvar_min, logvar_max, logvar_init_bias`.
- Adam optimizer (`lr`, `weight_decay`).
- Per-epoch training over `DataLoader(TensorDataset)` with shuffle on train.
- **Non-finite guards**: after each step, any non-finite tensor/gradient/parameter raises `FloatingPointError` with diagnostic context (particularly useful with entmax on MPS).
- Optional `grad_clip_norm`.
- **KL warmup** (probabilistic): `beta_w/beta_r` linearly ramp from 0 to their base over `kl_warmup_epochs`.
- **Validation**: after each epoch, compute reconstruction loss on `x_val` using `_predict_batches`. Probabilistic validation uses posterior mean for stability (`prob_eval_mode="mean"` by default).
- **Early stopping**: tracks best `val_recon`, stops after `patience` epochs without `min_delta` improvement; if `restore_best_weights`, reloads the best snapshot.
- **Final evaluation**: runs `_predict_batches` on train/val/full data; computes `_metric_summary`:
  `recon_mse, recon_mae, mean/median_marker_corr, mean_weight_entropy, usage_{min,max,std,entropy}, dominant_frac_gt_{0.5,0.8}, dead_archetypes_lt_1pct`.
- For probabilistic runs, also generates a single-sample pass (`prob_eval_mode="sample"`) to save `W_sample, U_sample, R_sample, X_hat_sample`.
- Saves parameters `Z, E, b, A, P_r, G` when present.

### 7.2 `_predict_batches(...)` — inference modes (probabilistic)

- `"mean"`: posterior mean, no sampling. Returns `X_hat, W, W_mean, U, mu_w, logvar_w[, mu_r, logvar_r]`.
- `"sample"`: a single reparameterized sample. Also returns `R` (residual sample) when enabled.
- `"mc"`: Monte Carlo average over `prob_eval_samples` reparameterized samples of `X_hat, W, U` (and `R`).
- `A_hat` is cached once per run (identical for all batches).

### 7.3 Run identity, caching, outputs

- `_config_identity_payload` drops display-only keys (`run_name, progress_*, print_every, skip_existing_runs, resolved_device, software_versions`).
- `_config_fingerprint`: SHA-1 (first 10 hex) of the JSON-sorted identity payload.
- `_stable_run_id`: `"run_<model_type>_<fingerprint>"`.
- `run_sweep(..., skip_existing_runs=True)` searches `output_root` for a directory with a matching fingerprint (prefers `_stable_run_id`), and if found, **reloads** its `summary_metrics.json` via `_load_saved_flat_summary` instead of retraining. This makes sweeps idempotent.
- Each run directory contains:
  - `config.json`, `summary_metrics.json`, `history.csv` (per-epoch metrics incl. loss parts, `val_recon`, `beta_w_eff/beta_r_eff` for probabilistic).
  - `model_state.pt` (PyTorch `state_dict`).
  - Array dumps (`.npy`): `X_hat, W, U, X_observed, A_hat, residuals`, and when present `W_mean, mu_w, logvar_w, R, R_norm, mu_r, logvar_r, W_sample, U_sample, R_sample, X_hat_sample, Z, E, b, A, P_r, G`.
  - Metadata: `marker_names.csv, cell_ids.csv, sample_ids.csv, cluster_ids.csv, train_idx.csv, val_idx.csv`.
  - `scaler.json` (scaler state), `per_marker_corr.csv`.

### 7.4 `RunResult` dataclass

Returned from `train_one_run`: `run_id, run_dir, summary (flat dict for sweep table), history (DataFrame)`. The flat summary is what `run_sweep` concatenates.

### 7.5 `build_sweep_configs(param_grid)` and `run_sweep(...)`

- `build_sweep_configs`: cartesian product of a `{param: [values]}` grid into a list of dicts.
- `run_sweep`: iterates configs, merges each over `base_config`, validates, either reloads an existing directory or calls `train_one_run`. Writes incremental `sweep_summary.partial.csv` and final `sweep_summary.csv` (sorted by `val_recon` ascending), plus `sweep_metadata.json` with splits and the whole grid.

Optional `tqdm` progress bars are used if available; `base_config` supports `progress_sweep`, `progress_epoch`, `print_every`, `skip_existing_runs`.

---

## 8. Analysis (`CyEmbed/analysis.py`)

All functions operate on numpy arrays / pandas DataFrames and are designed to be called from notebooks after `load_run_outputs`.

- `load_run_outputs(run_dir)`: loads `config.json, summary_metrics.json, history.csv`, all `*.npy` arrays (by stem), `marker_names, cell_ids, sample_ids, cluster_ids` (if present), and parameters `Z, E, b, A`. Provides backward-compatible aliases: `X = X_observed`, `residuals = X_observed − X_hat`.
- `per_marker_reconstruction_stats(x, x_hat, marker_names)` → DataFrame with `pearson_r, spearman_r, r2, mse, mae` sorted by R².
- `archetype_marker_rankings(A_hat, marker_names, top_n)` → long DataFrame with top-n positive and negative markers per archetype.
- `weight_entropy(w)`, `dominant_assignments(w, cell_ids)` (dominant index/weight/entropy per cell), `purity_summary(w, thresholds)`.
- `summarize_by_group(w, group_ids)` → `{"mean_weights": ..., "dominant_fractions": ...}` for e.g. per-sample or per-cluster aggregation.
- `cosine_similarity_matrix(x)`, `pairwise_distance_matrix(x)`, `nearest_neighbors_from_similarity(sim, names, k)`.
- `pca_projection(x, n_components)`, `umap_projection(x, ...)` (returns None if `umap` missing).
- `residual_summary(x, x_hat, marker_names)` → per-marker residual mean/std/MAE/MSE.
- `posterior_mean_weights(mu_w, tau)`: numerically stable softmax of `mu_w/τ` (used when reconstructing weights from saved `mu_w`).
- `residual_norms(r)`: L2 norm of each residual latent row.
- `kl_history_columns(history_df)`: returns KL-related column names.

---

## 9. Plotting (`CyEmbed/plotting.py`)

Thin matplotlib/seaborn wrappers:

- `plot_training_history(history_df)` — loss + val_recon curves.
- `plot_matrix_heatmap(...)`, `plot_clustermap(...)` — for archetype-by-marker or similarity matrices.
- `plot_observed_vs_reconstructed(x, x_hat, marker_names, markers)` — subsampled scatter with y=x line.
- `plot_weight_histograms(w)` — one histogram per archetype column.
- `plot_umap_overlay(xy, values)` — continuous coloring.
- `plot_umap_categorical(xy, labels)` — discrete coloring with `tab20`.
- `plot_embedding_scatter(xy, labels)` — labeled 2D scatter for archetype coords.

None of these open figures with `plt.show()` by default (except `clustermap`), so notebooks can customize before rendering.

---

## 10. Typical workflow (as used by the notebooks)

1. **Load data** → `DataBundle` via `extract_matrix(adata=...)` or from a DataFrame/matrix.
2. **Fit scaler** → `scaler, fit_idx = fit_scaler(bundle.X, mode="zscore", sample_ids=bundle.sample_ids, balanced_max_per_sample=...)`, then `X = preprocess_array(bundle.X, scaler)`.
3. **Split** → `train_idx, val_idx = split_train_val_indices(n_cells, val_fraction, seed, stratify_labels=bundle.sample_ids)`.
4. **Define grid** → `sweep_configs = build_sweep_configs({"K":[6,7,8], "d":[8,16], "tau":[0.5,1.0], ...})` and a `base_config` with fixed fields (`model_type, epochs, lr, batch_size, hidden_dims, ...`).
5. **Run sweep** → `summary_df = run_sweep(x=X, marker_names=..., cell_ids=..., output_root="outputs/<exp>_sweep", base_config=..., sweep_configs=..., train_idx=..., val_idx=..., sample_ids=..., cluster_ids=..., scaler_state=scaler.to_dict())`.
6. **Select best run** → inspect `outputs/<exp>_sweep/sweep_summary.csv` (sorted by `val_recon`).
7. **Analyze** → `out = load_run_outputs(best_run_dir)`; derive per-marker stats, archetype rankings, per-group mean weights, archetype-archetype similarity, UMAP, etc.
8. **Plot / report** → use `plotting` helpers or the `tools/generate_*_report.py` scripts (standalone ReportLab PDFs).

Example notebook pairs in the repo: `01_mcf7_probabilistic_archetype_embedding_sweep.ipynb` / `02_mcf7_probabilistic_archetype_embedding_analysis.ipynb`, plus similarly named pairs for `breast_vim`, `archetype_embedding`, `withvim_single_sample_probabilistic`, and a low-K variant.

---

## 11. Config reference (union of fields read by `train_one_run`)

Required/common:
- `model_type` ("deterministic" | "probabilistic"), `decoder_type` ("factorized" | "direct").
- `K` (archetypes), `d` (latent dim), `hidden_dims` (iterable of ints for encoder MLP).
- `tau` (simplex temperature), `logit_normalizer`, `entmax_alpha`.
- `lr`, `weight_decay`, `epochs`, `batch_size`, `dropout`.
- `recon_loss_type` ("mse" | "huber"), `huber_delta`.
- `lambda_entropy`, `lambda_sep`, `lambda_balance`.
- `separation_mode` (default `"cosine_sq"`), `balance_mode` (default `"l2_uniform"`), `rbf_gamma`.
- `grad_clip_norm` (optional).
- `seed`, `deterministic`, `device` ("auto"|"cuda"|"mps"|"cpu").
- `early_stopping`, `patience`, `min_delta`, `restore_best_weights`.
- Logging: `print_every`, `progress_epoch`, `progress_sweep`.

Probabilistic-only:
- `use_residual_latent`, `residual_dim`.
- `beta_w`, `beta_r`, `kl_warmup_epochs`.
- `logvar_min`, `logvar_max`, `logvar_init_bias`.
- `prob_eval_mode` ("mean" | "sample" | "mc"), `prob_eval_samples`.

Sweep-only:
- `run_name` (force a specific folder name), `skip_existing_runs`.

Identity-ignored (don't affect fingerprint): `run_name, progress_sweep, progress_epoch, print_every, skip_existing_runs, resolved_device, software_versions`.

---

## 12. Notes for LLM users

- Numeric stability: entmax variants on MPS can produce NaNs; `train.py` aggressively checks and raises with diagnostic context. Prefer `"mean"` evaluation for metrics that feed early stopping.
- Metrics logged in `history.csv`: `epoch, loss, recon, entropy, separation, balance, val_recon`, and for probabilistic: `kl_w, kl_r, beta_w_eff, beta_r_eff`.
- Sweep idempotency is fingerprint-based; changing any non-ignored config field forces a fresh run, while re-running the same grid is cheap.
- `DataBundle` is *not* required by the trainer: `train_one_run` takes raw numpy arrays and lists. `DataBundle` is a convenience container for notebooks.
- `posterior_mean_weights` lets you re-derive probabilistic `W_mean` from saved `mu_w` without the model object, but only for `softmax`; entmax would need the model.
- The `tools/` scripts are standalone (not imported by the package) and generate the PDFs you see in `Analysis/reports/` and `outputs/reports/`.
