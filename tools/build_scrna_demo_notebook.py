"""Emit 01_bck44_scrna_archetype_embedding_sweep.ipynb.

Kept as a script so the notebook can be regenerated after edits to the template
rather than hand-patched as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(
    "/Users/ronguy/Dropbox/Work/CyTOF/Experiments/CyEmbed/"
    "01_bck44_scrna_archetype_embedding_sweep.ipynb"
)

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- 0. header
md(
    """
# CyEmbed on scRNA-seq — BCK_44, single sample

The first CyEmbed notebook that runs on transcriptomes rather than CyTOF. It is the worked
counterpart to `SCRNA_SEQ_GUIDE.md`, and it deviates from every other `01_*_sweep.ipynb` in
this repo in three ways that matter:

| | CyTOF notebooks | this notebook |
|---|---|---|
| features | ~40 markers, arcsinh | 2,000 HVGs, analytic Pearson residuals |
| `fit_scaler` mode | `"zscore"` | **`"none"`** — residuals are already variance-stabilised |
| stratification | by cluster or sample | **none available** — see the split cell |

## Read this before trusting any number below

**352 cells.** That is very small for archetypal analysis. The K sweep here is a demonstration
of the procedure, not a confident biological answer — with 352 cells and a 20% validation split
you are selecting K on ~70 held-out cells. Treat the winning K as a hypothesis.

**Median library size 765, detection rate 12%.** A shallow multiome RNA arm. Residuals are noisier
than the guide's worked example assumes.

**Single sample, so `use_sample_offset` is not exercised.** BCK_44 is one CellRanger-ARC run
(every barcode carries the `-1` GEM suffix, and `.obs` has no patient/sample/batch column). The
per-patient intercept `B` has nothing to correct here. Section 8 shows where it would go.

**HVGs were chosen upstream by dispersion, not residual variance.** The guide asks for the top
genes by Pearson residual variance. The 2,000 genes in this file were selected by
`sc.pp.highly_variable_genes(flavor="seurat")` on log1p data in the ProbAE NB pipeline, and the
raw 10x matrix is a 0-byte Dropbox placeholder, so reselecting from all ~36k genes is not possible
here. Section 3 ranks the 2,000 by residual variance so you can see the disagreement and subset
further if you want.

**No ground truth.** Unlike `tools/verify_sample_offset_scrna.py`, nothing here plants known
archetypes, so there is no `w_recovery` oracle. Section 6 falls back on the criteria that do not
need one: `val_recon`, dead archetypes, archetype redundancy, and cross-seed agreement.
"""
)

# ---------------------------------------------------------------- 1. imports
code(
    """
from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from CyEmbed.analysis import (
    archetype_marker_rankings,
    cosine_similarity_matrix,
    dominant_assignments,
    load_run_outputs,
    weight_entropy,
)
from CyEmbed.data import DataBundle, fit_scaler, preprocess_array, split_train_val_indices
from CyEmbed.train import build_sweep_configs, run_sweep
from CyEmbed.utils import collect_software_versions, save_json, set_seed

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)
"""
)

# ---------------------------------------------------------------- 2. config
code(
    '''
# === Editable configuration ===
OUTPUT_ROOT = Path("outputs/bck44_scrna_archetype_sweep")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"OUTPUT_ROOT: {OUTPUT_ROOT.resolve()}  (completed runs are skipped, not overwritten)")

GLOBAL_CFG = {
    "seed": 7,             # overridden per run by SWEEP_GRID["seed"]
    "deterministic": True,
    # CPU, deliberately. MPS was measured 11.6x SLOWER than CPU on this workload
    # (23.3s vs 2.0s) -- the matrices are small and the transfer overhead dominates.
    "device": "cpu",
}

DATA_CFG = {
    # Derived h5ad from the ProbAE NB pipeline: (352, 2000), .X = raw integer counts.
    # NOT the 10x .h5 or the .h5seurat -- both are 0-byte Dropbox placeholders on this machine.
    "counts_h5ad": (
        "/Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv/data/"
        "bck44_scrna_hvg_counts.h5ad"
    ),
    "sample_col": None,    # single sample -- no patient column exists
    "cluster_col": None,   # no annotation in this object
}

RESIDUAL_CFG = {
    # Analytic Pearson residuals, Lause/Berens/Kobak (2021, Genome Biology).
    "theta": 100.0,        # NB overdispersion; 100 is the paper's default and is near-Poisson
    "clip": "sqrt_n",      # "sqrt_n" (paper), a float, or None
    "n_top_genes": None,   # None = keep all 2000; set e.g. 1000 to subset by residual variance
}

PREPROCESS_CFG = {
    # "none", NOT "zscore". Pearson residuals are already centred and unit-variance by
    # construction; z-scoring again re-inflates genes the residual transform deliberately
    # shrank. "robust_zscore" is worse -- it divides by an IQR that is 0 for any gene detected
    # in <25% of cells, which on 12%-detection data is most of them.
    "mode": "none",
}

SPLIT_CFG = {
    "val_fraction": 0.2,
    # No cluster labels and no sample column, so stratification is unavailable. On a
    # multi-patient object set this to "sample" -- an unstratified split can leave a patient
    # entirely out of train, and that patient's offset column in B then never gets fit.
    "stratify_by": None,
}

BASE_TRAIN_CFG = {
    # 400 was not enough -- every run hit the cap with stopped_early=False, meaning val_recon
    # was still improving when training was cut off. 1500 lets early stopping actually decide.
    "epochs": 1500,
    "early_stopping": True,
    "patience": 60,
    "min_delta": 0.0,
    "restore_best_weights": True,
    "weight_decay": 1e-5,
    "dropout": 0.0,
    "huber_delta": 1.0,
    "separation_mode": "cosine_sq",
    "balance_mode": "l2_uniform",
    "rbf_gamma": 1.0,
    "print_every": 50,
    "progress_sweep": True,
    "progress_epoch": False,
    "skip_existing_runs": True,
}

SWEEP_GRID = {
    "decoder_type": ["factorized"],
    # Extended to 8 after a first pass: val_recon was still falling at K=6, i.e. the minimum
    # sat on the grid boundary, which means the grid -- not the data -- was choosing K.
    "K": [3, 4, 5, 6, 7, 8],
    # d fixed. rank(A_hat) <= min(K, d), so d >= K is not a bottleneck -- but d=16 still beat
    # d=8 (w_recovery 0.988 vs 0.712) at identical rank, so d is doing something beyond rank.
    # Sweep it separately (section 7) rather than crossing it with K.
    "d": [8],
    "hidden_dims": [[256, 128]],
    "lr": [1e-3],
    "batch_size": [256],    # 352 cells total -- 2048 would be one batch per epoch
    "recon_loss_type": ["mse"],
    # The regulariser package. lambda_balance is the measured workhorse: on the synthetic
    # scRNA benchmark it moved w_recovery 0.834 -> 0.992.
    "lambda_entropy": [1e-3],
    "lambda_sep": [1e-3],
    "lambda_balance": [5e-2],
    "logit_normalizer": ["entmax"],
    "entmax_alpha": [1.5],
    "tau": [1.0],
    # Three seeds, not one. Run-to-run variance is ~5% and K margins are ~6% -- a single seed
    # cannot tell them apart. seed is part of the config fingerprint, so each lands in its
    # own run directory and they aggregate cleanly.
    "seed": [7, 17, 23],
}

MAX_RUNS = None
'''
)

# ---------------------------------------------------------------- 3. load
md("## 1. Load counts")

code(
    '''
# === Load raw counts ===
set_seed(GLOBAL_CFG["seed"], deterministic=GLOBAL_CFG["deterministic"])

counts_path = Path(DATA_CFG["counts_h5ad"])
if not counts_path.exists() or counts_path.stat().st_size == 0:
    raise FileNotFoundError(
        f"{counts_path} is missing or is a 0-byte Dropbox placeholder. "
        "Right-click -> 'Make available offline' in Dropbox, or regenerate it with "
        "ProbAE_Deconv/notebooks/experiment_suite/200_bck44_scrna_nb_k_sweep.ipynb."
    )

adata = ad.read_h5ad(counts_path)
counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
counts = np.asarray(counts, dtype=np.float64)

if not np.allclose(counts, np.round(counts)):
    raise ValueError(
        "Expected raw integer counts in .X. Pearson residuals computed from normalized or "
        "log1p values are meaningless -- check that .X was not overwritten."
    )

gene_names = [str(g) for g in adata.var_names]
cell_ids = [str(c) for c in adata.obs_names]

print(f"cells x genes      : {counts.shape[0]} x {counts.shape[1]}")
print(f"library size       : median {np.median(counts.sum(1)):.0f}, "
      f"range [{counts.sum(1).min():.0f}, {counts.sum(1).max():.0f}]")
print(f"detection rate     : {(counts > 0).mean():.1%} of the matrix is non-zero")
print(f"sample/patient col : {DATA_CFG['sample_col']}  (single sample -- B is not exercised)")
'''
)

# ---------------------------------------------------------------- 4. residuals
md(
    """
## 2. Analytic Pearson residuals

The load-bearing preprocessing step. CyEmbed's loss is MSE — a Gaussian with constant variance —
which on raw counts is simply the wrong noise model: a gene's variance grows with its mean, so
MSE would let a handful of high-expression genes dictate every archetype. Pearson residuals
divide that dependence out:

$$z_{ij} = \\frac{x_{ij} - \\hat\\mu_{ij}}{\\sqrt{\\hat\\mu_{ij} + \\hat\\mu_{ij}^2/\\theta}}
\\qquad \\hat\\mu_{ij} = \\frac{\\left(\\sum_j x_{ij}\\right)\\left(\\sum_i x_{ij}\\right)}{\\sum_{ij} x_{ij}}$$

After this, MSE is defensible and every gene contributes on comparable footing.

One deviation from the reference implementation, forced by the data: `mu` is computed from row
and column sums of the **2,000-gene** matrix, because the full transcriptome is not available
here. The 2,000 genes carry only part of each cell's library, so the depth estimate is coarser
than it would be if residuals were computed genome-wide and then subset. The cell below checks
the resulting residuals against `obs['total_counts']` (the true full-transcriptome depth) so you
can see whether it mattered.
"""
)

code(
    '''
# === Analytic Pearson residuals ===
def pearson_residuals(x: np.ndarray, theta: float = 100.0, clip: object = "sqrt_n") -> np.ndarray:
    """Analytic Pearson residuals under an NB null (Lause/Berens/Kobak 2021)."""
    counts_per_cell = x.sum(axis=1, keepdims=True)
    counts_per_gene = x.sum(axis=0, keepdims=True)
    total = x.sum()
    mu = counts_per_cell @ counts_per_gene / total
    z = (x - mu) / np.sqrt(mu + mu**2 / theta)
    if clip == "sqrt_n":
        limit = np.sqrt(x.shape[0])
    elif clip is None:
        limit = None
    else:
        limit = float(clip)
    if limit is not None:
        z = np.clip(z, -limit, limit)
    return z


resid = pearson_residuals(counts, theta=RESIDUAL_CFG["theta"], clip=RESIDUAL_CFG["clip"])

# Rank genes by residual variance -- the selection criterion the guide actually asks for.
resid_var = resid.var(axis=0)
order = np.argsort(resid_var)[::-1]

n_top = RESIDUAL_CFG["n_top_genes"]
if n_top is not None and int(n_top) < resid.shape[1]:
    keep = np.sort(order[: int(n_top)])
    resid = resid[:, keep]
    gene_names_kept = [gene_names[i] for i in keep]
    print(f"Subset to top {int(n_top)} genes by residual variance.")
else:
    gene_names_kept = list(gene_names)
    print(f"Keeping all {resid.shape[1]} genes (n_top_genes=None).")

print(f"residual matrix    : {resid.shape[0]} x {resid.shape[1]}")
# std well above 1 is EXPECTED here and is not a failure: these 2,000 genes were preselected
# as highly variable, so the transform's unit-variance null applies to the whole transcriptome,
# not to a variance-enriched subset of it. Mean near 0 is the check that matters.
print(f"residual mean/std  : {resid.mean():.4f} / {resid.std():.4f}  (want mean ~0; std > 1 on an HVG subset)")
print(f"residual range     : [{resid.min():.2f}, {resid.max():.2f}]  "
      f"(clip at +/-{np.sqrt(counts.shape[0]):.2f})")
print()
print("Top 15 genes by residual variance (upstream HVG rank in parentheses):")
disp_rank = {g: i for i, g in enumerate(gene_names)}
for r, i in enumerate(order[:15], start=1):
    print(f"  {r:2d}. {gene_names[i]:<12s} resid_var={resid_var[i]:7.2f}  (dispersion HVG #{disp_rank[gene_names[i]] + 1})")
'''
)

# ---------------------------------------------------------------- 5. QC
md(
    """
## 3. Preprocessing QC — check this before training

Two failure modes make everything downstream meaningless, and both are silent. If either check
below fails, stop: no choice of `K` or `d` will rescue it.
"""
)

code(
    '''
# === Preprocessing QC ===
lib_size = counts.sum(axis=1)
full_depth = adata.obs["total_counts"].to_numpy() if "total_counts" in adata.obs else lib_size

# CHECK 1: residuals must not track sequencing depth.
# If archetype weights end up correlated with library size, you have found a depth artifact,
# not biology. The residual transform is what removes it -- verify that it did.
cell_mean_resid = resid.mean(axis=1)
r_depth = np.corrcoef(np.log10(lib_size), cell_mean_resid)[0, 1]
r_depth_full = np.corrcoef(np.log10(full_depth), cell_mean_resid)[0, 1]
print(f"corr(log10 lib size, mean residual)      : {r_depth:+.3f}   (HVG-subset depth)")
print(f"corr(log10 total_counts, mean residual)  : {r_depth_full:+.3f}   (full transcriptome)")
print("  -> want |r| < 0.3.")
if max(abs(r_depth), abs(r_depth_full)) >= 0.3:
    print("  !! BORDERLINE on this dataset (measured about -0.32 / -0.36). The sign is negative:")
    print("     deeper cells have slightly LOWER mean residual, the opposite of depth leaking")
    print("     through as inflated signal. With a 766-median library the NB null is a poor fit")
    print("     in the low-depth tail, which is the likely cause. Check in section 8 whether")
    print("     archetype usage tracks library size -- that is the failure that would matter.")
print()

# CHECK 2: per-gene residual variance should not be dominated by a handful of genes.
top_share = np.sort(resid.var(axis=0))[::-1][:10].sum() / resid.var(axis=0).sum()
print(f"variance share of the top 10 genes       : {top_share:.1%}")
print("  -> want < ~15%. Higher means a few genes will define every archetype.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(resid.ravel(), bins=120, log=True)
axes[0].set_xlabel("Pearson residual")
axes[0].set_ylabel("count (log)")
axes[0].set_title("Residual distribution")

axes[1].scatter(np.log10(lib_size), cell_mean_resid, s=8, alpha=0.5)
axes[1].set_xlabel("log10 library size (HVG subset)")
axes[1].set_ylabel("mean residual per cell")
axes[1].set_title(f"Depth dependence (r = {r_depth:+.3f})")

axes[2].scatter(counts.mean(axis=0), resid.var(axis=0), s=6, alpha=0.4)
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("mean count")
axes[2].set_ylabel("residual variance")
axes[2].set_title("Mean-variance relation after transform")

plt.tight_layout()
'''
)

# ---------------------------------------------------------------- 6. bundle + split
md("## 4. Bundle, scale, split")

code(
    '''
# === Bundle and split ===
bundle = DataBundle(
    X=resid.astype(np.float32),
    marker_names=gene_names_kept,   # "marker" means "gene" throughout CyEmbed
    cell_ids=cell_ids,
    sample_ids=None,                # single sample
    cluster_ids=None,               # no annotation available
)

# mode="none" -- see PREPROCESS_CFG for why z-scoring residuals is wrong.
scaler, scaler_fit_idx = fit_scaler(bundle.X, mode=PREPROCESS_CFG["mode"])
X_proc = preprocess_array(bundle.X, scaler)

train_idx, val_idx = split_train_val_indices(
    n_cells=X_proc.shape[0],
    val_fraction=SPLIT_CFG["val_fraction"],
    seed=GLOBAL_CFG["seed"],
    stratify_labels=None,   # nothing to stratify on -- see SPLIT_CFG
)

print(f"scaler mode : {scaler.mode}")
print(f"train cells : {len(train_idx)}")
print(f"val cells   : {len(val_idx)}   <- K is being selected on this many cells")
'''
)

# ---------------------------------------------------------------- 7. sweep
md(
    """
## 5. K sweep

12 runs: K ∈ {3,4,5,6} × 3 seeds, `d` held at 8. Runs are fingerprinted on the full config
(seed included), so re-executing this cell skips completed runs rather than retraining them.
"""
)

code(
    '''
# === Build sweep and train ===
sweep_configs = build_sweep_configs(SWEEP_GRID)
if MAX_RUNS is not None:
    sweep_configs = sweep_configs[: int(MAX_RUNS)]
print(f"Total sweep runs: {len(sweep_configs)}")

save_json(
    OUTPUT_ROOT / "notebook_config.json",
    {
        "global": GLOBAL_CFG,
        "data": DATA_CFG,
        "residual": RESIDUAL_CFG,
        "preprocess": PREPROCESS_CFG,
        "split": SPLIT_CFG,
        "base_train": BASE_TRAIN_CFG,
        "sweep_grid": SWEEP_GRID,
        "software_versions": collect_software_versions(),
        "scaler": scaler.to_dict(),
    },
)

summary_df = run_sweep(
    x=X_proc,
    marker_names=bundle.marker_names,
    cell_ids=bundle.cell_ids,
    output_root=OUTPUT_ROOT,
    base_config={**GLOBAL_CFG, **BASE_TRAIN_CFG},
    sweep_configs=sweep_configs,
    train_idx=train_idx,
    val_idx=val_idx,
    sample_ids=bundle.sample_ids,
    cluster_ids=bundle.cluster_ids,
    scaler_state=scaler.to_dict(),
)

summary_df.to_csv(OUTPUT_ROOT / "sweep_summary_sorted.csv", index=False)
summary_df.head(20)
'''
)

# ---------------------------------------------------------------- 8. K selection
md(
    """
## 6. Choosing K without ground truth

Four criteria, in descending order of how much I would trust them here.

1. **Archetype redundancy** — max off-diagonal `|cos|` between archetype profiles. This is the
   leading criterion. Above the true K this model does not gracefully split archetypes, it
   **collapses**: two archetypes become literally identical. Redundancy near 1.0 means K is too
   high, and it is the earliest signal you get.
2. **`val_recon`** — has a genuine minimum, contrary to the intuition that reconstruction
   improves monotonically with K. Worth reading, but the margins are small.
3. **Dead archetypes** — note the metric `dead_archetypes_lt_1pct` uses an **absolute**
   threshold (`w_bar < 0.01`), not the relative `0.5/K` you might expect. At K=6 uniform usage
   is 0.167, so the absolute threshold is lenient; the relative count is computed below too.
4. **Cross-seed stability** — reported last and deliberately distrusted. It scores a *perfect*
   1.000 on collapsed models, because every seed reliably finds the same degenerate solution.
   High stability with high redundancy means agreement on garbage.

### What this run found

| K | val_recon | redundancy | dead (rel) | stability |
|---|---|---|---|---|
| 3 | 2.765 | 0.838 | 0.0 | 0.877 |
| 4 | 2.710 | 0.732 | 0.3 | 0.738 |
| 5 | 2.745 | 0.828 | 1.0 | 0.728 |
| **6** | **2.649** | **0.515** | 0.7 | 0.789 |
| 7 | 2.704 | 0.741 | 1.3 | 0.608 |
| 8 | 2.667 | 0.725 | 2.0 | 0.687 |

**K = 6**, on two criteria that agree independently: lowest `val_recon`, and a redundancy of
0.515 that is far below every other K (all ≥ 0.72). Note that `val_recon` has a genuine interior
minimum rather than falling monotonically — the thing that makes it usable for selecting K at all.

Two warnings the table earns:

- **Do not take "best by `val_recon`" from section 8 at face value.** The single best *run* is
  K=7, but K=7's *mean* is worse than K=6's and its spread is six times larger (sd 0.109 vs
  0.017) — one seed got lucky. This is exactly the failure that a one-seed sweep would have
  written down as a result.
- **Dead archetypes rise steadily with K** on the relative threshold (0.0 → 2.0), while the
  absolute `dead_archetypes_lt_1pct` metric stays at 0 until K=7. On 352 cells the relative
  count is the more honest of the two.
"""
)

code(
    '''
# === Per-K diagnostics ===
def archetype_profiles(run_dir: Path) -> np.ndarray | None:
    """(K, M) archetype profiles, whichever decoder produced them."""
    out = load_run_outputs(run_dir)
    for key in ("A_hat", "A"):
        if key in out and isinstance(out[key], np.ndarray):
            return out[key]
    if "Z" in out and "E" in out:
        return out["Z"] @ out["E"].T
    return None


def stability(a: np.ndarray, b: np.ndarray) -> float:
    """Hungarian-matched mean cosine between two archetype sets."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sim = np.abs(an @ bn.T)
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


# run_sweep's summary does not carry `seed`, so recover it from each run's saved config.
def run_seed(run_dir: Path) -> int | None:
    cfg = load_run_outputs(run_dir)["config"]
    return cfg.get("seed")


summary_df["seed"] = [run_seed(Path(p)) for p in summary_df["run_dir"]]

rows = []
for k, grp in summary_df.groupby("K"):
    profiles, redundancy, dead_rel = [], [], []
    for run_dir in grp["run_dir"]:
        a = archetype_profiles(Path(run_dir))
        if a is None:
            continue
        profiles.append(a)
        cos = np.abs(cosine_similarity_matrix(a))
        np.fill_diagonal(cos, 0.0)
        redundancy.append(cos.max())
        out = load_run_outputs(Path(run_dir))
        w = out.get("W_mean", out.get("W"))
        if w is not None:
            dead_rel.append(int((w.mean(axis=0) < 0.5 / int(k)).sum()))

    pairs = [
        stability(profiles[i], profiles[j])
        for i in range(len(profiles))
        for j in range(i + 1, len(profiles))
    ]
    rows.append({
        "K": int(k),
        "val_recon": grp["val_recon"].mean(),
        "val_recon_sd": grp["val_recon"].std(),
        "redundancy_max_cos": float(np.mean(redundancy)) if redundancy else np.nan,
        # NB: the summary column is `dead_archetypes_val`; the underlying metric key
        # (`dead_archetypes_lt_1pct`) is renamed on the way out in train.py:160.
        "dead_abs_1pct": grp["dead_archetypes_val"].mean(),
        "dead_rel_half_over_k": float(np.mean(dead_rel)) if dead_rel else np.nan,
        "stability_across_seeds": float(np.mean(pairs)) if pairs else np.nan,
        "n_seeds": len(grp),
    })

k_table = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
display(k_table)

print("Read it this way:")
print("  redundancy near 1.0  -> archetypes have collapsed; K is too high")
print("  val_recon            -> look for a minimum, not a monotone trend")
print("  dead_rel             -> archetypes carrying < half of uniform usage")
print("  stability            -> ignore when redundancy is high (it agrees on collapse)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].errorbar(k_table["K"], k_table["val_recon"], yerr=k_table["val_recon_sd"],
                 marker="o", capsize=4)
axes[0].set_xlabel("K"); axes[0].set_ylabel("val_recon"); axes[0].set_title("Reconstruction (mean +/- sd over seeds)")

axes[1].plot(k_table["K"], k_table["redundancy_max_cos"], marker="o", color="crimson")
axes[1].axhline(0.9, ls="--", c="grey", lw=1)
axes[1].set_xlabel("K"); axes[1].set_ylabel("max |cos| off-diagonal")
axes[1].set_title("Archetype redundancy (lower is better)")

axes[2].plot(k_table["K"], k_table["stability_across_seeds"], marker="o", label="stability")
axes[2].plot(k_table["K"], k_table["dead_rel_half_over_k"], marker="s", label="dead (relative)")
axes[2].set_xlabel("K"); axes[2].legend(frameon=False)
axes[2].set_title("Stability and dead archetypes")

plt.tight_layout()
'''
)

# ---------------------------------------------------------------- 9. d sweep
md(
    """
## 7. `d` sweep (optional, run after fixing K)

`d` sizes `Z (K,d)` and `E (M,d)`, and `rank(A_hat) ≤ min(K, d)` — so above `d = K` it cannot
increase the rank of what the decoder can express. The natural conclusion is that `d > K` is
wasted compute. **That conclusion is wrong**: on the synthetic scRNA benchmark `d=16` beat `d=8`
(w_recovery 0.988 vs 0.712) at identical rank. Extra embedding dimensions appear to help
optimisation even when they add no expressive power. So sweep it rather than deriving it.

Set `K_FIXED` to whatever section 6 chose, then run.
"""
)

code(
    '''
# === d sweep at fixed K ===
K_FIXED = int(k_table.loc[k_table["redundancy_max_cos"].idxmin(), "K"])
print(f"Sweeping d at K={K_FIXED} (chosen by lowest redundancy; override K_FIXED to change).")

D_GRID = {**SWEEP_GRID, "K": [K_FIXED], "d": [4, 8, 16, 32]}
d_configs = build_sweep_configs(D_GRID)
print(f"Total d-sweep runs: {len(d_configs)}")

# Its OWN output root, deliberately. run_sweep writes `sweep_summary.csv` at the root it is
# given, so pointing both sweeps at OUTPUT_ROOT makes the second silently overwrite the first
# and leaves a K-sweep summary on disk containing only K=K_FIXED rows.
D_OUTPUT_ROOT = OUTPUT_ROOT / "d_sweep"
D_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

d_summary = run_sweep(
    x=X_proc,
    marker_names=bundle.marker_names,
    cell_ids=bundle.cell_ids,
    output_root=D_OUTPUT_ROOT,
    base_config={**GLOBAL_CFG, **BASE_TRAIN_CFG},
    sweep_configs=d_configs,
    train_idx=train_idx,
    val_idx=val_idx,
    sample_ids=bundle.sample_ids,
    cluster_ids=bundle.cluster_ids,
    scaler_state=scaler.to_dict(),
)

d_table = (
    d_summary.groupby("d")
    .agg(val_recon=("val_recon", "mean"),
         val_recon_sd=("val_recon", "std"),
         marker_corr=("mean_marker_corr_val", "mean"),
         n_seeds=("val_recon", "size"))
    .reset_index()
)
display(d_table)
'''
)

# ---------------------------------------------------------------- 10. interpret
md(
    """
## 8. Interpreting the winning model

### Reading archetype loadings on residuals

`A_hat` rows are archetype profiles **in Pearson-residual space**, not expression space. A
positive loading means "this archetype has more of this gene than a depth-matched average cell";
a negative loading is genuine *depletion*, not a modelling artifact. Do not read these as
expression levels, and do not expect them to be non-negative — CyEmbed is not NMF, and the sign
carries information.

### What this run found

At K=7 the archetypes are largely readable as breast tissue, which is the basic sanity check:

- **luminal secretory** — SCGB2A2, SCGB1D2, SCGB2A1, PIP, MUCL1
- **basal / myoepithelial** — KRT15, KRT17, SFRP1, GABRP, PTN
- **smooth muscle** — ACTA2, MYH11, TAGLN, MYL9, TPM2
- **endothelial** — VWF, EMCN, A2M, SPARCL1, LDB2

**But one archetype is an artifact, and the depth check below is what catches it.** Archetype 1
is the *highest-usage* archetype (0.254) and correlates **−0.467** with log library size. Its
gene list — TALAM1, AKAP13, BTRC, LINC00472, NOVA1 — is a grab-bag of long transcripts and
lncRNAs with no coherent program. It is the shallow cells, wearing a plausible-looking gene list.
That is precisely the failure mode that makes a gene-list-only reading of archetypes dangerous:
the list looked fine, and only the correlation gave it away.

### Where the per-patient offset would go

Not exercised here — BCK_44 is one sample. On a multi-patient object you would add
`"use_sample_offset": True` to `BASE_TRAIN_CFG`, pass a real `sample_ids` array into
`run_sweep`, and set `SPLIT_CFG["stratify_by"] = "sample"`. The decoder then becomes
`x̂ = w Z Eᵀ + b + B[s]`, so archetypes model deviation from each patient's own baseline rather
than re-encoding patient identity. `B` is warm-started at the centred per-patient mean and
excluded from weight decay; without the warm start the factorized decoder loses the race for the
shift and bakes it into the archetypes instead.
"""
)

code(
    '''
# === Best run: archetypes, usage, gene modules ===
best_row = summary_df.sort_values("val_recon").iloc[0]
best_dir = Path(best_row["run_dir"])
print(f"Best by val_recon: {best_row['run_id']}  (K={best_row['K']}, d={best_row['d']}, "
      f"seed={best_row['seed']})")
print("NOTE: 'best by val_recon' is not necessarily the K section 6 endorsed -- "
      "check redundancy before adopting it.")

out = load_run_outputs(best_dir)
A = out.get("A_hat")
if A is None and "Z" in out and "E" in out:
    A = out["Z"] @ out["E"].T
W = out.get("W_mean", out.get("W"))

# Top genes per archetype, positive and negative.
rankings = archetype_marker_rankings(A, bundle.marker_names, top_n=15)
display(rankings)

for i in range(A.shape[0]):
    order_i = np.argsort(A[i])
    up = [bundle.marker_names[j] for j in order_i[::-1][:10]]
    down = [bundle.marker_names[j] for j in order_i[:10]]
    print(f"\\nArchetype {i}  (mean usage {W[:, i].mean():.3f})")
    print(f"  enriched : {', '.join(up)}")
    print(f"  depleted : {', '.join(down)}")

# Usage and how mixed cells are. High entropy = cells sitting between archetypes, which is the
# thing archetypal analysis is for; entropy near 0 means it has degenerated into hard clustering.
ent = weight_entropy(W)
dom = dominant_assignments(W, bundle.cell_ids)
print(f"\\nmean entropy(W)       : {ent.mean():.3f}  (max = {np.log(A.shape[0]):.3f} for K={A.shape[0]})")
print(f"cells with w_max > 0.8: {(W.max(axis=1) > 0.8).mean():.1%}")

# The depth check that matters, promised in section 3. Section 3 only asked whether depth
# survived into the residuals; this asks whether it survived all the way into the archetypes.
# An archetype whose usage tracks library size is a sequencing-depth artifact wearing a
# biological costume -- it will have a plausible gene list and mean nothing.
print("\\ncorr(log10 library size, archetype usage):")
depth_corr = [np.corrcoef(np.log10(lib_size), W[:, i])[0, 1] for i in range(A.shape[0])]
for i, r in enumerate(depth_corr):
    flag = "   <-- suspect, inspect this one" if abs(r) > 0.4 else ""
    print(f"  archetype {i}: {r:+.3f}{flag}")
print("  -> |r| > 0.4 means that archetype is largely a depth axis, not a program.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(A.shape[0]), W.mean(axis=0))
axes[0].axhline(1.0 / A.shape[0], ls="--", c="grey", lw=1, label="uniform")
axes[0].set_xlabel("archetype"); axes[0].set_ylabel("mean usage"); axes[0].legend(frameon=False)
axes[0].set_title("Archetype usage")

axes[1].hist(ent, bins=40)
axes[1].set_xlabel("entropy(w)"); axes[1].set_title("How mixed are cells?")

cos = cosine_similarity_matrix(A)
im = axes[2].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
axes[2].set_title("Archetype similarity")
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
'''
)

# ---------------------------------------------------------------- 11. gene modules
md(
    """
## 9. Gene modules from the embedding `E`

`E ∈ R^{M×d}` places every gene in a `d`-dimensional space, and genes close together there load
similarly across archetypes — so yes, proximity in `E` is informative. Two honest caveats:

- **`E` is not privileged.** Gene co-membership recovered from `E` and from `A_hat.T` scored
  identically on the synthetic benchmark (AUC 1.000 both ways). `E` is a convenience, not the
  reason to prefer the factorized decoder.
- **Modules are only ~0.77 correlated across seeds.** The archetypes themselves move between
  restarts, so a module found in one run is a hypothesis, not a finding. Check it in the other
  two seeds below before believing it.
"""
)

code(
    '''
# === Gene neighbourhoods in E ===
E = out.get("E")
if E is None:
    print("No E in this run (direct decoder has no gene embedding). Skipping.")
else:
    from CyEmbed.analysis import nearest_neighbors_from_similarity

    gene_cos = cosine_similarity_matrix(E)
    # archetype_marker_rankings names the column `marker`, not `marker_name`.
    query_genes = rankings.loc[rankings["direction"] == "positive", "marker"].unique()[:6]

    for g in query_genes:
        gi = bundle.marker_names.index(g)
        nn = np.argsort(gene_cos[gi])[::-1][1:9]
        print(f"{g:<12s} -> {', '.join(bundle.marker_names[j] for j in nn)}")

    # Do these neighbourhoods survive a different seed?
    same_k = summary_df[summary_df["K"] == best_row["K"]]
    others = [Path(p) for p in same_k["run_dir"] if Path(p) != best_dir]
    if others:
        agreements = []
        for other in others:
            e2 = load_run_outputs(other).get("E")
            if e2 is None:
                continue
            c2 = cosine_similarity_matrix(e2)
            iu = np.triu_indices_from(gene_cos, k=1)
            agreements.append(np.corrcoef(gene_cos[iu], c2[iu])[0, 1])
        if agreements:
            print(f"\\nGene-gene similarity agreement across seeds: "
                  f"{np.mean(agreements):.3f} (n={len(agreements)} pairs)")
            print("  ~0.77 is what the synthetic benchmark gave. Much lower means the modules "
                  "in this run are seed-specific noise.")
'''
)

# ---------------------------------------------------------------- 12. next
md(
    """
## 10. Where to go next

- **Multi-patient.** The offset is the feature this notebook cannot demonstrate. Point section 1
  at an object with a real patient column, set `use_sample_offset: True`, and stratify the split
  by sample.
- **Deeper data.** 352 cells and a 765-median library is thin. The procedure here transfers
  unchanged to a larger object; the conclusions do not.
- **Reselect HVGs by residual variance.** Section 3 shows how far the dispersion-based selection
  in this file diverges from the residual-variance ranking the guide asks for. Once the raw 10x
  matrix is available offline, reselect from all ~36k genes rather than re-ranking these 2,000.
"""
)


def main() -> None:
    cells = []
    for idx, (kind, src) in enumerate(CELLS):
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell_id = f"cell-{idx:02d}"
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source})
        else:
            cells.append({
                "cell_type": "code",
                "id": cell_id,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
