# Running CyEmbed on scRNA-seq

CyEmbed was written for CyTOF. This guide covers what changes when you point it at single-cell
RNA-seq instead — which settings become mandatory, which become dangerous, and how to tell
whether the archetypes you get back are biology or artifact.

Companion documents: `CODEBASE_DESCRIPTION.md` for the full API, and
`../mcRBM/Desc/scrna_seq_guide.md` for the same treatment of mcRBM.

---

## 1. Orientation: "marker" means "gene"

The package says `marker_names`, `MarkerScaler`, `num_markers` everywhere. On scRNA-seq every one
of those means *gene*. Nothing needs renaming — just read it that way.

Three differences drive everything else in this guide:

| | CyTOF | scRNA-seq |
|---|---|---|
| Features | ~40 markers | 1,000–3,000 HVGs |
| Measurement | continuous ion counts | integer UMI counts |
| Zeros | rare | most of the matrix |

The last one is the important one. A zero in CyTOF means "not expressed." A zero in scRNA-seq
might mean that, or it might mean the transcript was there and simply wasn't captured. CyEmbed's
reconstruction loss is MSE or Huber (`losses.py:17-20`) — a Gaussian likelihood with constant
variance — which has no way to represent that ambiguity and no idea that a gene's variance grows
with its mean.

**The fix is entirely upstream.** Feed CyEmbed SCTransform Pearson residuals and MSE stops being
wrong: residuals are approximately homoscedastic, which is precisely the regime where a constant-
variance Gaussian is the correct likelihood. Everything below assumes you have done that.

One thing the package will not do for you: **feature selection**. Subset to HVGs before you call
`extract_matrix`. There is no gene filtering anywhere in CyEmbed.

---

## 2. Getting data in

The residual h5ads produced by
`../ProbAE_Deconv/notebooks/experiment_suite/300_scrna_sct_gaussian_k_sweep.ipynb` have a simple
shape, and CyEmbed reads them directly:

- Residuals live in **`X`** — dense float32. No layers, no `obsm`.
- `var` is indexed by gene symbol, carrying `residual_variance` as provenance.
- `obs` contains **`cell_id` and nothing else**.

```python
import anndata as ad
from CyEmbed.data import extract_matrix

adata = ad.read_h5ad("../ProbAE_Deconv/data/PDX_02_sct_pearson_residuals_hvg.h5ad")

bundle = extract_matrix(adata=adata, source="X")
# bundle.X            -> (5774, 1000) float32
# bundle.marker_names -> gene symbols, from adata.var_names
# bundle.cell_ids     -> barcodes, from adata.obs_names
```

`extract_matrix` is keyword-only (`data.py:103`). `source` accepts `"X"`, `"layer"` (requires
`layer=`), or `"obsm"` (requires `obsm_key=`). Sparse input is densified automatically by
`_to_numpy` (`data.py:10`), so a CSR `.X` works without ceremony.

Note the import style: the real notebooks import from submodules (`from CyEmbed.data import ...`)
rather than the top-level package. Several functions you will want — `balanced_downsample_indices`,
`purity_summary` — are not in `__init__.py`'s `__all__` and are only reachable that way.

---

## 3. The one setting you must change

```python
scaler, fit_idx = fit_scaler(bundle.X, mode="none")
X = preprocess_array(bundle.X, scaler)
```

**`mode="none"`.** Not `"zscore"`. SCT Pearson residuals are already centred and variance-
stabilised per gene — the source h5ad measures `mean = -0.0000, std = 0.9858`. Z-scoring on top
undoes the stabilisation you paid for, rescaling each gene by a standard deviation that
SCTransform already normalised away.

This is why the existing sweep config sets `transform: "none"` and `normalization: "none"` with
the comment *"SCT Pearson residuals are already ~N(0,1) per gene."* Follow it.

### `robust_zscore` is broken on sparse data — do not use it

This one will bite hard and it is worth understanding rather than just avoiding.

```python
# data.py:63-70
elif self.mode == "robust_zscore":
    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center[None, :]), axis=0)
    scale = 1.4826 * mad
...
scale = np.maximum(scale, self.eps)   # eps = 1e-8
```

For any gene detected in **under 50% of cells** — which is most genes — the median is 0. Then
`|x - 0| = |x|`, whose median is *also* 0. MAD is zero, `scale` clamps to `1e-8`, and every
detected value gets multiplied by roughly 1e8.

You will not get silently wrong numbers; `train.py`'s non-finite guards raise `FloatingPointError`
after the first step. But the mode is simply unusable here, and the error message will not tell you
why. If you are working on raw or log1p data rather than residuals, this is the trap.

---

## 4. Turn off the residual latent

```python
"use_residual_latent": False
```

`ProbabilisticArchetypeModel` can give each cell a private residual latent that absorbs whatever
the archetypes cannot explain. On CyTOF that is useful. On scRNA-seq, "whatever the archetypes
cannot explain" is largely dropout noise, and handing it a dedicated place to live defeats the
main thing protecting you.

That protection is worth naming, because it is CyEmbed's real advantage on sparse data: the
simplex bottleneck is aggressive. At K=6 you have five free dimensions. Per-gene dropout noise
*cannot be represented* in five dimensions, so it averages out across cells that share an
archetype. The model never attempts to describe gene-gene structure, which is exactly what dropout
corrupts worst. Give the residual latent room and you hand that noise a channel.

If you need it, keep `residual_dim` small and `beta_r` firm — but start with it off.

---

## 5. Two traps that fail silently

**`sample_col` swallows typos.** The AnnData branch guards with:

```python
if sample_col is not None and sample_col in adata.obs:
    resolved_samples = adata.obs[sample_col].to_numpy()
```

A misspelled column name yields `sample_ids=None` with no warning, no error, and no log line.
Downstream, that silently disables stratified splitting *and* balanced scaling — the model trains
fine and you never learn that your patient handling did nothing. Assert it landed:

```python
bundle = extract_matrix(adata=adata, source="X", sample_col="sample")
assert bundle.sample_ids is not None, "sample_col did not resolve — check adata.obs"
```

Same shape of failure for `cluster_col`.

**`marker_names` is ignored unless `source="obsm"`.** For `"X"` and `"layer"`, the argument is
accepted and then silently discarded in favour of `adata.var_names`. That is the behaviour you
want here, but it means passing `marker_names` gives you a false sense of control.

Also note: the `sample_ids` and `cluster_ids` *arguments* only apply to the DataFrame and ndarray
paths. On AnnData you must go through `sample_col` / `cluster_col`.

---

## 6. A working recipe

Adapted from `01_mcf7_probabilistic_archetype_embedding_sweep.ipynb` — the only sweep notebook
that uses the AnnData path — retuned for residuals. Departures from the CyTOF original are
flagged.

```python
import anndata as ad
from CyEmbed.data import extract_matrix, fit_scaler, preprocess_array, split_train_val_indices
from CyEmbed.train import build_sweep_configs, run_sweep
from CyEmbed.utils import collect_software_versions, set_seed

GLOBAL_CFG = {"seed": 7, "deterministic": True, "device": "auto"}

adata = ad.read_h5ad("../ProbAE_Deconv/data/PDX_02_sct_pearson_residuals_hvg.h5ad")
bundle = extract_matrix(adata=adata, source="X")

set_seed(GLOBAL_CFG["seed"], deterministic=GLOBAL_CFG["deterministic"])

scaler, _ = fit_scaler(bundle.X, mode="none")          # CHANGED: was "zscore"
X = preprocess_array(bundle.X, scaler)

train_idx, val_idx = split_train_val_indices(
    n_cells=X.shape[0], val_fraction=0.2, seed=GLOBAL_CFG["seed"],
    stratify_labels=None,                               # no labels in this h5ad
)

BASE_TRAIN_CFG = {
    "epochs": 1500,
    "early_stopping": True,
    "patience": 20,
    "min_delta": 0.0,
    "restore_best_weights": True,
    "weight_decay": 1e-5,
    "dropout": 0.0,
    "logit_normalizer": "entmax",
    "entmax_alpha": 1.5,
    "separation_mode": "cosine_sq",
    "balance_mode": "l2_uniform",
    "grad_clip_norm": 5.0,
    "print_every": 10,
    "progress_sweep": True,
    "progress_epoch": True,
    "skip_existing_runs": True,
    "beta_w": 1e-3,
    "beta_r": 1e-3,
    "kl_warmup_epochs": 10,
    "prob_eval_mode": "mean",
    "prob_eval_samples": 3,
}

SWEEP_GRID = {
    "model_type": ["probabilistic"],
    "use_residual_latent": [False],                      # CHANGED: keep off on sparse data
    "decoder_type": ["factorized"],
    "K": [4, 5, 6, 7, 8],
    "d": [8],
    "hidden_dims": [[256, 64]],
    "lr": [1e-3],
    "batch_size": [1024],
    "recon_loss_type": ["mse"],
    "lambda_entropy": [1e-3],
    "lambda_sep": [1e-3],
    "lambda_balance": [5e-2],
    "tau": [0.7, 1.0],
}

summary_df = run_sweep(
    x=X,
    marker_names=bundle.marker_names,
    cell_ids=bundle.cell_ids,
    output_root="outputs/pdx2_sct_archetype_sweep",
    base_config={**GLOBAL_CFG, **BASE_TRAIN_CFG},       # note the merge
    sweep_configs=build_sweep_configs(SWEEP_GRID),
    train_idx=train_idx,
    val_idx=val_idx,
    sample_ids=bundle.sample_ids,
    cluster_ids=bundle.cluster_ids,
    scaler_state=scaler.to_dict(),
)
```

Two non-obvious mechanics. `base_config` is the **merge** of `GLOBAL_CFG` and `BASE_TRAIN_CFG`,
not the training dict alone. And `skip_existing_runs` / `progress_sweep` are read off
`base_config` (`train.py:880-881`), not from `run_sweep` arguments — there is no keyword for them.

### On `decoder_type`: the gene embedding is not a compression

A natural question at 1,000 genes is whether to use the "gene embedding" or feed all the HVGs.
That is a false choice: **the encoder always takes all M genes**, whichever decoder you pick.
`decoder_type` only changes how the decoder reconstructs.

- `"factorized"` — `x̂ = w Z Eᵀ + b`, with `E ∈ R^{M×d}` the gene embedding (`model.py:134`)
- `"direct"` — `x̂ = w A`, with `A ∈ R^{K×M}` learned directly (`model.py:138`)

It is tempting to read the factorized form as a dimensionality reduction over genes. It isn't.
`Z @ E.T` is `(K,d) @ (d,M)`, so `A_hat` has rank ≤ min(K, d). At K=6 and d=8 that is rank ≤ 6 —
exactly what direct's `A` already spans. The parameter count runs the wrong way too:

| | parameters at M=1000, K=6, d=8 |
|---|---|
| factorized (`K·d + M·d + M`) | 48 + 8,000 + 1,000 = **9,048** |
| direct (`K·M`) | **6,000** |

Factorization only constrains anything — and only saves parameters — when `d < K`. At M=1000 and
K=6 that means `d ≤ 4`. Above that it is a reparameterisation, not a bottleneck.

**The real reason to prefer factorized here is `E` itself.** Each gene gets a d-dimensional
coordinate, so genes that behave alike across archetypes sit near each other:

```python
from CyEmbed.analysis import cosine_similarity_matrix, nearest_neighbors_from_similarity

gene_sim = cosine_similarity_matrix(run["E"])
modules = nearest_neighbors_from_similarity(gene_sim, run["marker_names"], k=8)
```

That is gene modules, free, as a by-product of fitting. At 40 CyTOF markers it is a curiosity; at
1,000 genes it is a genuine readout, and the direct decoder gives you nothing comparable. This is
why the recipe above sets `"factorized"` while the CyTOF notebooks use `"direct"`.

One asymmetry to know: **the direct decoder has no bias term at all** (`self.b = None`,
`model.py:120` and `:238`). At 40 CyTOF markers that costs little. At 2000 genes it costs a lot,
and this turns out to be the second — and stronger — argument for `factorized`.

`tools/verify_sample_offset_scrna.py` simulates the real thing (NB counts → analytic Pearson
residuals → 2000 HVGs by residual variance, 6 patients, 26% median gene detection) and measures
how well each decoder recovers known cell composition:

| decoder | w_recovery @ 20 markers | w_recovery @ 2000 HVGs |
|---|---|---|
| factorized | 0.993 | **0.818** |
| direct | 0.993 | **0.575** |

Identical at CyTOF scale; a wide gap at scRNA scale. It is *not* expressiveness — both `A_hat`s
are rank ≤ K. It is the intercept: with a per-patient `B` centred to zero-sum, `direct` has no
per-gene intercept anywhere, so all K archetype rows must redundantly encode a 2000-gene baseline
instead of spending their capacity on biology. `factorized`'s `b` absorbs it for free.

So on scRNA-seq you want `factorized` twice over: for the `E` gene-module readout, and because it
measurably recovers more biology.

### On `model_type`: probabilistic buys less than you'd think

The recipe above sweeps `probabilistic`, but be deliberate about why. On the scRNA-seq-scale
benchmark (`tools/verify_sample_offset_scrna.py`, 2000 HVGs, factorized decoder, measuring
recovery of known composition):

| | no regularisers | with this guide's regularisers |
|---|---|---|
| deterministic | 0.834 | **0.992** |
| probabilistic | 0.978 | 0.856 |

Read that as a 2×2, because the interaction is the whole story: **the KL and the explicit
regularisers are substitutes, not complements.** Without `lambda_*`, the probabilistic model's KL
does the regularising and helps a lot (0.834 → 0.978). With `lambda_*` already in place, adding
the KL over-regularises and *costs* you accuracy (0.992 → 0.856). The best configuration here is
**deterministic + the regularisers**.

So the reason to choose `probabilistic` is not accuracy. It is the one thing the deterministic
model cannot give you: a per-cell posterior (`mu_w`, `logvar_w`), so a shallow or ambiguous cell
can report that its composition is uncertain instead of returning a confident point estimate. That
is genuinely valuable on scRNA-seq, where depth varies enormously between cells.

But note that `prob_eval_mode="mean"` — the default, and what the real mcf7 analysis notebook
uses — takes the posterior mean and discards the uncertainty. If that is your workflow, you are
paying seven extra hyperparameters (`beta_w`, `beta_r`, `kl_warmup_epochs`, `logvar_min/max`,
`logvar_init_bias`, `prob_eval_*`) and some accuracy for something you then throw away. Use
`probabilistic` when you will act on `logvar_w`; otherwise prefer `deterministic`.

(Two seeds, one synthetic regime — evidence, not law. Sweep both on your own data.)

### On `recon_loss_type`

`"huber"` is available and would genuinely help: Pearson residuals have heavy tails, and Huber
downweights them. Use it if you are running CyEmbed on its own. **Do not use it if you are
comparing against mcRBM**, whose loss is hardcoded MSE (`../mcRBM/cytof_mcrbm/losses.py:44`) with
no robustness knob — the comparison would be unfair by construction.

---

## 7. Multi-patient work

**Your h5ads are single-sample.** `BCK_44_sct_pearson_residuals_hvg.h5ad` and
`PDX_02_sct_pearson_residuals_hvg.h5ad` each contain one sample; identity lives in the *filename*,
driven by `sample:` in `sct_gaussian_k_sweep.yaml`. `.obs` has only `cell_id`. Nothing in the
current pipeline concatenates them.

So before any of this applies, you need a combined object with a real sample column:

```python
import anndata as ad
a = ad.read_h5ad(".../BCK_44_sct_pearson_residuals_hvg.h5ad")
b = ad.read_h5ad(".../PDX_02_sct_pearson_residuals_hvg.h5ad")
a.obs["sample"] = "BCK_44"
b.obs["sample"] = "PDX_02"
combined = ad.concat([a, b], join="inner")   # inner join: HVG sets differ per sample
```

`join="inner"` matters — HVGs are selected per sample, so the gene sets are not identical. An
outer join would fabricate residuals for genes never modelled in one sample.

Then:

```python
bundle = extract_matrix(adata=combined, source="X", sample_col="sample")
assert bundle.sample_ids is not None

train_idx, val_idx = split_train_val_indices(
    n_cells=bundle.X.shape[0], val_fraction=0.2, seed=7,
    stratify_labels=bundle.sample_ids,     # every patient lands in BOTH halves
)
```

That stratification is important and CyEmbed gets it right: `split_train_val_indices` splits
*within* each group (`data.py:255-260`), so every patient contributes cells to train and val.
(mcRBM's equivalent does not — see its guide.)

### The per-patient offset

`use_sample_offset=True` gives the decoder a learned per-patient intercept `B ∈ R^{S×M}`, so
`x̂ = wZEᵀ + b + B[s]`. Archetypes then model deviation from *that patient's own baseline*, and
additive patient effects cannot leak into the weights because the decoder explains them for free.

```python
SWEEP_GRID = {..., "use_sample_offset": [True]}
```

Verified for **both** decoders on planted-shift synthetic data (`tools/verify_sample_offset.py`,
8 patients × 3 seeds, paired arms). With the offset on, cross-patient spread collapses
(0.21 → 0.02 factorized, 0.28 → 0.02 direct), `B` recovers the planted shift at corr 1.000, and —
the check that matters — `w_recovery` rises from ~0.15 to ~0.99, meaning `W` *keeps the biology*
rather than merely losing its patient structure. So you can use `factorized` and get both gene
modules and patient correction.

Three things to know:

- It removes **additive** shifts only, not multiplicative or interaction effects.
- It is a **soft** fix. It removes the *incentive* for the weights to encode patient identity; it
  does not forbid it. The encoder still sees raw expression — deliberately, so you can still
  project an unseen patient in one forward pass (they decode at `B = 0`, the average baseline).
- `A_hat` does **not** include `B`. It remains the archetype profile at the average patient.
  Inspect `B.npy` separately.

**Do not reach for it reflexively.** In PDX and patient tumours, inter-patient variation
substantially *is* the biology — different tumours, different genotypes. Shared archetypes with
patient-varying composition is the *goal*, not the failure. Train with the offset off first and
look at what you get. The failure worth correcting is narrow and specific: an archetype with
`dominant_fraction ≈ 1.0` in one patient and ≈ 0 elsewhere. Graded usage differences across
patients are your result, and correcting them deletes it.

If patient and batch are perfectly confounded (each patient its own run), nothing separates them
and the offset removes both.

---

## 8. Reading the output

Residual space changes what the archetypes mean.

`A_hat` is the archetype profile in feature space (`Z @ E.T + b` for the factorized decoder,
`model.py:135`). On CyTOF you read it as expression intensity. On Pearson residuals, read it as
**enrichment relative to depth expectation**: a positive loading means "this gene appears more
than sequencing depth alone predicts in cells using this archetype."

Negative loadings are therefore meaningful, not a modelling artifact — they are genuine depletion.
This is the one place where CyEmbed's signed loadings are an advantage over NMF, whose
non-negativity cannot express depletion at all.

```python
from CyEmbed.analysis import load_run_outputs, archetype_marker_rankings, summarize_by_group

run = load_run_outputs(best_run_dir)
W = run.get("W_mean", run.get("W"))
ranking = archetype_marker_rankings(run["A_hat"], marker_names=run["marker_names"], top_n=12)
```

`load_run_outputs` (`analysis.py:12`) loads every `*.npy` in the run directory keyed by filename
stem, so `W`, `A_hat`, `mu_w`, `Z`, `E`, `b` all appear depending on model type. Prefer `W_mean`
over `W` for probabilistic runs — it is the posterior mean and is more stable than a single
sample.

---

## 9. Validating without labels

`label_column: null` throughout your pipeline, and `.obs` carries only `cell_id`. There are no
cell-type annotations to score against, so ARI/NMI against ground truth is not available.

The established alternative here is **UCell signature correlation**, from notebook 301's Figure 12:
join signature scores on `cell_id`, then Spearman-correlate each signature against each archetype
weight.

```python
import pandas as pd
from scipy.stats import spearmanr

ucell = pd.read_csv(".../PDX_02.RNA.UCell.Tumor.csv")
if "barcode" in ucell.columns:
    ucell = ucell.rename(columns={"barcode": "cell_id"})
ucell["cell_id"] = ucell["cell_id"].astype(str)

weights = pd.DataFrame(W, columns=[f"w_{k}" for k in range(W.shape[1])])
weights["cell_id"] = run["cell_ids"]
merged = weights.merge(ucell, on="cell_id", how="inner")
```

`cell_id` is the sole join key to any external annotation, carried in-band precisely so labels can
stay out of the modelling path. Upstream cluster annotations do exist (e.g.
`BCK_44.RNA.leiden_sct_94.csv`, `BCK_44.cluster_annotations_auto_from_hallmark.csv`) and can be
joined the same way if you want a harder check.

---

## 10. Diagnostics you should run every time

**Depth leakage.** The single cheapest and most informative check. If archetype weights correlate
with sequencing depth, your preprocessing failed and everything downstream is suspect:

```python
depth = np.asarray(adata.X.sum(axis=1)).ravel()   # or n_genes detected
for k in range(W.shape[1]):
    r, _ = spearmanr(W[:, k], depth)
    print(f"archetype {k}: rho vs depth = {r:+.3f}")
```

Note this needs a depth covariate, which the SCT residual h5ad does not carry — `.obs` has only
`cell_id`. Compute it from the NB-route h5ad (`bck44_scrna_hvg_counts.h5ad` has `total_counts`)
and join on `cell_id`.

**Patient leakage.**

```python
stats = summarize_by_group(W, sample_ids, group_name="sample")
# stats["dominant_fractions"] -> look for any archetype ~1.0 in one patient, ~0 elsewhere
```

**Stability.** Your existing sweep runs `seeds: [42]` — one seed. Archetypal autoencoders are
non-convex and seed-sensitive, and `val_recon` measures fit, not reproducibility. Run several
seeds at your chosen K and check archetype agreement across restarts (cosine similarity between
`A_hat` matrices, matched greedily). This is the one methodological gap where cNMF's consensus
procedure has an answer and CyEmbed currently does not.

---

## 11. The alternative worth knowing about

Pearson residuals are a workaround: you transform the data until MSE stops being wrong. The more
principled route models counts directly with a negative-binomial likelihood — no transform at all.

**You already have this.** `../ProbAE_Deconv/configs/bck44_scrna_nb_k_sweep.yaml` runs an
archetypal autoencoder with `decoder_family: "nb"`, `loss: {type: nb_nll}`,
`data: {encoder_input: log1p_normalized, decoder_target: raw_counts}`, and
`use_observed_library_size: true`. It reads a counts h5ad (`bck44_scrna_hvg_counts.h5ad`,
sparse CSR, with a full scanpy QC block: mito/ribo caps, doublet scoring).

If you are choosing between them: NB-on-counts is more defensible and a reviewer will prefer it.
Residuals-plus-CyEmbed is what lets you use *this* package, and is required if you want a
controlled comparison against mcRBM (which cannot do NB). Running both and comparing is the honest
move — if NB clearly wins, the transform is costing you signal.

---

## 12. A note on scale

PDX_02 is 5,774 cells × 1,000 genes. BCK_44 is **330 cells** × 1,000 genes.

At 330 cells, an encoder of `[256, 64]` over 1,000 inputs has ~272,000 parameters fit to 330
observations. The simplex bottleneck is the only reason this is not hopeless — it forces every
cell onto a K−1 dimensional object regardless of how expressive the encoder is, and that
constraint, not the data, is what regularises the fit. It works, but treat BCK_44 results as
provisional and prefer PDX_02 for anything you intend to defend.

This matters much more for mcRBM, which estimates second-order structure and needs
cells ≫ genes. See its guide.
