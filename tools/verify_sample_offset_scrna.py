"""scRNA-seq-scale verification of the per-patient decoder intercept (B).

`verify_sample_offset.py` is a CyTOF-shaped unit check: 20 dense, well-behaved Gaussian features.
Real scRNA-seq is a different regime and this script mimics it:

  * 2000 HVGs instead of 20 markers -- 100x the features, so B is (S, 2000) and E is (2000, d).
  * Sparse counts, not Gaussians. Genes are mostly zero, so Pearson residuals are variance-
    stabilised but decidedly NOT Gaussian: for a gene detected in a few percent of cells the
    residual marginal is a spike (all the zeros, at a mild negative value) plus a sparse right
    tail. That is the thing the toy test cannot exercise, and the thing most likely to break a
    method that assumes Gaussian visibles.
  * HVG selection by residual variance, matching the real pipeline
    (ProbAE_Deconv/notebooks/experiment_suite/300_scrna_sct_gaussian_k_sweep.ipynb selects by
    SCTransform's own residual_variance). This is not a neutral step: variance ranking favours
    genes with a few extreme residuals, i.e. the most tail-dominated ones.

Generative model (standard scRNA-seq simulation, mixture in EXPRESSION space not log space):
    p_k     ~ archetype gene-expression profile over genes (sums to 1)
    w_i     ~ Dirichlet(ALPHA)                     -- shared composition across patients
    lib_i   ~ lognormal                            -- sequencing depth
    mu_ig   = lib_i * (w_i @ p)_g
    x_ig    ~ NegBinomial(mu_ig, THETA)

then analytic Pearson residuals against the depth-aware null (Lause/Berens/Kobak 2021, the same
thing scanpy's experimental.pp.normalize_pearson_residuals computes):
    mu_hat_ig = row_sum_i * col_sum_g / total
    z_ig      = (x_ig - mu_hat_ig) / sqrt(mu_hat_ig + mu_hat_ig^2 / THETA)

The planted per-patient shift is added to z, i.e. in RESIDUAL space. That is deliberate: it is the
space B actually operates in, so "did B recover the shift" stays well-posed. A shift planted in
count or log space would not be additive after the residual transform and b_corr would be
meaningless.

Metrics are imported from verify_sample_offset rather than reimplemented.

Run:  python tools/verify_sample_offset_scrna.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CyEmbed.data import split_train_val_indices
from CyEmbed.train import train_one_run

# Reuse the metric definitions rather than duplicating them.
_spec = importlib.util.spec_from_file_location("_vso", Path(__file__).with_name("verify_sample_offset.py"))
_vso = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vso)
cross_patient_spread = _vso.cross_patient_spread
modal_distinct = _vso.modal_distinct
w_recovery = _vso.w_recovery
_safe_corr = _vso._safe_corr

K_TRUE = 5
N_GENES_TOTAL = 5000       # before HVG selection
N_HVG = 2000               # the number that matters
N_PATIENTS = 6
N_PER_PATIENT = 800        # 4800 cells total, ~PDX_02 scale (5774)
N_MARKER_GENES = 200       # per-archetype marker genes -- what makes programs distinguishable
MARKER_FOLD = 20.0         # fold-enrichment of an archetype's markers over baseline
THETA = 10.0               # NB dispersion
LIB_MEAN_LOG = np.log(5000.0)
LIB_SD_LOG = 0.35
SHIFT_SCALE = 1.0          # residuals are ~unit variance, so 1.0 sd is a substantial batch effect
SEEDS = (0, 1)
# A SPREAD Dirichlet, unlike the toy test's concentrated [6,3,3]. This matters more than any other
# constant here. Concentrated alphas make every cell's composition nearly identical
# (sd(w_0) ~ 0.12), so there is almost nothing for the model to recover -- fine for the toy test,
# which has no counting noise, but under Poisson shot noise it leaves biology at ~10% of the
# residual variance and w_recovery becomes vacuous. Spreading the alphas raises sd(w_0) to ~0.26
# and biology to ~34% of variance, at a realistic 26% median gene detection. The MEAN composition
# is still shared across patients, which is what keeps spread/modal_distinct meaningful.
ALPHA = np.array([0.6, 0.4, 0.4, 0.3, 0.3])
PATIENT_NAMES = np.array([f"patient_{p}" for p in range(N_PATIENTS)])


def _pearson_residuals(counts: np.ndarray, theta: float) -> np.ndarray:
    """Analytic Pearson residuals against the depth-aware independence null."""
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    total = counts.sum()
    mu = row @ col / total
    mu = np.maximum(mu, 1e-8)
    z = (counts - mu) / np.sqrt(mu + mu**2 / theta)
    # scanpy clips at sqrt(n_cells); keep the same convention.
    clip = np.sqrt(counts.shape[0])
    return np.clip(z, -clip, clip)


def make_scrna(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (z_hvg, sample_ids, shift[N_PATIENTS, N_HVG], w_true, stats)."""
    rng = np.random.default_rng(seed)

    # Archetype gene profiles. A shared, heavy-tailed baseline gives realistic detection rates
    # (most genes rare). Distinctness comes from PROGRAM MARKER GENES, not from a diffuse
    # per-gene factor: with only a shared base times lognormal(0,1) noise the profiles are
    # near-identical, w barely moves mu, and there is no recoverable biology at all -- the model
    # then learns the patient offsets and nothing else (measured: w_recovery 0.08, val_recon 1.09
    # against a data variance of 1.09). Real cell types differ by marker genes; so do these.
    base = rng.lognormal(mean=0.0, sigma=2.0, size=N_GENES_TOTAL)
    profiles = np.zeros((K_TRUE, N_GENES_TOTAL))
    for k in range(K_TRUE):
        p = base * rng.lognormal(mean=0.0, sigma=0.3, size=N_GENES_TOTAL)
        markers = rng.choice(N_GENES_TOTAL, size=N_MARKER_GENES, replace=False)
        p[markers] *= MARKER_FOLD
        profiles[k] = p / p.sum()

    w_true = rng.dirichlet(ALPHA, size=N_PATIENTS * N_PER_PATIENT)  # shared composition
    lib = rng.lognormal(LIB_MEAN_LOG, LIB_SD_LOG, size=w_true.shape[0])[:, None]
    mu = lib * (w_true @ profiles)
    # NB via gamma-Poisson.
    rate = rng.gamma(shape=THETA, scale=mu / THETA)
    counts = rng.poisson(rate).astype(np.float64)

    sample_ids = np.concatenate(
        [np.full(N_PER_PATIENT, PATIENT_NAMES[p]) for p in range(N_PATIENTS)]
    )

    z_all = _pearson_residuals(counts, THETA)

    # HVG selection by residual variance -- the real pipeline's criterion.
    rv = z_all.var(axis=0)
    hvg_idx = np.sort(np.argsort(rv)[::-1][:N_HVG])
    z = z_all[:, hvg_idx]

    detection = (counts[:, hvg_idx] > 0).mean(axis=0)
    stats = {
        "median_detection": float(np.median(detection)),
        "frac_genes_under_10pct": float((detection < 0.10).mean()),
        "zero_frac_all": float((counts == 0).mean()),
        "median_lib": float(np.median(counts.sum(axis=1))),
        "resid_sd": float(z.std()),
    }

    shift = rng.normal(0.0, SHIFT_SCALE, size=(N_PATIENTS, N_HVG))
    for p in range(N_PATIENTS):
        z[sample_ids == PATIENT_NAMES[p]] += shift[p]

    return z.astype(np.float32), sample_ids, shift, w_true, stats


def base_config(model_type: str, decoder_type: str, use_offset: bool, variant: str = "plain") -> dict:
    """variant="plain": softmax, no regularisers -- isolates the offset with fewest confounds.
    variant="guide": the settings SCRNA_SEQ_GUIDE.md actually recommends and the real mcf7
    notebook uses. Worth testing separately because lambda_sep penalises archetypes for being
    SIMILAR, and identity archetypes are maximally separated -- so the separation penalty
    plausibly rewards the very pathology the offset exists to remove. lambda_entropy and
    lambda_balance should push the other way (both favour diffuse w, which is what routing the
    shift through B produces), so the net effect is an empirical question, not an obvious one.
    """
    cfg = {
        "model_type": model_type,
        "decoder_type": decoder_type,
        "K": K_TRUE,
        "d": 16,
        "hidden_dims": [256, 64],   # matches the real sct_gaussian_k_sweep.yaml encoder
        "tau": 1.0,
        "lr": 1e-3,
        "epochs": 400,
        "batch_size": 512,
        "recon_loss_type": "mse",
        "weight_decay": 1e-4,
        "seed": 0,
        "device": "cpu",
        "early_stopping": True,
        "patience": 30,
        "min_delta": 1e-4,
        "progress_epoch": False,
        "print_every": 100000,
        "deterministic": False,
    }
    if variant == "guide":
        cfg.update({
            "logit_normalizer": "entmax",
            "entmax_alpha": 1.5,
            "lambda_entropy": 1e-3,
            "lambda_sep": 1e-3,
            "lambda_balance": 5e-2,
            "separation_mode": "cosine_sq",
            "balance_mode": "l2_uniform",
        })
    if model_type == "probabilistic":
        cfg.update({
            "use_residual_latent": False,   # per the guide: it absorbs dropout noise on sparse data
            "beta_w": 1e-3,
            "beta_r": 1e-3,
            "kl_warmup_epochs": 10,
            "prob_eval_mode": "mean",       # what the real mcf7 notebook uses
            "prob_eval_samples": 3,
        })
    if use_offset:
        cfg["use_sample_offset"] = True
    return cfg


def run_case(decoder_type, use_offset, out_root, data, seed, model_type="deterministic",
             variant="plain") -> dict:
    z, sample_ids, shift, w_true, _ = data
    n = z.shape[0]
    train_idx, val_idx = split_train_val_indices(
        n, val_fraction=0.2, seed=0, stratify_labels=sample_ids
    )
    run_id = f"{model_type}_{decoder_type}_{variant}_{'on' if use_offset else 'off'}_s{seed}"
    res = train_one_run(
        x_train=z[train_idx], x_val=z[val_idx], x_full=z,
        marker_names=[f"g{i}" for i in range(z.shape[1])],
        cell_ids=[f"c{i}" for i in range(n)],
        run_config=base_config(model_type, decoder_type, use_offset, variant),
        output_root=out_root, run_id=run_id, sample_ids=sample_ids,
        train_idx=train_idx, val_idx=val_idx,
    )
    run_dir = out_root / run_id
    # For probabilistic runs prefer the posterior mean over a single sample -- it is what
    # prob_eval_mode="mean" produces and what the real mcf7 analysis notebook reads.
    w_mean_path = run_dir / "W_mean.npy"
    w = np.load(w_mean_path if w_mean_path.exists() else run_dir / "W.npy")
    val_curve = res.history["val_recon"].to_numpy()
    out = {
        "spread": cross_patient_spread(w, sample_ids),
        "modal": modal_distinct(w, sample_ids),
        "w_rec": w_recovery(w, w_true),
        "val_recon": float(val_curve[-1]),
        "b_corr": None,
    }
    if use_offset:
        B = np.load(run_dir / "B.npy")
        levels = pd.read_csv(run_dir / "sample_offset_levels.csv")["sample_id"].to_numpy()
        order = [int(np.where(PATIENT_NAMES == lv)[0][0]) for lv in levels]
        sc = shift[order] - shift[order].mean(0, keepdims=True)
        out["b_corr"] = _safe_corr(B.ravel(), sc.ravel())
    return out


def main() -> None:
    failures: list[str] = []
    print(
        f"scRNA-seq-scale: {N_PATIENTS} patients x {N_PER_PATIENT} cells, "
        f"{N_GENES_TOTAL} genes -> {N_HVG} HVGs, NB(theta={THETA}), shift sd={SHIFT_SCALE}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)
        header = (
            f"{'case':<26}{'spread_off':>11}{'spread_on':>11}"
            f"{'wrec_off':>10}{'wrec_on':>9}{'modal_on':>9}"
            f"{'vrec_off':>10}{'vrec_on':>10}{'B~shift':>9}"
        )
        # (decoder, model_type, variant). Answers three questions in one matrix:
        #   factorized vs direct        -- decoder choice at 2000 genes
        #   plain vs guide              -- do the guide's regularisers (esp. lambda_sep) break it?
        #   deterministic vs probabilistic -- what the KL/posterior buys, read via W_mean
        CASES = [
            ("factorized", "deterministic", "plain"),
            ("direct", "deterministic", "plain"),
            ("factorized", "deterministic", "guide"),
            ("factorized", "probabilistic", "plain"),
            ("factorized", "probabilistic", "guide"),
        ]
        first = True
        for decoder_type, model_type, variant in CASES:
            offs, ons = [], []
            for seed in SEEDS:
                data = make_scrna(seed)
                if first and seed == SEEDS[0]:
                    s = data[4]
                    print(
                        f"\n[data realism] median gene detection {s['median_detection']:.1%} | "
                        f"{s['frac_genes_under_10pct']:.0%} of HVGs detected in <10% of cells | "
                        f"overall zeros {s['zero_frac_all']:.1%} | median library "
                        f"{s['median_lib']:.0f} | residual sd {s['resid_sd']:.2f}"
                    )
                    print("\n" + header)
                    print("-" * len(header))
                    first = False
                offs.append(run_case(decoder_type, False, out_root, data, seed, model_type, variant))
                ons.append(run_case(decoder_type, True, out_root, data, seed, model_type, variant))

            m = lambda rs, k: float(np.mean([r[k] for r in rs]))
            tag = f"{model_type[:4]}/{decoder_type}/{variant}"
            print(
                f"{tag:<26}{m(offs,'spread'):>11.4f}{m(ons,'spread'):>11.4f}"
                f"{m(offs,'w_rec'):>10.3f}{m(ons,'w_rec'):>9.3f}{m(ons,'modal'):>9.2f}"
                f"{m(offs,'val_recon'):>10.3f}{m(ons,'val_recon'):>10.3f}{m(ons,'b_corr'):>9.3f}"
            )

            # What this script tests is THE OFFSET, not decoder quality. Those are separate
            # questions and conflating them is how the original harness went wrong.
            #
            # Offset assertions (hard): B must recover the shift, spread must collapse, and the
            # dominant archetype must unify.
            if not (m(ons, "b_corr") > 0.9):
                failures.append(f"{tag}: B recovers shift poorly (corr={m(ons,'b_corr'):.3f})")
            if not (m(ons, "spread") < 0.5 * m(offs, "spread")):
                failures.append(
                    f"{tag}: offset did not collapse spread "
                    f"({m(ons,'spread'):.4f} vs {m(offs,'spread'):.4f})"
                )
            if not (m(ons, "modal") < 1.5):
                failures.append(f"{tag}: offset did not unify dominant archetype")

            # Biology assertion: the offset's PURPOSE is to free W to recover composition, so the
            # test is that it substantially improves recovery -- not that recovery clears an
            # absolute bar. verify_sample_offset.py can demand >0.7 absolute because its data is
            # noiseless; here ~2/3 of the residual variance is Poisson shot noise by construction,
            # so the achievable ceiling is decoder- and depth-dependent. The absolute level is a
            # decoder characterisation (reported above), not evidence about the offset.
            if not (m(ons, "w_rec") > 5.0 * m(offs, "w_rec") and m(ons, "w_rec") > 0.4):
                failures.append(
                    f"{tag}: offset did not free W to recover composition "
                    f"(w_recovery {m(offs,'w_rec'):.3f} -> {m(ons,'w_rec'):.3f})"
                )

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
