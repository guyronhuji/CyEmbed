"""Standalone verification of the per-patient decoder intercept (B) in CyEmbed.

CyEmbed is notebook-first (no pytest), so this is a runnable script rather than a test module.
It plants a known constant per-patient shift on top of cells with a *shared* archetype
composition and checks the behaviour that motivated the feature:

    use_sample_offset=False -> the additive shift has nowhere to go but W, so archetype
      composition looks different across patients even though the ground-truth composition is
      identical (the identity-archetype pathology: high cross-patient spread, patient-specific
      dominant archetypes).
    use_sample_offset=True  -> B absorbs the additive shift, so W is free to recover the shared
      composition and B recovers the planted shift up to centring.

Both decoders are held to the same bar. An earlier version of this script gated the decisive
assertions to decoder_type="direct" and justified it by claiming the simplex constraint on w makes
per-patient shifts expensive to represent through w for direct but not for factorized. That
rationale is false: h = w @ Z uses the same simplex w as w @ A, and both reachable sets are convex
hulls of K points, so identity archetypes are equally available to both. The real asymmetry is in
optimisation dynamics -- the factorized Z@E.T product parametrisation outruns B's linear growth --
which is a thing to fix, not to encode as expected behaviour.

Design notes, each earning its keep:
  * off/on run on IDENTICAL data (seeded per case), so the comparison is paired.
  * w_recovery is the load-bearing metric. Measuring only "patient structure absent from W" is
    insufficient: a W that has collapsed to near-constant has no patient structure at all and
    scores a perfect spread=0/modal=1 while having learned nothing. w_recovery separates the two.
  * val_recon is reported. Adding a parameter that can always be zero must not make the fit worse;
    if it does, that is a real signal.
  * Multiple seeds, because spread is a std across patients and a single draw is not evidence.

Run:  python tools/verify_sample_offset.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CyEmbed.analysis import summarize_by_group
from CyEmbed.data import split_train_val_indices
from CyEmbed.train import _config_fingerprint, train_one_run
from CyEmbed.utils import validate_run_config

K_TRUE = 3
N_MARKERS = 20
N_PER_PATIENT = 300
N_PATIENTS = 8
# Signal-comparable, not signal-swamping: archetype rows are N(0, 3^2), so a shift of sd 2.0 is a
# real confounder without making the dataset almost entirely patient offset.
SHIFT_SCALE = 2.0
SEEDS = (0, 1, 2)
# Shared composition skewed toward archetype 0, so the true dominant archetype is the same for
# every patient -- any per-patient difference in the recovered dominant archetype is leakage.
ALPHA = np.array([6.0, 3.0, 3.0])
PATIENT_NAMES = np.array([f"patient_{p}" for p in range(N_PATIENTS)])


def make_synthetic(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cells from K_TRUE archetypes with an IDENTICAL composition per patient, plus a known
    constant per-patient shift.

    Returns (x, sample_ids, planted_shift[N_PATIENTS, M], w_true[N, K_TRUE]).

    Seeded per call so that the off/on arms of a case see the same dataset -- comparing spread
    across two different random datasets, as an earlier version did, measures nothing.
    """
    rng = np.random.default_rng(seed)
    archetypes = rng.normal(0.0, 1.0, size=(K_TRUE, N_MARKERS)) * 3.0
    shift = rng.normal(0.0, SHIFT_SCALE, size=(N_PATIENTS, N_MARKERS))
    x_blocks, sample_blocks, w_blocks = [], [], []
    for p in range(N_PATIENTS):
        w = rng.dirichlet(ALPHA, size=N_PER_PATIENT)
        clean = w @ archetypes
        noise = rng.normal(0.0, 0.05, size=clean.shape)
        x_blocks.append(clean + shift[p] + noise)
        sample_blocks.append(np.full(N_PER_PATIENT, PATIENT_NAMES[p]))
        w_blocks.append(w)
    x = np.concatenate(x_blocks, axis=0).astype(np.float32)
    sample_ids = np.concatenate(sample_blocks, axis=0)
    w_true = np.concatenate(w_blocks, axis=0)
    return x, sample_ids, shift, w_true


def _mean_weight_table(w: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    summary = summarize_by_group(w, sample_ids)["mean_weights"]
    cols = [c for c in summary.columns if c.startswith("archetype_")]
    return summary[cols].to_numpy()


def cross_patient_spread(w: np.ndarray, sample_ids: np.ndarray) -> float:
    """Max over archetypes of the std across patients of that archetype's mean weight.
    Ground-truth composition is identical across patients, so a faithful model gives ~0."""
    return float(_mean_weight_table(w, sample_ids).std(axis=0, ddof=0).max())


def modal_distinct(w: np.ndarray, sample_ids: np.ndarray) -> int:
    """Number of distinct per-patient dominant archetypes. 1 == fully shared, N == identity."""
    modal = _mean_weight_table(w, sample_ids).argmax(axis=1)
    return len(set(modal.tolist()))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, returning 0.0 for a degenerate (constant) input rather than nan.

    The degenerate case is exactly the collapse we are hunting: a constant W column carries no
    information, and 0.0 is the honest score for it.
    """
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(r) else r


def w_recovery(w: np.ndarray, w_true: np.ndarray) -> float:
    """Mean correlation between recovered and true archetype weights, after optimally matching
    archetype identity (which is arbitrary up to permutation).

    This is the check that distinguishes 'B absorbed the shift and W kept the biology' from
    'W collapsed to a constant'. Both score spread~0; only the former scores high here.

    Normalised by max(k_rec, k_true), NOT by the number of matched pairs. linear_sum_assignment
    returns only min(k_rec, k_true) pairs, so averaging over those scores a model on the
    archetypes it happened to find and silently forgives the ones it missed -- at k_rec=3 vs
    k_true=5 that let three good matches tie a complete recovery. Dividing by the larger dimension
    charges for missed and spurious archetypes alike, which is what makes this usable to select K
    rather than only to compare at fixed K. When k_rec == k_true the two are identical, so this
    does not change any same-K result.
    """
    k_rec, k_true = w.shape[1], w_true.shape[1]
    corr = np.zeros((k_rec, k_true))
    for i in range(k_rec):
        for j in range(k_true):
            corr[i, j] = _safe_corr(w[:, i], w_true[:, j])
    rows, cols = linear_sum_assignment(-corr)
    return float(corr[rows, cols].sum() / max(k_rec, k_true))


def run_case(
    model_type: str,
    decoder_type: str,
    use_offset: bool,
    out_root: Path,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> dict:
    x, sample_ids, shift, w_true = data
    n = x.shape[0]
    # Stratify within patient so every patient appears in both halves (required once B is on).
    train_idx, val_idx = split_train_val_indices(
        n, val_fraction=0.2, seed=0, stratify_labels=sample_ids
    )
    run_id = f"{model_type}_{decoder_type}_{'on' if use_offset else 'off'}_s{seed}"
    res = train_one_run(
        x_train=x[train_idx],
        x_val=x[val_idx],
        x_full=x,
        marker_names=[f"m{i}" for i in range(N_MARKERS)],
        cell_ids=[f"c{i}" for i in range(n)],
        run_config=base_config(model_type, decoder_type, use_offset),
        output_root=out_root,
        run_id=run_id,
        sample_ids=sample_ids,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    run_dir = out_root / run_id
    w = np.load(run_dir / "W.npy")

    # train_one_run returns the full per-epoch history in memory, so persistence is unnecessary.
    hist = res.history
    val_curve = hist["val_recon"].to_numpy()
    result = {
        "spread": cross_patient_spread(w, sample_ids),
        "modal": modal_distinct(w, sample_ids),
        "w_rec": w_recovery(w, w_true),
        "val_recon": float(val_curve[-1]),
        "val_best": float(val_curve.min()),
        # Still descending at the end => the run was truncated, not converged.
        "still_descending": bool(val_curve[-1] < val_curve[max(0, len(val_curve) - 20)] - 1e-4),
        "b_corr": None,
        "b_zero_sum": None,
    }
    if use_offset:
        B = np.load(run_dir / "B.npy")  # saved already centred
        assert (run_dir / "sample_offset_levels.csv").exists(), "levels csv missing"
        levels = pd.read_csv(run_dir / "sample_offset_levels.csv")["sample_id"].to_numpy()
        order = [int(np.where(PATIENT_NAMES == lv)[0][0]) for lv in levels]
        shift_centered = shift[order] - shift[order].mean(0, keepdims=True)
        result["b_corr"] = _safe_corr(B.ravel(), shift_centered.ravel())
        result["b_zero_sum"] = float(np.abs(B.mean(0)).max())
    return result


def base_config(model_type: str, decoder_type: str, use_offset: bool) -> dict:
    cfg = {
        "model_type": model_type,
        "decoder_type": decoder_type,
        "K": K_TRUE,
        "d": 8,
        "hidden_dims": [64, 32],
        "tau": 1.0,
        "lr": 5e-3,
        # Generous cap + early stopping, so every run is compared at convergence rather than at an
        # arbitrary truncation point. At a fixed 150-300 epochs these runs are still descending,
        # which made earlier val_recon comparisons meaningless.
        "epochs": 3000,
        "batch_size": 256,
        "recon_loss_type": "mse",
        "weight_decay": 1e-4,
        "seed": 0,
        # CPU, deliberately. These tensors are tiny (20 features, K=3, batch 256), so MPS
        # kernel-launch overhead swamps the arithmetic: measured 2.0s on one CPU thread vs 23.3s
        # on MPS for an identical 300-epoch run -- an 11.6x speedup by NOT using the GPU.
        "device": "cpu",
        "early_stopping": True,
        "patience": 60,
        "min_delta": 1e-5,
        "progress_epoch": False,
        "print_every": 100000,
        "deterministic": False,
    }
    if use_offset:
        cfg["use_sample_offset"] = True
    return cfg


def _agg(runs: list[dict], key: str) -> tuple[float, float]:
    vals = np.array([r[key] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def main() -> None:
    failures: list[str] = []

    # --- Regression: fingerprint unchanged when the offset is off, changed when on. ---
    cfg_plain = {
        "model_type": "deterministic", "decoder_type": "factorized", "K": 3, "d": 8,
        "hidden_dims": [64, 32], "tau": 1.0, "lr": 5e-3, "epochs": 10, "batch_size": 256,
        "recon_loss_type": "mse",
    }
    fp_ref = _config_fingerprint(validate_run_config(dict(cfg_plain)))
    fp_off = _config_fingerprint(validate_run_config({**cfg_plain, "use_sample_offset": False}))
    fp_on = _config_fingerprint(validate_run_config({**cfg_plain, "use_sample_offset": True}))
    if fp_ref != fp_off:
        failures.append(f"fingerprint changed with offset explicitly off: {fp_ref} != {fp_off}")
    if fp_on == fp_ref:
        failures.append("fingerprint did NOT change when offset turned on (should differ)")
    print(f"[fingerprint] off={fp_ref}  explicit-off={fp_off}  on={fp_on}")

    # --- Behavioural: planted-shift recovery across decoder/model types. ---
    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)
        print(
            f"\n{N_PATIENTS} patients x {N_PER_PATIENT} cells, {N_MARKERS} markers, "
            f"shift sd={SHIFT_SCALE}, seeds={list(SEEDS)}\n"
        )
        header = (
            f"{'case':<26}{'spread_off':>11}{'spread_on':>11}"
            f"{'wrec_off':>10}{'wrec_on':>9}{'modal_on':>9}"
            f"{'vrec_off':>10}{'vrec_on':>10}{'B~shift':>9}"
        )
        print(header)
        print("-" * len(header))
        for model_type in ("deterministic", "probabilistic"):
            for decoder_type in ("factorized", "direct"):
                offs, ons = [], []
                for seed in SEEDS:
                    data = make_synthetic(seed)  # SAME data for both arms
                    offs.append(run_case(model_type, decoder_type, False, out_root, data, seed))
                    ons.append(run_case(model_type, decoder_type, True, out_root, data, seed))

                tag = f"{model_type}/{decoder_type}"
                spread_off, spread_off_sd = _agg(offs, "spread")
                spread_on, spread_on_sd = _agg(ons, "spread")
                wrec_off, _ = _agg(offs, "w_rec")
                wrec_on, _ = _agg(ons, "w_rec")
                modal_on, _ = _agg(ons, "modal")
                vrec_off, _ = _agg(offs, "val_recon")
                vrec_on, _ = _agg(ons, "val_recon")
                bcorr, _ = _agg(ons, "b_corr")

                print(
                    f"{tag:<26}{spread_off:>11.4f}{spread_on:>11.4f}"
                    f"{wrec_off:>10.3f}{wrec_on:>9.3f}{modal_on:>9.2f}"
                    f"{vrec_off:>10.3f}{vrec_on:>10.3f}{bcorr:>9.3f}"
                )

                # Same bar for every decoder. No gating.
                if not (np.mean([r["b_zero_sum"] for r in ons]) < 1e-4):
                    failures.append(f"{tag}: saved B not zero-sum")
                if not (bcorr > 0.9):
                    failures.append(f"{tag}: B recovers shift poorly (corr={bcorr:.3f})")
                if not (spread_on < 0.5 * spread_off):
                    failures.append(
                        f"{tag}: offset did not collapse cross-patient spread "
                        f"({spread_on:.4f} vs {spread_off:.4f})"
                    )
                if not (modal_on < 1.5):
                    failures.append(
                        f"{tag}: offset did not unify dominant archetype (modal={modal_on:.2f})"
                    )
                # The check that catches a collapsed W masquerading as a clean one.
                if not (wrec_on > 0.7):
                    failures.append(
                        f"{tag}: W lost the biology with offset on (w_recovery={wrec_on:.3f}) "
                        f"-- low spread here is collapse, not correction"
                    )
                # Adding a parameter that can always be zero must not hurt the fit.
                if not (vrec_on <= vrec_off * 1.1 + 1e-3):
                    failures.append(
                        f"{tag}: offset made val_recon worse ({vrec_on:.3f} vs {vrec_off:.3f})"
                    )
                truncated = [r for r in ons if r["still_descending"]]
                if truncated:
                    print(
                        f"  note: {len(truncated)}/{len(ons)} 'on' runs still descending at the "
                        f"final epoch -- results may be truncation, not convergence"
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
