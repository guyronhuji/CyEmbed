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

Decoder asymmetry (a real, expected result, not a bug):
    * direct decoder  (x_hat = w @ A + B): the simplex constraint on w makes representing
      arbitrary per-patient shifts through w expensive, so B is strongly preferred. Enabling it
      collapses cross-patient spread to ~0 and unifies the dominant archetype. Decisive.
    * factorized decoder (x_hat = h @ E.T + b + B): the low-rank latent path h = w @ Z can also
      represent the shift, so the fix is genuinely *soft* -- B partially absorbs the shift but W
      may still carry patient information. We assert the directional (not decisive) claim here.

Also asserts the regression guarantee: turning the offset off must not change the config
fingerprint that keys cached run directories.

Run:  python tools/verify_sample_offset.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CyEmbed.analysis import summarize_by_group
from CyEmbed.data import split_train_val_indices
from CyEmbed.train import _config_fingerprint, train_one_run
from CyEmbed.utils import validate_run_config

RNG = np.random.default_rng(0)
K_TRUE = 3
N_MARKERS = 20
N_PER_PATIENT = 800
N_PATIENTS = 3
SHIFT_SCALE = 6.0
# Shared composition skewed toward archetype 0, so the true dominant archetype is the same for
# every patient -- any per-patient difference in the recovered dominant archetype is leakage.
ALPHA = np.array([6.0, 3.0, 3.0])
PATIENT_NAMES = np.array([f"patient_{p}" for p in range(N_PATIENTS)])


def make_synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells from K_TRUE archetypes with an IDENTICAL composition per patient, plus a known
    constant per-patient shift. Returns (x, sample_ids, planted_shift[N_PATIENTS, M])."""
    archetypes = RNG.normal(0.0, 1.0, size=(K_TRUE, N_MARKERS)) * 3.0
    shift = RNG.normal(0.0, SHIFT_SCALE, size=(N_PATIENTS, N_MARKERS))
    x_blocks, sample_blocks = [], []
    for p in range(N_PATIENTS):
        w = RNG.dirichlet(ALPHA, size=N_PER_PATIENT)
        clean = w @ archetypes
        noise = RNG.normal(0.0, 0.05, size=clean.shape)
        x_blocks.append(clean + shift[p] + noise)
        sample_blocks.append(np.full(N_PER_PATIENT, PATIENT_NAMES[p]))
    x = np.concatenate(x_blocks, axis=0).astype(np.float32)
    sample_ids = np.concatenate(sample_blocks, axis=0)
    return x, sample_ids, shift


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


def base_config(model_type: str, decoder_type: str, use_offset: bool) -> dict:
    cfg = {
        "model_type": model_type,
        "decoder_type": decoder_type,
        "K": K_TRUE,
        "d": 8,
        "hidden_dims": [64, 32],
        "tau": 1.0,
        "lr": 5e-3,
        "epochs": 150,
        "batch_size": 256,
        "recon_loss_type": "mse",
        "weight_decay": 1e-4,
        "seed": 0,
        "early_stopping": False,
        "progress_epoch": False,
        "print_every": 1000,
        "deterministic": False,
    }
    if use_offset:
        cfg["use_sample_offset"] = True
    return cfg


def run_case(model_type: str, decoder_type: str, use_offset: bool, out_root: Path) -> dict:
    x, sample_ids, shift = make_synthetic()
    n = x.shape[0]
    # Stratify within patient so every patient appears in both halves (required once B is on).
    train_idx, val_idx = split_train_val_indices(
        n, val_fraction=0.2, seed=0, stratify_labels=sample_ids
    )
    run_id = f"{model_type}_{decoder_type}_{'on' if use_offset else 'off'}"
    train_one_run(
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
    result = {
        "spread": cross_patient_spread(w, sample_ids),
        "modal": modal_distinct(w, sample_ids),
        "b_corr": None,
        "b_zero_sum": None,
    }
    if use_offset:
        B = np.load(run_dir / "B.npy")  # saved already centred
        assert (run_dir / "sample_offset_levels.csv").exists(), "levels csv missing"
        levels = pd.read_csv(run_dir / "sample_offset_levels.csv")["sample_id"].to_numpy()
        order = [int(np.where(PATIENT_NAMES == lv)[0][0]) for lv in levels]
        shift_centered = shift[order] - shift[order].mean(0, keepdims=True)
        result["b_corr"] = float(np.corrcoef(B.ravel(), shift_centered.ravel())[0, 1])
        result["b_zero_sum"] = float(np.abs(B.mean(0)).max())
    return result


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
        header = f"{'case':<26}{'spread_off':>11}{'spread_on':>11}{'modal_off':>11}{'modal_on':>10}{'B~shift':>9}{'B_zero':>9}"
        print("\n" + header)
        for model_type in ("deterministic", "probabilistic"):
            for decoder_type in ("factorized", "direct"):
                off = run_case(model_type, decoder_type, False, out_root)
                on = run_case(model_type, decoder_type, True, out_root)
                tag = f"{model_type}/{decoder_type}"
                print(f"{tag:<26}{off['spread']:>11.4f}{on['spread']:>11.4f}"
                      f"{off['modal']:>11d}{on['modal']:>10d}{on['b_corr']:>9.3f}{on['b_zero_sum']:>9.1e}")

                # Invariants for both decoders.
                if not (on["b_zero_sum"] < 1e-4):
                    failures.append(f"{tag}: saved B not zero-sum (max|mean|={on['b_zero_sum']:.2e})")
                if not (on["b_corr"] > 0.4):
                    failures.append(f"{tag}: B does not positively recover shift (corr={on['b_corr']:.3f})")
                if not (on["spread"] <= off["spread"] + 0.05):
                    failures.append(f"{tag}: offset increased cross-patient spread "
                                    f"({on['spread']:.4f} > {off['spread']:.4f})")

                # The direct decoder (simplex-constrained w) strongly prefers B for the shift.
                if decoder_type == "direct" and not (on["b_corr"] > 0.9):
                    failures.append(f"{tag}: direct B recovers shift poorly (corr={on['b_corr']:.3f})")

                # Decisive claim -- cleanest, fully-identifiable config: deterministic + direct.
                # Here B absorbing the shift collapses cross-patient spread to ~0 and unifies the
                # dominant archetype. (Probabilistic sampling/KL and the factorized low-rank path
                # both make the fix soft, so we don't demand a collapse there.)
                if model_type == "deterministic" and decoder_type == "direct":
                    if not (on["spread"] < 0.5 * off["spread"]):
                        failures.append(f"{tag}: offset did not collapse spread "
                                        f"({on['spread']:.4f} vs {off['spread']:.4f})")
                    if not (on["modal"] <= off["modal"] and on["modal"] == 1):
                        failures.append(f"{tag}: offset did not unify dominant archetype "
                                        f"(modal off={off['modal']} on={on['modal']})")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
