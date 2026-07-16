"""Which criterion actually picks the right K and d? Tested against known ground truth.

Picking K is normally guesswork because you never know the answer. Here we do: the scRNA-seq
simulator in verify_sample_offset_scrna.py plants K_TRUE archetypes, so we can ask which
selection criterion recovers it -- rather than trusting folklore.

Two sweeps, both at 2000 HVGs on simulated NB counts -> Pearson residuals:

  d-sweep (K fixed at truth). Hypothesis from the algebra: A_hat = Z @ E.T + b with Z (K,d),
    E (M,d), so rank(A_hat) <= min(K, d). Once d >= K the rank constraint is INACTIVE and d
    cannot change what the model represents -- it only changes the E gene-embedding resolution,
    the optimisation path, and parameter count (M*d). Prediction: metrics degrade for d < K and
    plateau for d >= K.

  K-sweep (d fixed). Criteria scored against K_TRUE:
    * val_recon        -- expected to improve monotonically with K, i.e. NOT a selection
                          criterion. CyEmbed's run_sweep sorts by exactly this, which is a trap.
    * dead archetypes  -- archetypes used by <1% of cells; should appear once K exceeds truth.
    * archetype stability across seeds -- Hungarian-matched cosine between A_hat from independent
                          restarts. This is cNMF's consensus idea. CyEmbed has no such function.
    * w_recovery       -- only computable with ground truth, so it is the ORACLE here: it tells
                          us which of the usable criteria agrees with the truth.

Run:  python tools/select_k_and_d_scrna.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CyEmbed.data import split_train_val_indices
from CyEmbed.train import train_one_run

_spec = importlib.util.spec_from_file_location(
    "_scrna", Path(__file__).with_name("verify_sample_offset_scrna.py")
)
_scrna = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scrna)

make_scrna = _scrna.make_scrna
w_recovery = _scrna.w_recovery
K_TRUE = _scrna.K_TRUE

D_VALUES = (2, 3, 5, 8, 16, 32, 64)
K_VALUES = (3, 4, 5, 6, 7, 8)
SEEDS = (0, 1, 2)          # >=2 needed for the stability criterion


def cfg(K: int, d: int, seed: int) -> dict:
    return {
        "model_type": "deterministic",
        "decoder_type": "factorized",
        "K": K, "d": d,
        "hidden_dims": [256, 64],
        "tau": 1.0, "lr": 1e-3, "epochs": 400, "batch_size": 512,
        "recon_loss_type": "mse", "weight_decay": 1e-4,
        "seed": seed,                       # torch init/shuffle seed -- varied for stability
        "device": "cpu",
        "early_stopping": True, "patience": 30, "min_delta": 1e-4,
        "progress_epoch": False, "print_every": 100000, "deterministic": False,
        # The regulariser package the guide recommends; validated as the better setting.
        "logit_normalizer": "entmax", "entmax_alpha": 1.5,
        "lambda_entropy": 1e-3, "lambda_sep": 1e-3, "lambda_balance": 5e-2,
        "separation_mode": "cosine_sq", "balance_mode": "l2_uniform",
        "use_sample_offset": True,
    }


def fit(K, d, seed, data, out_root) -> dict:
    z, sample_ids, shift, w_true, _ = data
    n = z.shape[0]
    tr, va = split_train_val_indices(n, val_fraction=0.2, seed=0, stratify_labels=sample_ids)
    run_id = f"K{K}_d{d}_s{seed}"
    res = train_one_run(
        x_train=z[tr], x_val=z[va], x_full=z,
        marker_names=[f"g{i}" for i in range(z.shape[1])],
        cell_ids=[f"c{i}" for i in range(n)],
        run_config=cfg(K, d, seed), output_root=out_root, run_id=run_id,
        sample_ids=sample_ids, train_idx=tr, val_idx=va,
    )
    rd = out_root / run_id
    w = np.load(rd / "W.npy")
    a_hat = np.load(rd / "A_hat.npy")
    usage = w.mean(axis=0)
    return {
        "val_recon": float(res.history["val_recon"].min()),
        "w_rec": w_recovery(w, w_true),
        "dead": int((usage < 0.01).sum()),          # archetypes used by <1% of cells
        "A_hat": a_hat,
    }


def stability(a_list: list[np.ndarray]) -> float:
    """Mean Hungarian-matched cosine between archetype profiles from independent seeds.

    cNMF's consensus idea: at the right K, restarts find the same archetypes. Above it, the extra
    archetypes split arbitrarily and agreement falls. CyEmbed has no equivalent function.
    """
    def norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    scores = []
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            c = norm(a_list[i]) @ norm(a_list[j]).T
            r, cc = linear_sum_assignment(-c)
            scores.append(float(c[r, cc].mean()))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    data = make_scrna(0)
    print(f"K_TRUE = {K_TRUE}, 2000 HVGs, {data[0].shape[0]} cells\n")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        print("=== d-sweep (K fixed at truth). Does d matter above K? ===")
        print(f"{'d':>4}{'rank cap':>10}{'val_recon':>11}{'w_rec':>8}")
        for d in D_VALUES:
            r = fit(K_TRUE, d, 0, data, out)
            cap = min(K_TRUE, d)
            note = "  <-- d < K: rank-limited" if d < K_TRUE else ""
            print(f"{d:>4}{cap:>10}{r['val_recon']:>11.4f}{r['w_rec']:>8.3f}{note}")

        print(f"\n=== K-sweep (d=16). Which criterion finds K_TRUE={K_TRUE}? ===")
        print(f"{'K':>3}{'val_recon':>11}{'dead':>6}{'stability':>11}{'w_rec (oracle)':>16}")
        rows = []
        for K in K_VALUES:
            runs = [fit(K, 16, s, data, out) for s in SEEDS]
            row = {
                "K": K,
                "val_recon": float(np.mean([r["val_recon"] for r in runs])),
                "dead": float(np.mean([r["dead"] for r in runs])),
                "stab": stability([r["A_hat"] for r in runs]),
                "w_rec": float(np.mean([r["w_rec"] for r in runs])),
            }
            rows.append(row)
            print(f"{K:>3}{row['val_recon']:>11.4f}{row['dead']:>6.1f}"
                  f"{row['stab']:>11.3f}{row['w_rec']:>16.3f}")

        print("\n--- what each criterion picks ---")
        best_vr = min(rows, key=lambda r: r["val_recon"])["K"]
        best_stab = max(rows, key=lambda r: r["stab"])["K"]
        best_wrec = max(rows, key=lambda r: r["w_rec"])["K"]
        alive = [r for r in rows if r["dead"] < 0.5]
        best_dead = max(alive, key=lambda r: r["K"])["K"] if alive else None
        print(f"  val_recon (what run_sweep sorts by) picks K={best_vr}   truth={K_TRUE}")
        print(f"  largest K with no dead archetypes   picks K={best_dead}   truth={K_TRUE}")
        print(f"  archetype stability across seeds    picks K={best_stab}   truth={K_TRUE}")
        print(f"  w_recovery (oracle)                 picks K={best_wrec}   truth={K_TRUE}")


if __name__ == "__main__":
    main()
