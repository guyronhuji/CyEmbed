from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import load_json


def load_run_outputs(run_dir: str | Path) -> dict[str, Any]:
    """Load one saved run and reconstruct core analysis tables."""
    run_dir = Path(run_dir)
    outputs: dict[str, Any] = {
        "run_dir": run_dir,
        "config": load_json(run_dir / "config.json"),
        "summary_metrics": load_json(run_dir / "summary_metrics.json"),
        "history": pd.read_csv(run_dir / "history.csv"),
        "marker_names": pd.read_csv(run_dir / "marker_names.csv")["marker_name"].astype(str).tolist(),
        "cell_ids": pd.read_csv(run_dir / "cell_ids.csv")["cell_id"].astype(str).tolist(),
    }

    # Keep deterministic compatibility but also auto-load probabilistic arrays.
    for npy_path in sorted(run_dir.glob("*.npy")):
        key = npy_path.stem
        try:
            outputs[key] = np.load(npy_path)
        except Exception as exc:
            print(f"Warning: unable to load {npy_path.name}: {exc}")

    # Backward-compatible aliases for prior notebooks.
    if "X_observed" in outputs:
        outputs.setdefault("X", outputs["X_observed"])
    if "X_hat" in outputs and "X_observed" in outputs:
        outputs.setdefault("residuals", outputs["X_observed"] - outputs["X_hat"])

    if (run_dir / "sample_ids.csv").exists():
        outputs["sample_ids"] = pd.read_csv(run_dir / "sample_ids.csv")["sample_id"].to_numpy()
    else:
        outputs["sample_ids"] = None

    if (run_dir / "cluster_ids.csv").exists():
        outputs["cluster_ids"] = pd.read_csv(run_dir / "cluster_ids.csv")["cluster_id"].to_numpy()
    else:
        outputs["cluster_ids"] = None

    for arr_name in ("Z", "E", "b", "A"):
        path = run_dir / f"{arr_name}.npy"
        if path.exists():
            outputs[arr_name] = np.load(path)

    return outputs


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom < eps:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return _safe_pearson(x_rank, y_rank)


def per_marker_reconstruction_stats(
    x: np.ndarray,
    x_hat: np.ndarray,
    marker_names: list[str],
) -> pd.DataFrame:
    """Per-marker reconstruction statistics."""
    rows = []
    residual = x - x_hat
    for m, marker in enumerate(marker_names):
        pearson_r = _safe_pearson(x[:, m], x_hat[:, m])
        spearman_r = _safe_spearman(x[:, m], x_hat[:, m])
        ss_res = float(np.sum((x[:, m] - x_hat[:, m]) ** 2))
        ss_tot = float(np.sum((x[:, m] - np.mean(x[:, m])) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - (ss_res / ss_tot)
        rows.append(
            {
                "marker": marker,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "r2": r2,
                "mse": float(np.mean(residual[:, m] ** 2)),
                "mae": float(np.mean(np.abs(residual[:, m]))),
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def archetype_marker_rankings(a_hat: np.ndarray, marker_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """Top positive and negative markers per archetype."""
    rows: list[dict[str, Any]] = []
    for k in range(a_hat.shape[0]):
        row = a_hat[k]
        pos_idx = np.argsort(-row)[:top_n]
        neg_idx = np.argsort(row)[:top_n]
        for rank, idx in enumerate(pos_idx, start=1):
            rows.append(
                {
                    "archetype": k,
                    "direction": "positive",
                    "rank": rank,
                    "marker": marker_names[idx],
                    "value": float(row[idx]),
                }
            )
        for rank, idx in enumerate(neg_idx, start=1):
            rows.append(
                {
                    "archetype": k,
                    "direction": "negative",
                    "rank": rank,
                    "marker": marker_names[idx],
                    "value": float(row[idx]),
                }
            )
    return pd.DataFrame(rows)


def weight_entropy(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return -np.sum(w * np.log(w + eps), axis=1)


def dominant_assignments(w: np.ndarray, cell_ids: list[str] | None = None) -> pd.DataFrame:
    """Dominant archetype index and purity per cell."""
    n_cells = w.shape[0]
    ids = cell_ids if cell_ids is not None else [f"cell_{i}" for i in range(n_cells)]
    dom_idx = np.argmax(w, axis=1)
    dom_weight = np.max(w, axis=1)
    entropy = weight_entropy(w)
    return pd.DataFrame(
        {
            "cell_id": ids,
            "dominant_archetype": dom_idx,
            "dominant_weight": dom_weight,
            "entropy": entropy,
        }
    )


def purity_summary(w: np.ndarray, thresholds: tuple[float, ...] = (0.5, 0.8)) -> pd.DataFrame:
    dom_weight = np.max(w, axis=1)
    rows = [{"threshold": float(t), "fraction_cells": float(np.mean(dom_weight > t))} for t in thresholds]
    return pd.DataFrame(rows)


def summarize_by_group(
    w: np.ndarray,
    group_ids: np.ndarray,
    group_name: str = "group",
) -> dict[str, pd.DataFrame]:
    """Group-level mean weights and dominant archetype fractions."""
    df = pd.DataFrame(w)
    df[group_name] = group_ids
    mean_df = df.groupby(group_name, dropna=False).mean(numeric_only=True)
    mean_df.columns = [f"archetype_{c}" for c in mean_df.columns]
    mean_df = mean_df.reset_index()

    dom = np.argmax(w, axis=1)
    dom_df = pd.DataFrame({group_name: group_ids, "dominant_archetype": dom})
    frac_df = (
        dom_df.groupby([group_name, "dominant_archetype"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    total = frac_df.groupby(group_name)["n_cells"].transform("sum")
    frac_df["fraction"] = frac_df["n_cells"] / total
    return {"mean_weights": mean_df, "dominant_fractions": frac_df}


def cosine_similarity_matrix(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    x_norm = x / norm
    return x_norm @ x_norm.T


def pairwise_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def nearest_neighbors_from_similarity(
    sim: np.ndarray,
    names: list[str],
    k: int = 5,
) -> pd.DataFrame:
    """Nearest-neighbor table from pairwise similarity matrix."""
    if sim.shape[0] != len(names):
        raise ValueError("sim shape and names length mismatch.")
    rows = []
    for i, name in enumerate(names):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:k]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "query": name,
                    "rank": rank,
                    "neighbor": names[j],
                    "similarity": float(sim[i, j]),
                }
            )
    return pd.DataFrame(rows)


def pca_projection(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(n_components, u.shape[1])
    return u[:, :n_comp] * s[:n_comp]


def umap_projection(
    x: np.ndarray,
    *,
    n_neighbors: int = 10,
    min_dist: float = 0.3,
    random_state: int = 0,
) -> np.ndarray | None:
    """Optional UMAP projection if umap-learn is installed."""
    try:
        import umap  # type: ignore
    except Exception:
        return None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(np.asarray(x, dtype=np.float32))


def residual_summary(x: np.ndarray, x_hat: np.ndarray, marker_names: list[str]) -> pd.DataFrame:
    residual = x - x_hat
    rows = []
    for m, marker in enumerate(marker_names):
        r = residual[:, m]
        rows.append(
            {
                "marker": marker,
                "residual_mean": float(np.mean(r)),
                "residual_std": float(np.std(r)),
                "residual_mae": float(np.mean(np.abs(r))),
                "residual_mse": float(np.mean(r**2)),
            }
        )
    return pd.DataFrame(rows).sort_values("residual_mse", ascending=False).reset_index(drop=True)


def posterior_mean_weights(mu_w: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute posterior-mean simplex weights from mean logits."""
    tau = max(float(tau), 1e-6)
    logits = mu_w / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)


def residual_norms(r: np.ndarray) -> np.ndarray:
    """L2 norm of residual latent vectors per cell."""
    return np.linalg.norm(np.asarray(r, dtype=np.float32), axis=1)


def kl_history_columns(history_df: pd.DataFrame) -> list[str]:
    """List KL-related history columns if present."""
    return [c for c in history_df.columns if "kl" in c.lower()]
